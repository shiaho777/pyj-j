#!/usr/bin/env python3
"""
cowpaper.py -- the B-line money measurement: our solver vs production,
on REAL settled CoW Protocol auctions.

No API access needed: settlements are on-chain. The GPv2Settlement
contract emits Settlement(solver); the tx calldata IS the full
production solution (tokens, clearing prices, trades with executed
amounts, AMM interaction counts). We decode it with the same pure-Python
ABI decoder validated in battle 4, rebuild the order book, and run OUR
solver (bilateral + 3-cycles, peer-only v1) on the identical book.
Both solutions go through the identical integer judge; both are scored
with the identical surplus function.

The honest caveats, stated up front:
  1. We see the orders production CHOSE to settle, not the full auction
     (unmatched orders leave no on-chain trace without API access).
     So this measures: given the same book, does our solver extract
     more user surplus than the professional winner did?
  2. Production often routes through AMM interactions (counted and
     reported separately). Our v1 is peer-only, so on AMM-heavy
     settlements we are handicapped by construction -- that subset is
     reported apart from the peer-comparable subset.
  3. Production surplus here = users' surplus; solver margin is the
     difference between limit and clearing, which production sets by
     uniform clearing price -- we score the same way.
"""
import sys
from fractions import Fraction

sys.path.insert(0, ".")
from solver.cow import Order, user_surplus, validate_settlement  # noqa: E402
from solver.mainnet import rpc, SETTLEMENT, _dyn, judge_trade    # noqa: E402
from solver.tournament import solve_exact                        # noqa: E402

SETTLEMENT_TOPIC = ("0x40338ce1a7c49204f0099533b1e9a7ee0a3d261f8497"
                    "4ab7af36105b8c4e9db4")


def fetch_settlements(n_blocks=120):
    """Recent settlement txs via address-scoped getLogs. publicnode
    rate-limits getLogs (HTTP 403 under load); drpc serves address-scoped
    queries within its ~128-block horizon, which is exactly our window."""
    from solver.scan import rpc_post, CHAINS
    head = int(rpc("eth_blockNumber", []), 16)
    logs = rpc_post(CHAINS["eth"]["logs"], "eth_getLogs", [{
        "fromBlock": hex(head - n_blocks), "toBlock": hex(head),
        "address": SETTLEMENT, "topics": [SETTLEMENT_TOPIC]}])
    txs = []
    for l in logs:
        txs.append((l["transactionHash"], int(l["blockNumber"], 16)))
    return txs, head


def decode_settlement(txh):
    tx = rpc("eth_getTransactionByHash", [txh])
    if not tx or not tx.get("input"):
        return None
    data = bytes.fromhex(tx["input"][2:])
    if len(data) < 4 or data[:4].hex() != "13d79a0b":
        return None
    try:
        tokens, prices, trades, n_inter = _dyn(data[4:])
    except Exception:                                     # noqa: BLE001
        return None
    return dict(tokens=tokens, prices=prices, trades=trades,
                n_inter=sum(n_inter))


def production_fill(trade, prices):
    """Production's (fill, buy) per trade, exactly as the contract
    computes it. Sell orders: executed is the sell fill, buy =
    ceil(f*ps/pb). Buy orders: executed is the buy fill, sell =
    floor(f*pb/ps) (rounded in the user's favor)."""
    f = trade["executed"]
    ps = prices[trade["sell_i"]]
    pb = prices[trade["buy_i"]]
    if trade["flags"] & 1:            # buy order
        buys = f
        fills = f * pb // ps
    else:                              # sell order
        fills = f
        _, buys, _ = judge_trade(trade, prices)
    return fills, buys


def build_orders(dec):
    orders = []
    for i, t in enumerate(dec["trades"]):
        orders.append(Order(i, dec["tokens"][t["sell_i"]],
                            dec["tokens"][t["buy_i"]],
                            t["sell_amt"], t["buy_amt"]))
    return orders


def main(n_blocks=120):
    txs, head = fetch_settlements(n_blocks)
    print(f"settlements in last {n_blocks} blocks (head {head}): {len(txs)}")
    rows = []
    n_decode_fail = n_judge_fail = 0
    for txh, block in txs:
        dec = decode_settlement(txh)
        if dec is None or not dec["trades"]:
            n_decode_fail += 1
            continue
        orders = build_orders(dec)
        # production solution: per-order LIMIT checks only. Full
        # conservation does not apply to production settlements -- their
        # token balance includes AMM interaction legs (the pool absorbs
        # the user-side imbalance), which is exactly why the chain
        # accepted them. Battle 4 validated the limit core this way.
        prod_f, prod_b = {}, {}
        prod_ok = True
        for i, t in enumerate(dec["trades"]):
            ok_i, _, _ = judge_trade(t, dec["prices"])
            if not ok_i:
                prod_ok = False
                break
            f, b = production_fill(t, dec["prices"])
            if f <= 0:
                continue
            prod_f[i], prod_b[i] = f, b
        if not prod_ok:
            n_judge_fail += 1
            continue
        # our solution (peer-only v1)
        our_f, our_b = solve_exact(orders)
        ok2, _ = validate_settlement(orders, our_f, our_b)
        if not ok2:
            print(f"!! OUR SOLVER INVALID on real book {txh[:16]}… -- BUG")
            continue
        prod_sup = user_surplus(orders, prod_f, prod_b)
        our_sup = user_surplus(orders, our_f, our_b)
        prod_vol = sum(prod_f.values())
        our_vol = sum(our_f.values())
        rows.append(dict(tx=txh, n_orders=len(orders),
                         amm=dec["n_inter"] > 0,
                         prod_sup=prod_sup, our_sup=our_sup,
                         prod_vol=prod_vol, our_vol=our_vol))

    print(f"decoded: {len(rows) + n_judge_fail} usable, "
          f"{n_decode_fail} skipped, {n_judge_fail} judge-failed "
          f"(decode bug alarm if >0)")
    if not rows:
        return
    peer = [r for r in rows if not r["amm"]]
    ammr = [r for r in rows if r["amm"]]

    def summarize(name, rs):
        if not rs:
            print(f"\n{name}: none")
            return
        wins = sum(1 for r in rs if r["our_sup"] > r["prod_sup"])
        ties = sum(1 for r in rs if r["our_sup"] == r["prod_sup"])
        losses = sum(1 for r in rs if r["our_sup"] < r["prod_sup"])
        tp = sum(r["prod_sup"] for r in rs)
        to = sum(r["our_sup"] for r in rs)
        vp = sum(r["prod_vol"] for r in rs)
        vo = sum(r["our_vol"] for r in rs)
        print(f"\n{name}: {len(rs)} settlements")
        print(f"  surplus  win/tie/loss: {wins}/{ties}/{losses}")
        print(f"  total user surplus: ours {float(to):.4e} vs "
              f"production {float(tp):.4e}  "
              f"({float(to / tp * 100) if tp else 0:.1f}% of production)")
        print(f"  total matched volume: ours {vo:,} vs "
              f"production {vp:,} ({(vo / vp * 100) if vp else 0:.1f}%)")

    print("=" * 70)
    print("  OUR SOLVER vs PRODUCTION -- real settled CoW auctions")
    print("=" * 70)
    summarize("peer-comparable (no AMM legs)", peer)
    summarize("AMM-heavy (we are peer-only v1)", ammr)
    summarize("ALL", rows)
    print("=" * 70)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
