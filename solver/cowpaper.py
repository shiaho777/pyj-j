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
import time
from fractions import Fraction

sys.path.insert(0, ".")
from solver.amm import V2Pool, V3Pool                  # noqa: E402
from solver.cow import Order, user_surplus, validate_settlement  # noqa: E402
from solver.mainnet import rpc, SETTLEMENT, _dyn, judge_trade    # noqa: E402
from solver.tickwalk import tick_of                     # noqa: E402
from solver.tournament import solve_exact               # noqa: E402
from solver.v3math import get_sqrt_ratio_at_tick        # noqa: E402

SETTLEMENT_TOPIC = ("0x40338ce1a7c49204f0099533b1e9a7ee0a3d261f8497"
                    "4ab7af36105b8c4e9db4")
V2_FACTORY = "0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f"
V3_FACTORY = "0x1f98431c8ad98523631ae4a59f267346ea31f984"
PANCAKE_V3_FACTORY = "0x0bfbcf9fa4f9c56b0f40a671ad40e0805a091865"
V3_FEES = (100, 500, 3000, 10000)   # all canonical tiers: long-tail
                                    # pairs live at 10000 (1%)
ZERO_ADDR = "0x" + "0" * 40


def _pad(addr):
    return addr[2:].lower().rjust(64, "0")


def call_at(block, to, data):
    from solver.scan import rpc_retry, CHAINS
    tag = block if isinstance(block, str) else hex(block)
    return rpc_retry(CHAINS["eth"]["logs"], "eth_call",
                     [{"to": to, "data": data}, tag])


# hubs for two-hop discovery (lazy: only queried for orders whose direct
# pair has no pool). WETH/USDC/USDT/DAI/WBTC cover most routing paths.
HUBS = (
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",   # WETH
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",   # USDC
    "0xdac17f958d2ee523a2206206994597c13d831ec7",   # USDT
    "0x6b175474e89094c44da98b954edeac495271d0f",   # DAI
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",   # WBTC
)
V3_FACTORIES = (V3_FACTORY, PANCAKE_V3_FACTORY)


def v3_bounds(pool, sqrtP, block):
    """Nearest initialized tick ratios below/above the current price,
    from the tick bitmap at `block` (4-word scan ~= +/-512 ticks; the
    window edge is the conservative fallback)."""
    t = tick_of(sqrtP)
    w0 = t >> 8
    lo_t = hi_t = None
    for w in range(w0, w0 - 3, -1):
        bm = int(call_at(block, pool, "0x5339c296"
                         + (w & 0xFFFF).to_bytes(32, "big").hex()), 16)
        for i in range(255, -1, -1):
            tt = w * 256 + i
            if tt <= t and (bm >> i) & 1:
                lo_t = tt
                break
        if lo_t is not None:
            break
    for w in range(w0, w0 + 3):
        bm = int(call_at(block, pool, "0x5339c296"
                         + (w & 0xFFFF).to_bytes(32, "big").hex()), 16)
        for i in range(256):
            tt = w * 256 + i
            if tt > t and (bm >> i) & 1:
                hi_t = tt
                break
        if hi_t is not None:
            break
    if lo_t is None:
        lo_t = t - 512
    if hi_t is None:
        hi_t = t + 512
    return get_sqrt_ratio_at_tick(lo_t), get_sqrt_ratio_at_tick(hi_t)


_ADDR_CACHE = {}     # frozenset(pair) -> [(kind, addr, t0, t1), ...]
                     # addresses are stable; STATE is always read fresh


def discover_pair_addrs(t0, t1):
    """ALL canonical candidates for the pair (cached): V3 across both
    factories and all four tiers (existence-filtered by L>0 at latest),
    plus the V2 pair. The block-state probe picks the winner -- latest-L
    ranking alone misselects (a deep tier at latest may be the wrong
    venue for this order at this block)."""
    key = frozenset((t0, t1))
    if key in _ADDR_CACHE:
        return _ADDR_CACHE[key]
    cands = []
    for fac in V3_FACTORIES:
        for fee in V3_FEES:
            try:
                r = call_at("latest", fac, "0x1698ee82"
                            + _pad(t0) + _pad(t1)
                            + hex(fee)[2:].rjust(64, "0"))
                p3 = "0x" + r[2:][-40:]
                if p3 == ZERO_ADDR:
                    continue
                L = int(call_at("latest", p3, "0x1a686502"), 16)
                if L > 0:
                    cands.append(("v3", p3, t0, t1))
            except RuntimeError:
                continue
    try:
        r = call_at("latest", V2_FACTORY,
                    "0xe6a43905" + _pad(t0) + _pad(t1))
        pair = "0x" + r[2:][-40:]
        if pair != ZERO_ADDR:
            cands.append(("v2", pair, t0, t1))
    except RuntimeError:
        pass
    _ADDR_CACHE[key] = cands
    return cands


def fetch_pool_state(found, block):
    """Pool state AT `block` (the settlement's before-state)."""
    if found is None:
        return None
    kind, addr, t0, t1 = found
    try:
        if kind == "v2":
            res = call_at(block, addr, "0x0902f1ac")[2:]
            r0, r1 = int(res[0:64], 16), int(res[64:128], 16)
            if r0 <= 0 or r1 <= 0:
                return None
            return V2Pool(t0, t1, r0, r1)
        slot0 = call_at(block, addr, "0x3850c7bd")[2:]
        sqrtP = int(slot0[0:64], 16)
        L = int(call_at(block, addr, "0x1a686502"), 16)
        if L <= 0:
            return None
        lo, hi = v3_bounds(addr, sqrtP, block)
        return V3Pool(t0, t1, sqrtP, L, lo, hi)
    except RuntimeError:
        return None


def _depth_at_block(found, block):
    """Cheap depth read (1 call) for candidate ranking: V3 -> liquidity,
    V2 -> reserve product. Units differ across kinds, so ranking is
    within-kind only."""
    kind, addr, _t0, _t1 = found
    try:
        if kind == "v2":
            res = call_at(block, addr, "0x0902f1ac")[2:]
            r0, r1 = int(res[0:64], 16), int(res[64:128], 16)
            return r0 * r1 if r0 > 0 and r1 > 0 else 0
        return int(call_at(block, addr, "0x1a686502"), 16)
    except RuntimeError:
        return 0


def _best_candidate(cands, block, probe_tok, probe_amt):
    """Two-phase venue selection: cheap depth reads rank candidates
    within each kind, then the top of each kind is fully built and
    probe-quoted -- selection by execution, at bounded rpc cost."""
    if not cands:
        return None
    depths = [(c, _depth_at_block(c, block)) for c in cands]
    v3 = [c for c, d in depths if c[0] == "v3" and d > 0]
    v2 = [c for c, d in depths if c[0] == "v2" and d > 0]
    v3.sort(key=lambda c: -dict((id(cc), dd) for cc, dd in depths)[id(c)])
    v2.sort(key=lambda c: -dict((id(cc), dd) for cc, dd in depths)[id(c)])
    finalists = v3[:2] + v2[:1]
    if not finalists:
        return None
    best_q, best_p = None, None
    for found in finalists:
        p = fetch_pool_state(found, block)
        if p is None:
            continue
        if probe_tok is None or probe_amt <= 0:
            if best_p is None:
                best_p = p
            continue
        try:
            q = p.quote(probe_tok, probe_amt)
        except Exception:                              # noqa: BLE001
            continue
        if q > 0 and (best_q is None or q > best_q):
            best_q, best_p = q, p
    return best_p


def discover_pools(orders, block):
    """Direct pairs first (probe = the pair's largest order); hub pairs
    lazily, only for orders whose direct pair has no pool."""
    pools = {}
    uncovered = []
    probe_by_pair = {}
    for o in orders:
        key = frozenset((o.sell_tok, o.buy_tok))
        cur = probe_by_pair.get(key)
        if cur is None or o.sell_amt > cur[1]:
            probe_by_pair[key] = (o.sell_tok, o.sell_amt)
    for o in orders:
        key = frozenset((o.sell_tok, o.buy_tok))
        if key in pools:
            continue
        cands = discover_pair_addrs(*sorted(key))
        ptok, pamt = probe_by_pair[key]
        p = _best_candidate(cands, block, ptok, pamt)
        if p is not None:
            pools[key] = p
        else:
            uncovered.append(o)
    for o in uncovered:
        for h in HUBS:
            if h == o.sell_tok or h == o.buy_tok:
                continue
            for (a, b, probe) in ((o.sell_tok, h, (o.sell_tok, o.sell_amt)),
                                  (h, o.buy_tok, (h, 0))):
                key = frozenset((a, b))
                if key in pools:
                    continue
                cands = discover_pair_addrs(*sorted(key))
                p = _best_candidate(cands, block, *probe)
                if p is not None:
                    pools[key] = p
    return pools


def fetch_settlements(n_blocks=120):
    """Recent settlement txs via address-scoped getLogs. publicnode
    rate-limits getLogs (HTTP 403 under load); drpc serves address-scoped
    queries within its ~128-block horizon, which is exactly our window."""
    from solver.scan import rpc_post, rpc_retry, CHAINS
    head = int(rpc_retry(CHAINS["eth"]["logs"], "eth_blockNumber", []), 16)
    logs = rpc_post(CHAINS["eth"]["logs"], "eth_getLogs", [{
        "fromBlock": hex(head - n_blocks), "toBlock": hex(head),
        "address": SETTLEMENT, "topics": [SETTLEMENT_TOPIC]}])
    txs = []
    for l in logs:
        txs.append((l["transactionHash"], int(l["blockNumber"], 16)))
    return txs, head


def decode_settlement(txh):
    from solver.scan import rpc_retry, CHAINS
    tx = rpc_retry(CHAINS["eth"]["logs"], "eth_getTransactionByHash", [txh])
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


def log_measurement(rows, head):
    import datetime, json, os
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scanlog")
    os.makedirs(d, exist_ok=True)
    rec = dict(ts=datetime.datetime.utcnow().isoformat() + "Z",
               head=head, settlements=len(rows),
               wins=sum(1 for r in rows if r["our_sup"] > r["prod_sup"]),
               full_matches=sum(1 for r in rows
                                if r["prod_vol"] and
                                r["our_vol"] * 100 // r["prod_vol"] >= 99),
               avg_pools=(sum(r["n_pools"] for r in rows) / len(rows)
                          if rows else 0),
               rows=[dict(tx=r["tx"], orders=r["n_orders"],
                          pools=r["n_pools"],
                          vol_ratio=(r["our_vol"] / r["prod_vol"]
                                     if r["prod_vol"] else None),
                          sup_ratio=(float(r["our_sup"] / r["prod_sup"])
                                     if r["prod_sup"] else None))
                     for r in rows])
    with open(os.path.join(d, "cowpaper.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def measure(n_blocks=100, quiet=False):
    """Run the measurement, returning rich per-settlement rows (orders,
    decoded settlement, both solutions) for downstream money models."""
    txs, head = fetch_settlements(n_blocks)
    if not quiet:
        print(f"settlements in last {n_blocks} blocks (head {head}): {len(txs)}")
    rows = []
    n_decode_fail = n_judge_fail = 0
    for txh, block in sorted(txs, key=lambda tb: -tb[1]):
      try:
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
        # our solution: peers + AMM routing through pools discovered
        # from the canonical factories, state read at the settlement block
        pools = None
        try:
            pools = discover_pools(orders, block - 1)
        except RuntimeError:
            pools = None
        our_f, our_b = solve_exact(orders, pools)
        ok2, _ = validate_settlement(orders, our_f, our_b, pools)
        if not ok2:
            print(f"!! OUR SOLVER INVALID on real book {txh[:16]}… -- BUG")
            continue
        prod_sup = user_surplus(orders, prod_f, prod_b)
        our_sup = user_surplus(orders, our_f, our_b)
        prod_vol = sum(prod_f.values())
        our_vol = sum(our_f.values())
        rows.append(dict(tx=txh, n_orders=len(orders),
                         amm=dec["n_inter"] > 0,
                         n_pools=len(pools or {}),
                         prod_sup=prod_sup, our_sup=our_sup,
                         prod_vol=prod_vol, our_vol=our_vol,
                         orders=orders, dec=dec,
                         our_f=our_f, our_b=our_b,
                         prod_f=prod_f, prod_b=prod_b))
      except Exception as e:                          # noqa: BLE001
        if not quiet:
            print(f"  [skip] {txh[:16]}… {type(e).__name__}: {str(e)[:70]}")
        n_decode_fail += 1

    if not quiet:
        print(f"decoded: {len(rows) + n_judge_fail} usable, "
              f"{n_decode_fail} skipped, {n_judge_fail} judge-failed "
              f"(decode bug alarm if >0)")
    return rows, head


def main(n_blocks=100):
    rows, head = measure(n_blocks)
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
        print(f"\n{name}: {len(rs)} settlements "
              f"(avg pools discovered: "
              f"{sum(r.get('n_pools', 0) for r in rs) / len(rs):.1f})")
        print(f"  surplus  win/tie/loss: {wins}/{ties}/{losses}")
        print(f"  total user surplus: ours {float(to):.4e} vs "
              f"production {float(tp):.4e}  "
              f"({float(to / tp * 100) if tp else 0:.1f}% of production)")
        print(f"  total matched volume: ours {vo:,} vs "
              f"production {vp:,} ({(vo / vp * 100) if vp else 0:.1f}%)")

    print("=" * 70)
    print("  OUR SOLVER vs PRODUCTION -- real settled CoW auctions")
    print("=" * 70)
    # per-settlement table: surplus ratios are only unit-clean within a
    # settlement (and exactly clean for single-order books); cross-
    # settlement surplus sums mix token units and are NOT trustworthy
    print(f"  {'tx':16} {'ord':>3} {'pool':>4} {'vol ratio':>10} "
          f"{'sup ratio':>10}")
    for r in sorted(rows, key=lambda r: -(r["our_sup"] / r["prod_sup"]
                                          if r["prod_sup"] else 0)):
        vr = (r["our_vol"] / r["prod_vol"] * 100) if r["prod_vol"] else 0
        sr = (float(r["our_sup"] / r["prod_sup"]) * 100
              if r["prod_sup"] else float("inf"))
        print(f"  {r['tx'][:16]}… {r['n_orders']:>3} {r['n_pools']:>4} "
              f"{vr:>9.1f}% {sr:>9.1f}%")
    summarize("peer-comparable (no AMM legs)", peer)
    summarize("AMM-heavy", ammr)
    log_measurement(rows, head)
    print("\n  NOTE: aggregate surplus sums mix token units across")
    print("  settlements -- per-settlement ratios above are the signal.")
    print("=" * 70)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
