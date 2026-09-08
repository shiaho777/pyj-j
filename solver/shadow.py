#!/usr/bin/env python3
"""
shadow.py -- shadow-run: replay REAL CoW auctions through our exact solver.

The simulation-fidelity ladder ("as good as real money"):
  v1 (this file):
    1. Pull real settlements from mainnet (recent blocks).
    2. Extract the order set: (sellTok, buyTok, sellAmt, buyMin, executed).
    3. REPLAY: with the production clearing prices, run our exact integer
       machinery (execBuy = ceil(f*ps/pb), limit check b*S >= B*f) and
       verify we reproduce the production solution's validity and fills.
    4. PRICE EXPERIMENT: for ring-matchable order pairs, compute our own
       price split (midpoint of the two implied limit prices) and compare
       total surplus and its distribution vs production.

Everything is integer/Fraction arithmetic -- the same stack that validated
0 disagreements against the production contract (mainnet.py).
"""
import sys
from fractions import Fraction

sys.path.insert(0, ".")
from solver.mainnet import rpc, _dyn, SETTLEMENT  # noqa: E402


# ------------------------------------------------------------- extraction

def extract_orders(txh):
    """Decode a settlement into (tokens, prices, orders, receipt_ok).

    Order = dict(sell_i, buy_i, S, B, f, kind) where f is the executed
    sell amount. kind: 'full' (fill-or-kill, f == S) or 'partial'.
    """
    tx = rpc("eth_getTransactionByHash", [txh])
    rc = rpc("eth_getTransactionReceipt", [txh])
    ok = int(rc["status"], 16) == 1
    tokens, prices, trades, n_inter = _dyn(bytes.fromhex(tx["input"][10:]))
    orders = []
    for t in trades:
        orders.append(dict(
            sell_i=t["sell_i"], buy_i=t["buy_i"],
            S=t["sell_amt"], B=t["buy_amt"], f=t["executed"],
            kind="full" if t["executed"] == t["sell_amt"] else "partial"))
    return tokens, prices, orders, ok, n_inter


# ------------------------------------------------------------- exact core

def exec_buy(f, ps, pb):
    """CoW contract semantics: ceil(f * priceSell / priceBuy)."""
    return -((-f * ps) // pb)


def judge(orders, prices):
    """Integer judge: every filled order must satisfy b*S >= B*f."""
    for o in orders:
        if o["f"] <= 0:
            continue
        ps, pb = prices[o["sell_i"]], prices[o["buy_i"]]
        b = exec_buy(o["f"], ps, pb)
        if b * o["S"] < o["B"] * o["f"]:
            return False, (o, b)
    return True, None


def surplus(orders, prices):
    """Total user surplus per order, in sell-token units (Fraction)."""
    out = []
    for o in orders:
        if o["f"] <= 0:
            out.append(Fraction(0))
            continue
        ps, pb = prices[o["sell_i"]], prices[o["buy_i"]]
        b = exec_buy(o["f"], ps, pb)
        out.append(Fraction(b * o["S"] - o["B"] * o["f"], o["S"]))
    return out


# ------------------------------------------------------------- ring match

def ring_match(o1, o2, tokens):
    """Two orders that close a ring (o1 sells T1 buys T2; o2 sells T2
    buys T1), compared by token ADDRESS (the same token can appear at
    several indices in the tokens array).

    Returns the CONSERVATION-PINNED analysis: the ring's total user
    surplus is conserved -- price choice only redistributes it between
    the two users (and the solver's margin). We report the spread and
    the max-user-surplus split.
    """
    if (tokens[o1["sell_i"]] != tokens[o2["buy_i"]]
            or tokens[o1["buy_i"]] != tokens[o2["sell_i"]]):
        return None
    lo = Fraction(o1["B"], o1["S"])          # T2 per T1, o1's minimum
    hi = Fraction(o2["S"], o2["B"])         # T2 per T1, o2's maximum
    if lo > hi:
        return None

    f1, f2 = o1["S"], o2["S"]
    # max-user split: direct exchange, o1 receives o2's entire f2,
    # o2 receives exactly f1 (at its limit)
    return dict(
        lo=lo, hi=hi, f1=f1, f2=f2,
        # total T2 that o1 can receive above its minimum B1
        max_user_surplus_t2=f2 - o1["B"],
        # feasibility: o2 accepts f1 for all its f2  <=>  f1*S2 >= B2*f2
        max_split_ok=(f1 * o2["S"] >= o2["B"] * f2))


# ------------------------------------------------------------- shadow run

def shadow_one(txh):
    tokens, prices, orders, ok, n_inter = extract_orders(txh)
    rep = dict(hash=txh, onchain_ok=ok, n_orders=len(orders),
               kinds=[o["kind"] for o in orders],
               tokens=len(set(tokens)))

    # 1. replay with production prices: validity + surplus
    valid, bad = judge(orders, prices)
    rep["replay_valid"] = valid
    sup = surplus(orders, prices)
    rep["prod_surplus"] = [str(s) for s in sup]

    # 2. ring experiment: if exactly two orders form a ring, try our
    #    midpoint price and compare
    exp = None
    if len(orders) == 2:
        r = ring_match(orders[0], orders[1], tokens)
        if r:
            exp = r
    rep["ring"] = exp
    return rep


def main(n_settlements=8):
    head = int(rpc("eth_blockNumber", []), 16)
    txs = []
    for b in range(head - 80, head):
        blk = rpc("eth_getBlockByNumber", [hex(b), True]) or {}
        for tx in blk.get("transactions", []):
            if (tx.get("to") or "").lower() == SETTLEMENT:
                txs.append(tx["hash"])
        if len(txs) >= n_settlements:
            break
    print(f"shadow-running {len(txs)} real settlements "
          f"(blocks {head-80}..{head})\n")

    n_replay_ok = 0
    n_rings = 0
    n_ring_better = 0
    for txh in txs:
        r = shadow_one(txh)
        mark = "OK " if r["replay_valid"] else "BAD"
        n_replay_ok += 1 if r["replay_valid"] else 0
        print(f"[{mark}] {r['hash'][:20]}… orders={r['n_orders']} "
              f"kinds={r['kinds']} tokens={r['tokens']} "
              f"onchain={'success' if r['onchain_ok'] else 'REVERTED'}")
        print(f"     production surplus (sell-token units): "
              f"{r['prod_surplus']}")
        if r["ring"]:
            n_rings += 1
            ring = r["ring"]
            print(f"     RING: price spread [{float(ring['lo']):.6g}, "
                  f"{float(ring['hi']):.6g}] T2-per-T1")
            print(f"       max user surplus (all-to-users split): "
                  f"{ring['max_user_surplus_t2']:,} T2-units  "
                  f"feasible={ring['max_split_ok']}")
        print()

    print("=" * 60)
    print(f"replay validity (our judge, production prices): "
          f"{n_replay_ok}/{len(txs)}")
    print(f"ring-matchable settlements analyzed: {n_rings}")
    print("=" * 60)


if __name__ == "__main__":
    main()
