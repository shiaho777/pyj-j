#!/usr/bin/env python3
"""
arbscan.py -- retrospective cross-pool arbitrage on the native archive.

The B-line's first measurement against REALITY: the archive gives us the
state timeline of every pool we captured, so we can reconstruct the
simultaneous state of any two pools sharing a token pair at any block
boundary -- and ask the only question that matters for the
latency-free edge:

    how often, and how big, are price discrepancies that SURVIVE the
    block boundary? (Intrablock discrepancies are MEV-bot territory;
    boundary-survivors are batch-window territory -- where math wins,
    not speed.)

Everything is exact integer math (the same three execution semantics
the differential scanner validated). Fees are the recorded first-sight
values; gas is NOT included (Base: ~$0.001-0.01; noted in the report).
"""
import os
import sqlite3
import sys
from collections import defaultdict

sys.path.insert(0, ".")
from solver.v3math import (                               # noqa: E402
    Q96, get_amount0_delta, get_amount1_delta,
    get_next_sqrt_price_from_amount0_rounding_up,
    get_next_sqrt_price_from_amount1_rounding_down)

FEE_ONE = 1_000_000
HERE = os.path.dirname(os.path.abspath(__file__))
STORE_PATH = os.path.join(HERE, "scanstore.db")

PROBE_FRACS = (10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1)


def swap_exact_in(sqrtP, L, amount_in, zf1):
    """exactInput single-range swap: returns (out, sqrtP_next)."""
    if zf1:
        s_next = get_next_sqrt_price_from_amount0_rounding_up(
            sqrtP, L, amount_in)
        out = get_amount1_delta(sqrtP, s_next, L, False)
    else:
        s_next = get_next_sqrt_price_from_amount1_rounding_down(
            sqrtP, L, amount_in)
        out = get_amount0_delta(sqrtP, s_next, L, False)
    return out, s_next


def round_trip(x, s_cheap, L_cheap, f_cheap, s_rich, L_rich, f_rich):
    """x token0 -> cheap pool (T0 in) -> token1 -> rich pool (T1 in)
    -> token0'. Returns token0'."""
    net = x * (FEE_ONE - f_cheap) // FEE_ONE
    y, _ = swap_exact_in(s_cheap, L_cheap, net, True)
    net2 = y * (FEE_ONE - f_rich) // FEE_ONE
    x2, _ = swap_exact_in(s_rich, L_rich, net2, False)
    return x2


def load_archive(chain):
    db = sqlite3.connect(STORE_PATH)
    meta = {}
    for pool, tok0, tok1, fee in db.execute(
            "SELECT pool, token0, token1, fee FROM pool_meta "
            "WHERE chain=? AND token0 IS NOT NULL", (chain,)):
        meta[pool] = (tok0, tok1, fee)
    timelines = defaultdict(list)          # pool -> [(block, sqrtP, L)]
    for pool, block, sp, liq in db.execute(
            "SELECT pool, block, sqrt_p, liq FROM events "
            "WHERE chain=? AND kind='swap' ORDER BY pool, block, log_index",
            (chain,)):
        if pool in meta and sp and liq:
            timelines[pool].append((block, int(sp), int(liq)))
    return meta, timelines


def pair_groups(meta):
    groups = defaultdict(list)
    for pool, (t0, t1, _f) in meta.items():
        groups[(min(t0, t1), max(t0, t1))].append(pool)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def main():
    chain = sys.argv[1] if len(sys.argv) > 1 else "base"
    meta, timelines = load_archive(chain)
    groups = pair_groups(meta)
    print(f"[{chain}] archive pools with meta: {len(meta)}; "
          f"same-token-pair groups (>=2 pools): {len(groups)}")
    for (t0, t1), pools in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  {t0[:10]}…/{t1[:10]}…: {len(pools)} pools")

    if not groups:
        print("no cross-pool pairs captured in this archive window yet")
        return

    # block range
    blocks = sorted({b for tl in timelines.values() for b, _, _ in tl})
    lo, hi = blocks[0], blocks[-1]

    # walk block boundaries, maintaining per-pool current state
    state = {}                             # pool -> (sqrtP, L) at boundary
    events_at = defaultdict(list)
    for pool, tl in timelines.items():
        for b, sp, L in tl:
            events_at[b].append((pool, sp, L))

    pair_cache = {}
    for pair in groups.values():
        for i in range(len(pair)):
            for j in range(i + 1, len(pair)):
                pair_cache[(pair[i], pair[j])] = True

    opps = []                               # (block, poolA, poolB, ratio, x_best, x2_best)
    for b in range(lo, hi + 1):
        for pool, sp, L in events_at.get(b, ()):
            state[pool] = (sp, L)
        for (pa, pb) in pair_cache:
            if pa not in state or pb not in state:
                continue
            sa, La = state[pa]
            sb, Lb = state[pb]
            fa, fb = meta[pa][2], meta[pb][2]
            if fa is None or fb is None or La <= 0 or Lb <= 0:
                continue
            price_a, price_b = (sa * sa) // (Q96 * Q96), (sb * sb) // (Q96 * Q96)
            if price_a == price_b:
                continue
            # cheap pool = lower price; buy T1 cheap, sell rich
            if price_a < price_b:
                cheap, rich = (pa, sa, La, fa), (pb, sb, Lb, fb)
            else:
                cheap, rich = (pb, sb, Lb, fb), (pa, sa, La, fa)
            # marginal band check (fees kill sub-band discrepancies)
            p_c = (cheap[1] * cheap[1]) / (Q96 * Q96)
            p_r = (rich[1] * rich[1]) / (Q96 * Q96)
            if p_r / p_c < 1.0 / ((1 - cheap[3] / FEE_ONE)
                                  * (1 - rich[3] / FEE_ONE)):
                continue
            # quantify with probes scaled to the cheap pool's virtual T0
            vres = cheap[2] // cheap[1]
            best = None
            for frac in PROBE_FRACS:
                x = max(1, int(vres * frac))
                x2 = round_trip(x, cheap[1], cheap[2], cheap[3],
                                rich[1], rich[2], rich[3])
                if best is None or x2 / x > best[1]:
                    best = (x, x2 / x, x2)
            if best and best[1] > 1.0:
                opps.append((b, cheap[0], rich[0], best[1], best[0], best[2]))

    print(f"\nblocks [{lo}, {hi}] ({hi - lo + 1} boundaries), "
          f"opportunity-boundaries: {len(opps)}")
    if not opps:
        print("no cross-boundary arb above fees in this window "
              "(gas would only make it harder)")
        return
    ratios = sorted(o[3] for o in opps)
    print(f"  ratio: median {ratios[len(ratios)//2]:.6f}  "
          f"max {ratios[-1]:.6f}  (>1.001: {sum(1 for r in ratios if r > 1.001)})")
    # persistence: how many consecutive blocks did each pool-pair stay open
    by_pair = defaultdict(list)
    for b, ca, ri, r, x, x2 in opps:
        by_pair[(ca, ri)].append(b)
    print("\nper pair (top by opportunity-blocks):")
    for (ca, ri), blks in sorted(by_pair.items(), key=lambda kv: -len(kv[1]))[:10]:
        runs, cur = [], 1
        for i in range(1, len(blks)):
            if blks[i] == blks[i - 1] + 1:
                cur += 1
            else:
                runs.append(cur); cur = 1
        runs.append(cur)
        print(f"  {ca[:12]}… <-> {ri[:12]}…  {len(blks)} opp-blocks, "
              f"longest streak {max(runs)}, fees "
              f"{meta[ca][2]}/{meta[ri][2]}")


if __name__ == "__main__":
    main()
