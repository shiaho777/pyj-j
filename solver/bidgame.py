#!/usr/bin/env python3
"""
bidgame.py -- the blind-bid strategy curve.

Reality: we submit a settlement WITHOUT seeing production's bid; the
protocol ranks by user surplus and picks the best. Post-hoc we know
both numbers (profit.py logs them per auction). The question:

    If we bid to keep a target margin m (bid = extractable - m),
    what is the win rate, the expected net per auction, and which m
    maximizes it?

    win(m)      : true_margin > m     (true_margin = extractable - prod_bid)
    net_i(m)    : m - gas_i  if we win AND m > gas_i, else 0 (no settle)
    E[net](m)   : mean over logged auctions of net_i(m)

The empirical distribution is solver/scanlog/profitlog.jsonl, which
every profit.py run (and CI) appends to. n is printed loudly: with
small n this is a FRAMEWORK, not a forecast -- its value compounds as
the log grows.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "scanlog", "profitlog.jsonl")


def load_rows():
    rows, windows = [], 0
    if not os.path.exists(LOG):
        return rows, windows
    for line in open(LOG):
        rec = json.loads(line)
        windows += 1
        for r in rec["rows"]:
            r["head"] = rec["head"]
            rows.append(r)
    return rows, windows


def curve(rows, m_grid):
    """E[net per auction](m) and win rate over the logged sample."""
    out = []
    for m in m_grid:
        wins = 0
        net = 0.0
        for r in rows:
            tm = r["margin_weth"]                    # true margin, WETH
            gas = r["gas_weth"]
            if tm > m and m > gas:
                wins += 1
                net += m - gas
        n = len(rows)
        out.append((m, wins / n if n else 0, net / n if n else 0))
    return out


def main():
    rows, windows = load_rows()
    print("=" * 70)
    print("  THE BID GAME -- keep margin m, win iff extractable-prod > m")
    print("=" * 70)
    print(f"  sample: {len(rows)} auctions over {windows} logged windows")
    if not rows:
        print("  (no profitlog yet -- run solver/profit.py first)")
        return
    tms = sorted(r["margin_weth"] for r in rows)
    pos = [t for t in tms if t > 0]
    print(f"  true-margin distribution (WETH): "
          f"min {tms[0]:+.8f}  median {tms[len(tms)//2]:+.8f}  "
          f"max {tms[-1]:+.8f}")
    print(f"  positive margins: {len(pos)}/{len(rows)}")
    if not pos:
        print("  no positive margins logged yet -- nothing to optimize")
        return

    hi = max(pos)
    grid = [hi * i / 40 for i in range(0, 41)]
    c = curve(rows, grid)
    best = max(c, key=lambda p: p[2])
    print(f"\n  optimal target margin m* = {best[0]:.8f} WETH")
    print(f"    win rate at m* : {best[1]*100:.1f}%")
    print(f"    E[net]/auction : {best[2]:.8f} WETH")
    per_window = best[2] * (len(rows) / windows if windows else 0)
    print(f"    E[net]/window  : {per_window:.8f} WETH "
          f"({len(rows)/windows:.1f} auctions/window)")
    print(f"    E[net]/day     : {per_window*72:.6f} WETH "
          f"(72 windows/day)")

    print("\n  strategy curve (m, win%, E[net]/auction in WETH):")
    step = max(1, len(c) // 12)
    for m, wr, en in c[::step]:
        bar = "#" * int(wr * 40)
        print(f"    {m:.8f}  {wr*100:5.1f}%  {en:+.8f}  {bar}")
    print("=" * 70)
    print("  caveats: sample is tiny (printed above); prod-bid is known")
    print("  only post-hoc -- the live game is blind and competitive;")
    print("  margins assume the same-book subset. framework, not forecast.")
    print("=" * 70)


if __name__ == "__main__":
    main()
