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

    # ---- the deployable decision rule -------------------------------
    # pre-bid we KNOW our own extractable but not prod's bid. If the
    # ratio prod/extractable is tight across auctions, expected margin
    # ~= extractable * (1 - median_ratio) -- a rule runnable live.
    ratios = sorted(r["prod_weth"] / r["our_weth"]
                    for r in rows if r["our_weth"] > 1e-12)
    if ratios:
        med = ratios[len(ratios) // 2]
        q1 = ratios[len(ratios) // 4]
        q3 = ratios[3 * len(ratios) // 4]
        print(f"\n  prod/extractable ratio: median {med:.4f} "
              f"[IQR {q1:.4f}..{q3:.4f}]  n={len(ratios)}")
        print("  -> live rule: est_margin = extractable * "
              f"{1-med:.4f}; enter iff est_margin > k*gas")

    # margin vs notional (the whale hypothesis, sliced)
    withn = [r for r in rows if r.get("notional_weth")]
    if len(withn) >= 4:
        withn.sort(key=lambda r: r["notional_weth"])
        half = len(withn) // 2
        lo, hi = withn[:half], withn[half:]
        for name, grp in (("small-notional half", lo),
                          ("large-notional half", hi)):
            ms = [r["margin_weth"] for r in grp]
            pos = sum(1 for m in ms if m > 0)
            print(f"  {name}: n={len(grp)}  positive {pos}/{len(grp)}  "
                  f"mean margin {sum(ms)/len(ms):+.6f} WETH  "
                  f"notional range {grp[0]['notional_weth']:.3f}"
                  f"..{grp[-1]['notional_weth']:.3f} WETH")

    # whale filter sweep: enter iff margin > k*gas
    print("\n  whale-filter sweep (enter iff true margin > k*gas):")
    for k in (1, 3, 5, 10, 20):
        n_enter = sum(1 for r in rows
                      if r["margin_weth"] > k * r["gas_weth"])
        net = sum(r["margin_weth"] - r["gas_weth"] for r in rows
                  if r["margin_weth"] > k * r["gas_weth"])
        print(f"    k={k:>2}: enter {n_enter:>3}/{len(rows)}  "
              f"net {net:+.6f} WETH/auction-set")
    print("=" * 70)
    print("  caveats: sample is tiny (printed above); prod-bid is known")
    print("  only post-hoc -- the live game is blind and competitive;")
    print("  margins assume the same-book subset. framework, not forecast.")
    print("=" * 70)


if __name__ == "__main__":
    main()
