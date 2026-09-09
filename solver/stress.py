#!/usr/bin/env python3
"""
stress.py -- stress matrix for the tournament solver.

Dimensions:
  batch size   : sparse (2-6), normal (4-16), dense (50-150)
  tightness    : loose (0-50bps), normal (0-20bps), razor (0-2bps),
                 boundary (exactly 0bps -- knife edge)
  size mix     : uniform (10-1000), whale (1-100000), dust (1-10)
  adversarial  : one-sided book (no counterpart), duplicate orders

Metrics: exact validity (must be 100% or we have a bug), float64
validity, matched volume, capture rate = matched / offered sell volume
(the solver's true quality metric -- greedy pairing loses matches).
"""
import random
import sys
from fractions import Fraction

sys.path.insert(0, ".")
from solver.amm import make_pools                            # noqa: E402
from solver.cow import Order, validate_settlement            # noqa: E402
from solver.tournament import solve_exact, solve_float64     # noqa: E402

UNIT = 10 ** 18
FAIR = {("A", "B"): Fraction(3, 2), ("B", "C"): Fraction(5, 4),
        ("C", "D"): Fraction(2), ("D", "A"): Fraction(1, 3),
        ("A", "C"): Fraction(15, 8), ("B", "D"): Fraction(3)}
PAIRS = list(FAIR.keys())


def gen_book(rng, n, tight_bps, size_lo, size_hi, one_side=False):
    orders = []
    for oid in range(n):
        a, b = rng.choice(PAIRS)
        if rng.random() < 0.5 and not one_side:
            sell, buy, fair = a, b, FAIR[(a, b)]
        else:
            sell, buy, fair = b, a, 1 / FAIR[(a, b)]
        if tight_bps == 0:
            lim = fair
        else:
            lim = fair * (1 - Fraction(rng.randrange(0, tight_bps), 10_000))
        S = rng.randrange(size_lo, size_hi) * UNIT
        B = (lim * S).__floor__() + rng.randrange(0, 3)
        orders.append(Order(oid, sell, buy, S, B))
    return orders


def offered_sell(orders):
    """Total sell volume that has a counterpart direction present."""
    by_dir = {}
    for o in orders:
        by_dir.setdefault((o.sell_tok, o.buy_tok), []).append(o)
    total = 0
    for (s, b), olist in by_dir.items():
        if (b, s) in by_dir:
            total += sum(o.sell_amt for o in olist)
    return total


def run_scenario(name, rng, n_books, pool=None, **kw):
    """pool=None: peer-only (battle-9 metric). pool=(offset_bps, depth):
    AMM backstop on; capture is measured against ALL offered sell
    volume, since a pool takes any side."""
    ex_v = f_v = 0
    ex_vol = f_vol = 0
    offered = 0
    routed = 0
    ex_orders_filled = 0
    n_orders = 0
    for _ in range(n_books):
        orders = gen_book(rng, kw["n"], kw["tight"], kw["lo"], kw["hi"],
                           kw.get("one_side", False))
        n_orders += len(orders)
        off = (offered_sell(orders) if pool is None
               else sum(o.sell_amt for o in orders))
        offered += off

        pools = (make_pools(FAIR, pool[1], pool[0], rng) if pool else None)
        f, b = solve_exact(orders, pools)
        ok, _ = validate_settlement(orders, f, b, pools)
        if ok:
            ex_v += 1
            ex_vol += sum(f.values())
            ex_orders_filled += len(f)
            if pools:
                routed += sum(x for p in pools.values() for _, x, _ in p.ops)
        else:
            print(f"  !! EXACT SOLVER INVALID in scenario {name} -- BUG")

        p2 = ({k: p.snapshot() for k, p in pools.items()} if pools else None)
        gf, gb = solve_float64(orders, p2)
        ok2, _ = validate_settlement(orders, gf, gb, p2)
        if ok2:
            f_v += 1
            f_vol += sum(gf.values())

    cap = ex_vol / offered if offered else 0
    rout = (routed / ex_vol * 100) if ex_vol else 0
    print(f"{name:<26} exact: {ex_v}/{n_books} valid, capture "
          f"{cap*100:5.1f}%, filled {ex_orders_filled}/{n_orders} orders "
          f"| f64: {f_v}/{n_books} valid, vol-ratio "
          f"{(f_vol/ex_vol if ex_vol else 0)*100:5.1f}%"
          + (f", routed {rout:4.1f}%" if pool else ""))
    return dict(name=name, cap=cap, f64_valid=f_v, n=n_books)


def main():
    rng = random.Random(42)
    print("=" * 96)
    print("  STRESS MATRIX -- exact solver must stay 100% valid; capture rate is quality")
    print("=" * 96)

    base = dict(n=10, tight=20, lo=10, hi=1000)

    print("\n[batch size]")
    run_scenario("sparse (2-6)", rng, 200, **{**base, "n": 4})
    run_scenario("normal (10)", rng, 200, **{**base, "n": 10})
    run_scenario("dense (100)", rng, 60, **{**base, "n": 100})

    print("\n[limit tightness]")
    run_scenario("loose 0-50bps", rng, 200, **{**base, "tight": 50})
    run_scenario("normal 0-20bps", rng, 200, **{**base, "tight": 20})
    run_scenario("razor 0-2bps", rng, 200, **{**base, "tight": 2})
    run_scenario("boundary 0bps", rng, 200, **{**base, "tight": 0})

    print("\n[size disparity]")
    run_scenario("dust 1-10", rng, 200, **{**base, "lo": 1, "hi": 10})
    run_scenario("whale 1-100000", rng, 200, **{**base, "lo": 1, "hi": 100000})

    print("\n[adversarial]")
    run_scenario("one-sided book", rng, 100, **{**base, "one_side": True})
    rng2 = random.Random(7)
    run_scenario("exact-boundary dup", rng2, 100, **{**base, "tight": 0})

    print("\n[amm routing: pool price offset vs book fair]")
    DEPTH = 1_000_000
    run_scenario("pool aligned 0bps", rng, 200, pool=(0, DEPTH), **base)
    run_scenario("pool tight +/-15bps", rng, 200, pool=(15, DEPTH), **base)
    run_scenario("pool normal +/-60bps", rng, 200, pool=(60, DEPTH), **base)
    run_scenario("pool wide +/-200bps", rng, 200, pool=(200, DEPTH), **base)

    print("\n[amm routing: depth]")
    run_scenario("shallow 50k units", rng, 200, pool=(60, 50_000), **base)
    run_scenario("deep 10M units", rng, 200, pool=(60, 10_000_000), **base)

    print("\n[amm routing: adversarial]")
    run_scenario("one-sided + pools", rng, 100, pool=(60, DEPTH),
                 **{**base, "one_side": True})
    run_scenario("razor + stale pools", rng, 200, pool=(200, DEPTH),
                 **{**base, "tight": 2})

    print("\n" + "=" * 96)


if __name__ == "__main__":
    main()
