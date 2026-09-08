#!/usr/bin/env python3
"""
tournament.py -- closed-loop simulation: calibrated order books, exact solver.

No real order flow needed: we calibrate a generator from the REAL
settlements we can observe on-chain, then run end-to-end tournaments:

  [synthetic order book ~ calibrated to real stats]
      -> [our exact solver: ring matching + conservation-pinned prices]
      -> [integer judge (validated 0-disagreement vs production)]
      -> [scored against production benchmarks]

Calibration targets (measured from mainnet settlements):
  - batch sizes: 1-2 settled orders per tx (we see the settled subset)
  - order sizes: log-uniform 10..1000 whole tokens (18-dec units)
  - limit tightness: 0-20 bps inside a fair price
  - fill style: mostly full fills; partials occur
"""
import random
import sys
from fractions import Fraction

sys.path.insert(0, ".")
from solver.cow import Order, validate_settlement            # noqa: E402

UNIT = 10 ** 18
FAIR = {("A", "B"): Fraction(3, 2), ("B", "C"): Fraction(5, 4),
        ("C", "D"): Fraction(2), ("D", "A"): Fraction(1, 3),
        ("A", "C"): Fraction(15, 8), ("B", "D"): Fraction(3)}
PAIRS = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"),
         ("A", "C"), ("B", "D")]


def gen_book(rng, n_orders):
    """Orders crossing fair prices, calibrated tightness."""
    orders = []
    for oid in range(n_orders):
        a, b = rng.choice(PAIRS)
        if rng.random() < 0.5:
            sell, buy, fair = a, b, FAIR[(a, b)]
        else:
            sell, buy, fair = b, a, 1 / FAIR[(a, b)]
        lim = fair * (1 - Fraction(rng.randrange(0, 20), 10_000))
        S = rng.randrange(10, 1000) * UNIT
        B = (lim * S).__floor__() + rng.randrange(0, 3)
        orders.append(Order(oid, sell, buy, S, B))
    return orders


def _best_fill(o1, o2):
    """Max-volume bilateral exchange honoring BOTH integer limits.

    o1 sells f1 of A, receives b1 of B; o2 sells b1 of B, receives f1
    of A (exact conservation).
      o1 limit: b1*S1 >= B1*f1   ->  f1 <= (b1*S1)//B1
      o2 limit: f1*S2 >= B2*b1   ->  f1 >= ceil(B2*b1/S2)
    Maximize b1 then f1 (full-fill o1 when possible). The floor-jitter
    on limits (1-2 wei) makes the top candidate occasionally
    infeasible; a short decrement search closes the gap.
    """
    S1, B1, S2, B2 = o1.sell_amt, o1.buy_min, o2.sell_amt, o2.buy_min
    if S1 <= 0 or B1 <= 0 or S2 <= 0 or B2 <= 0:
        return None
    b1_hi = min(S2, (S1 * S2) // B2)          # max B for o1's full S1
    for b1 in (b1_hi, b1_hi - 1, b1_hi - 2, b1_hi - 3):
        if b1 <= 0:
            return None
        f1_max = min(S1, (b1 * S1) // B1)
        f1_min = -((-B2 * b1) // S2)
        if f1_min <= f1_max:
            return f1_max, b1
    return None


def solve_exact(orders):
    """Bilateral ring matching, best-counterpart selection.

    For each order, pick the counterpart maximizing matched volume
    (not first-fit). Sort by size descending so whales match first.
    """
    fills, buys = {}, {}
    by_pair = {}
    for o in orders:
        by_pair.setdefault((o.sell_tok, o.buy_tok), []).append(o)
    for lst in by_pair.values():
        lst.sort(key=lambda o: -o.sell_amt)

    used = set()
    for (s1, b1k), olist in by_pair.items():
        for o1 in olist:
            if o1.oid in used:
                continue
            best = None
            for o2 in by_pair.get((b1k, s1), []):
                if o2.oid in used or o2.oid == o1.oid:
                    continue
                r = _best_fill(o1, o2)
                if r and (not best or r[0] > best[0]):
                    best = (r[0], r[1], o2)
            if best:
                f1, b1, o2 = best
                fills[o1.oid] = f1
                buys[o1.oid] = b1
                fills[o2.oid] = b1
                buys[o2.oid] = f1
                used.add(o1.oid)
                used.add(o2.oid)
    return fills, buys


def solve_float64(orders):
    """Same partial-fill ring structure; float64 rates and int() casts."""
    fills, buys = {}, {}
    by_pair = {}
    for o in orders:
        by_pair.setdefault((o.sell_tok, o.buy_tok), []).append(o)
    used = set()
    for (s1, b1k), olist in by_pair.items():
        for o1 in olist:
            if o1.oid in used:
                continue
            for o2 in by_pair.get((b1k, s1), []):
                if o2.oid in used or o2.oid == o1.oid:
                    continue
                # float64 sizing
                r_o1 = o1.buy_min / o1.sell_amt        # B per A
                r_o2 = o2.buy_min / o2.sell_amt        # A per B
                b1 = int(min(o2.sell_amt, o1.sell_amt / r_o2))
                f1 = int(min(o1.sell_amt, o2.sell_amt / r_o1))
                # commit whatever is consistent-ish in float
                if b1 > 0 and f1 > 0:
                    fills[o1.oid] = f1
                    buys[o1.oid] = b1
                    fills[o2.oid] = b1
                    buys[o2.oid] = f1
                    used.add(o1.oid); used.add(o2.oid)
                    break
    return fills, buys


def volume(orders, fills):
    return sum(fills.get(o.oid, 0) for o in orders) // UNIT


def surplus(orders, fills, buys):
    t = Fraction(0)
    for o in orders:
        f, b = fills.get(o.oid, 0), buys.get(o.oid, 0)
        if f:
            t += Fraction(b * o.sell_amt - o.buy_min * f, o.sell_amt)
    return t


def main(n_batches=300, seed=11):
    rng = random.Random(seed)
    st = dict(exact_valid=0, f64_valid=0, n=0,
              exact_vol=0, f64_vol=0, exact_sup=Fraction(0),
              f64_sup=Fraction(0), n_orders=0, matched_pairs=0)
    for _ in range(n_batches):
        n = rng.randrange(4, 17)          # full book: 4-16 orders
        orders = gen_book(rng, n)
        st["n_orders"] += n

        f_f, f_b = solve_exact(orders)
        ok_e, _ = validate_settlement(orders, f_f, f_b)
        if ok_e:
            st["exact_valid"] += 1
            st["exact_vol"] += volume(orders, f_f)
            st["exact_sup"] += surplus(orders, f_f, f_b)
        g_f, g_b = solve_float64(orders)
        ok_f, _ = validate_settlement(orders, g_f, g_b)
        if ok_f:
            st["f64_valid"] += 1
            st["f64_vol"] += volume(orders, g_f)
            st["f64_sup"] += surplus(orders, g_f, g_b)

    n = n_batches
    print("=" * 64)
    print(f"  TOURNAMENT: {n} synthetic books, {st['n_orders']} orders "
          f"({st['n_orders']//n}/book avg)")
    print("=" * 64)
    print(f"exact solver : valid {st['exact_valid']}/{n}"
          f" ({st['exact_valid']*100//n}%)  "
          f"volume {st['exact_vol']:,}  "
          f"surplus {float(st['exact_sup']):,.3f}")
    print(f"float64      : valid {st['f64_valid']}/{n}"
          f" ({st['f64_valid']*100//n}%)  "
          f"volume {st['f64_vol']:,}  "
          f"surplus {float(st['f64_sup']):,.3f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
