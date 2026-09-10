#!/usr/bin/env python3
"""
tournament.py -- closed-loop simulation: calibrated books, exact solver.

Solver: pass 1 bilateral rings (best-counterpart max-volume), pass 2
three-cycles among the remainder. All conservation is exact and all
limits are integer-verified before commit.
"""
import random
import sys
from fractions import Fraction

sys.path.insert(0, ".")
from solver.amm import make_pools, max_route                    # noqa: E402
from solver.cow import Order, validate_settlement               # noqa: E402

UNIT = 10 ** 18
FAIR = {("A", "B"): Fraction(3, 2), ("B", "C"): Fraction(5, 4),
        ("C", "D"): Fraction(2), ("D", "A"): Fraction(1, 3),
        ("A", "C"): Fraction(15, 8), ("B", "D"): Fraction(3)}
PAIRS = list(FAIR.keys())


def gen_book(rng, n_orders):
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
    """Bilateral: o1 sells f1 A receives b1 B; o2 sells b1 B receives f1 A.
    Limits: b1*S1>=B1*f1, f1*S2>=B2*b1. Max b1 then f1."""
    S1, B1, S2, B2 = o1.sell_amt, o1.buy_min, o2.sell_amt, o2.buy_min
    if min(S1, B1, S2, B2) <= 0:
        return None
    b1_hi = min(S2, (S1 * S2) // B2)
    for b1 in (b1_hi, b1_hi - 1, b1_hi - 2, b1_hi - 3):
        if b1 <= 0:
            return None
        f1_max = min(S1, (b1 * S1) // B1)
        f1_min = -((-B2 * b1) // S2)
        if f1_min <= f1_max:
            return f1_max, b1
    return None


def _cycle3_fill(o1, o2, o3):
    """Cycle o1: A->B, o2: B->C, o3: C->A.

    Flows: o1 sells f1(A) and RECEIVES f2(B) from o2; o2 sells f2(B)
    and receives f3(C) from o3; o3 sells f3(C) and receives f1(A) from
    o1. Conservation is exact by this rotation. Limits:
        o1: f2*S1 >= B1*f1 ; o2: f3*S2 >= B2*f2 ; o3: f1*S3 >= B3*f3
    Feasible (in the small) iff l1*l2*l3 <= 1 with li = Bi/Si.
    Try each order as the fully-filled anchor; verify exactly.
    """
    l1 = Fraction(o1.buy_min, o1.sell_amt)
    l2 = Fraction(o2.buy_min, o2.sell_amt)
    l3 = Fraction(o3.buy_min, o3.sell_amt)
    if l1 * l2 * l3 > 1:
        return None
    S1, S2, S3 = o1.sell_amt, o2.sell_amt, o3.sell_amt
    B1, B2, B3 = o1.buy_min, o2.buy_min, o3.buy_min

    def ceil_fr(fr, x):
        return -((-fr.numerator * x) // fr.denominator)

    cands = []
    f1 = S1
    f2 = ceil_fr(l1, f1)                      # o1 needs f2 >= l1*f1
    f3 = ceil_fr(l2, f2)                      # o2 needs f3 >= l2*f2
    cands.append((f1, f2, f3))
    f2 = S2
    f3 = ceil_fr(l2, f2)
    f1 = ceil_fr(l3, f3)                      # o3 needs f1 >= l3*f3
    cands.append((f1, f2, f3))
    f3 = S3
    f1 = ceil_fr(l3, f3)
    f2 = ceil_fr(l1, f1)
    cands.append((f1, f2, f3))

    best = None
    for f1, f2, f3 in cands:
        f1, f2, f3 = min(f1, S1), min(f2, S2), min(f3, S3)
        if min(f1, f2, f3) <= 0:
            continue
        if (f2 * S1 >= B1 * f1 and f3 * S2 >= B2 * f2
                and f1 * S3 >= B3 * f3):
            v = f1 + f2 + f3
            if best is None or v > best[0]:
                best = (v, f1, f2, f3)
    return best


def _routable(orders, fills):
    """Orders with remaining sell volume, tightest limit first.

    The tightest limits lose routability fastest as the pool price
    walks away, so they route first; loose orders can still route after.
    """
    rem = [o for o in orders if o.sell_amt - fills.get(o.oid, 0) > 0]
    rem.sort(key=lambda o: (-Fraction(o.buy_min, o.sell_amt), o.oid))
    return rem


def _route_two_hop(p1, p2, sell_tok, hub, S, B, rem):
    """Max x <= rem through p1 (sell->hub) then p2 (hub->buy), requiring
    leg 2 to ABSORB leg 1's output fully (any hub leftover would break
    conservation). Returns (x, y1, y2) or None."""
    def feasible(x):
        y1, used1 = p1.quote_full(sell_tok, x)
        if y1 <= 0:
            return None
        y2, used2 = p2.quote_full(hub, y1)
        if used2 != y1:            # leg-2 cap -> hub leftover -> infeasible
            return None
        if y2 * S < B * x:         # limit at the requested size
            return None
        return y1, y2

    lo, hi, best = 0, rem, None
    while lo <= hi:
        mid = (lo + hi) // 2
        r = feasible(mid)
        if r:
            best = (mid, r[0], r[1])
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def solve_exact(orders, pools=None):
    """Pass 1: bilateral best-counterpart rings. Pass 2: 3-cycles.
    Pass 3: route remainders through AMM liquidity (EVM-exact)."""
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

    # pass 2: 3-cycles among remaining orders
    by_dir = {}
    for o in orders:
        if o.oid not in used:
            by_dir.setdefault((o.sell_tok, o.buy_tok), []).append(o)
    for (a, b), l1 in list(by_dir.items()):
        for (b2, c), l2 in list(by_dir.items()):
            if b2 != b:
                continue
            l3 = by_dir.get((c, a))
            if not l3:
                continue
            done = False
            for o1 in l1:
                if done or o1.oid in used:
                    continue
                for o2 in l2:
                    if o2.oid in used:
                        continue
                    for o3 in l3:
                        if o3.oid in used:
                            continue
                        r = _cycle3_fill(o1, o2, o3)
                        if r:
                            _, f1, f2, f3 = r
                            fills[o1.oid] = f1
                            buys[o1.oid] = f2
                            fills[o2.oid] = f2
                            buys[o2.oid] = f3
                            fills[o3.oid] = f3
                            buys[o3.oid] = f1
                            used.update((o1.oid, o2.oid, o3.oid))
                            done = True
                            break
                    if done:
                        break

    # pass 3: AMM backstop for the remainder (peers first, pools second
    # -- the CoW architecture: coincidence of wants beats AMM fees)
    if pools:
        for o in _routable(orders, fills):
            left = o.sell_amt - fills.get(o.oid, 0)
            pool = pools.get(frozenset((o.sell_tok, o.buy_tok)))
            if pool is None:
                continue
            x = max_route(pool, o.sell_tok, o.sell_amt, o.buy_min, left)
            if x > 0:
                y, used = pool.swap(o.sell_tok, x)
                fills[o.oid] = fills.get(o.oid, 0) + used
                buys[o.oid] = buys.get(o.oid, 0) + y

        # pass 3b: two-hop routes sell -> hub -> buy through any token
        # with pools to both legs (production routes most long-tail flow
        # this way; single-hop discovery misses all of it)
        hub_tokens = set()
        for key in pools:
            hub_tokens.update(key)
        for o in _routable(orders, fills):
            left = o.sell_amt - fills.get(o.oid, 0)
            if left <= 0:
                continue
            best = None
            for hub in hub_tokens:
                if hub == o.sell_tok or hub == o.buy_tok:
                    continue
                p1 = pools.get(frozenset((o.sell_tok, hub)))
                p2 = pools.get(frozenset((hub, o.buy_tok)))
                if p1 is None or p2 is None:
                    continue
                r = _route_two_hop(p1, p2, o.sell_tok, hub,
                                   o.sell_amt, o.buy_min, left)
                if r and (best is None or r[0] > best[0][0]):
                    best = (r, hub, p1, p2)
            if best:
                (x, _y1, _y2), hub, p1, p2 = best
                ya, used1 = p1.swap(o.sell_tok, x)
                yb, used2 = p2.swap(hub, ya)
                if used2 != ya:        # should not happen (feasible checked)
                    continue
                fills[o.oid] = fills.get(o.oid, 0) + used1
                buys[o.oid] = buys.get(o.oid, 0) + yb
    return fills, buys


def solve_float64(orders, pools=None):
    """Baseline: same partial-fill structure, float64 + int() casts."""
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
                r_o1 = o1.buy_min / o1.sell_amt
                r_o2 = o2.buy_min / o2.sell_amt
                b1 = int(min(o2.sell_amt, o1.sell_amt / r_o2))
                f1 = int(min(o1.sell_amt, o2.sell_amt / r_o1))
                if b1 > 0 and f1 > 0:
                    fills[o1.oid] = f1
                    buys[o1.oid] = b1
                    fills[o2.oid] = b1
                    buys[o2.oid] = f1
                    used.add(o1.oid)
                    used.add(o2.oid)
                    break
    # float64 pass 3: same backstop, float arithmetic. The float brain
    # sizes the crossing and the payout; the chain executes evm_out()
    # exactly. Whatever disagrees, the judge catches.
    if pools:
        for o in _routable(orders, fills):
            left = o.sell_amt - fills.get(o.oid, 0)
            pool = pools.get(frozenset((o.sell_tok, o.buy_tok)))
            if pool is None:
                continue
            if o.sell_tok == pool.tok_a:
                r_in, r_out = pool.r_a, pool.r_b
            else:
                r_in, r_out = pool.r_b, pool.r_a
            # float crossing: 997*x*r_out*S == B*x*(1000*r_in + 997*x)
            xstar = r_out * o.sell_amt / o.buy_min - 1000 * r_in / 997
            x = int(min(left, xstar))
            if x <= 0:
                continue
            y_claim = int(997 * x * r_out / (1000 * r_in + 997 * x))
            y_true, _used = pool.swap(o.sell_tok, x)  # chain pays exactly
            if y_claim <= 0:
                continue
            fills[o.oid] = fills.get(o.oid, 0) + x
            buys[o.oid] = buys.get(o.oid, 0) + y_claim
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


def main(n_batches=300, seed=11, depth_units=1_000_000, offset_bps=60):
    rng = random.Random(seed)
    pool_rng = random.Random(seed + 1000)   # separate stream: books keep
    ex_v = amm_v = f_v = 0                  # the battle-9 sequence exactly
    ex_vol = amm_vol = f_vol = 0
    ex_sup = amm_sup = f_sup = Fraction(0)
    routed = 0
    for _ in range(n_batches):
        n = rng.randrange(4, 17)
        orders = gen_book(rng, n)
        pools = make_pools(FAIR, depth_units, offset_bps, pool_rng)

        f, b = solve_exact(orders)                     # peers only
        ok, _ = validate_settlement(orders, f, b)
        if ok:
            ex_v += 1
            ex_vol += volume(orders, f)
            ex_sup += surplus(orders, f, b)
        else:
            print("!! EXACT (peer) INVALID -- BUG")

        f2, b2 = solve_exact(orders, pools)            # peers + amm
        ok2, _ = validate_settlement(orders, f2, b2, pools)
        if ok2:
            amm_v += 1
            amm_vol += volume(orders, f2)
            amm_sup += surplus(orders, f2, b2)
            for p in pools.values():
                routed += sum(x for _, x, _ in p.ops)
        else:
            print("!! EXACT (+AMM) INVALID -- BUG")

        p2 = {k: p.snapshot() for k, p in pools.items()}
        gf, gb = solve_float64(orders, p2)
        ok3, _ = validate_settlement(orders, gf, gb, p2)
        if ok3:
            f_v += 1
            f_vol += volume(orders, gf)
            f_sup += surplus(orders, gf, gb)

    print("=" * 64)
    print(f"  TOURNAMENT (+AMM routing): {n_batches} books, pools at "
          f"depth {depth_units:,} units, price offset +/-{offset_bps}bps")
    print("=" * 64)
    print(f"exact (peer only) : valid {ex_v}/{n_batches}  "
          f"volume {ex_vol:,}  surplus {float(ex_sup):.3e}")
    print(f"exact (+amm)       : valid {amm_v}/{n_batches}  "
          f"volume {amm_vol:,} ({(amm_vol/ex_vol-1)*100:+.1f}%)  "
          f"surplus {float(amm_sup):.3e}  "
          f"routed {routed//UNIT:,} units")
    print(f"f64   (+amm)       : valid {f_v}/{n_batches}  "
          f"volume {f_vol:,}  surplus {float(f_sup):.3e}")
    print("=" * 64)


if __name__ == "__main__":
    main()
