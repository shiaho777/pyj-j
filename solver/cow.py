"""
cow.py -- batch-auction settlement: float64 solver vs exact solver.

The scenario mirrors CoW Protocol batch auctions, reduced to its integer
core (two tokens, coincidence-of-wants, uniform clearing price):

  Orders. Each order: "sell up to S units of token A, receive at least
  B units of token B" -- partial fills allowed. All amounts are integer
  base units (18-dec scale), exactly as on-chain.

  Settlement. The solver picks an integer fill f_i <= S_i for every order
  and an integer buy amount b_i for every filled order. The settlement
  contract then validates, in PURE INTEGER ARITHMETIC:

    (L) limit:        b_i * S_i >= B_i * f_i          for every order
    (C) conservation: for each token: amount sold == amount bought

  A single violated wei reverts the whole settlement: the solver loses
  the batch to a competitor, pays gas, and takes a reputation hit.
  This is why exactness is non-negotiable here -- unlike v2 arbitrage,
  where float64 only loses profit at the margin, here float64 rounding
  can invalidate the solution outright.

  Solvers (same greedy structure, only the arithmetic differs):
    float64_settle: price/limits/fills in float64, int() casts at the end
    exact_settle:   Fractions throughout, integer snaps in the
                    constraint-preserving direction

  validate_settlement is the shared judge: Python ints only, never
  floats -- it stands in for the on-chain contract.
"""
import random
import sys
from fractions import Fraction

UNIT = 10 ** 18          # 18-dec base units


class Order:
    __slots__ = ("oid", "sell_tok", "buy_tok", "sell_amt", "buy_min")

    def __init__(self, oid, sell_tok, buy_tok, sell_amt, buy_min):
        self.oid = oid
        self.sell_tok = sell_tok
        self.buy_tok = buy_tok
        self.sell_amt = sell_amt          # integer base units
        self.buy_min = buy_min            # integer base units


def validate_settlement(orders, fills, buys, pools=None):
    """The on-chain judge. Integer-only. Returns (ok, reason).

    With pools (solver.amm.V2Pool dict): the AMM interaction layer is
    double-entry checked too. Every recorded pool op is replayed from
    the pool's INITIAL reserves with EVM-exact math -- the claimed
    output must equal what the chain would actually pay -- and token
    conservation becomes  sold + ext_out == bought + ext_in  per token.
    """
    for o in orders:
        f = fills.get(o.oid, 0)
        b = buys.get(o.oid, 0)
        if f < 0 or f > o.sell_amt:
            return False, f"order {o.oid}: fill {f} out of range"
        if f > 0 and b <= 0:
            return False, f"order {o.oid}: filled but no buy"
        if b * o.sell_amt < o.buy_min * f:
            return False, (f"order {o.oid}: limit violated "
                           f"(b*S={b*o.sell_amt} < B*f={o.buy_min*f})")
    sold, bought = {}, {}
    for o in orders:
        f = fills.get(o.oid, 0)
        b = buys.get(o.oid, 0)
        sold[o.sell_tok] = sold.get(o.sell_tok, 0) + f
        bought[o.buy_tok] = bought.get(o.buy_tok, 0) + b
    ext_in, ext_out = {}, {}
    if pools:
        from solver.v2math import evm_out     # lazy: cow stays standalone
        from solver.v3math import (get_amount0_delta, get_amount1_delta,
                                   get_next_sqrt_price_from_amount0_rounding_up,
                                   get_next_sqrt_price_from_amount1_rounding_down)
        for pool in pools.values():
            if hasattr(pool, "r_a"):          # V2: replay reserves
                r_a, r_b = pool.init
                for sell_tok, x, y in pool.ops:
                    if sell_tok == pool.tok_a:
                        buy_tok = pool.tok_b
                        y_true = evm_out(x, r_a, r_b)
                        r_a += x
                        r_b -= y_true
                    else:
                        buy_tok = pool.tok_a
                        y_true = evm_out(x, r_b, r_a)
                        r_b += x
                        r_a -= y_true
                    if y_true != y:
                        return False, (f"pool {pool.tok_a}/{pool.tok_b} op "
                                       f"x={x}: claimed {y} != replay {y_true}")
                    ext_in[sell_tok] = ext_in.get(sell_tok, 0) + x
                    ext_out[buy_tok] = ext_out.get(buy_tok, 0) + y
            else:                             # V3: replay price walk
                sp, L = pool.init
                for sell_tok, x, y in pool.ops:
                    if sell_tok == pool.tok_a:
                        buy_tok = pool.tok_b
                        s_full = get_next_sqrt_price_from_amount0_rounding_up(
                            sp, L, x)
                        s_next = s_full if s_full >= pool.lo else pool.lo
                        y_true = get_amount1_delta(s_next, sp, L, False)
                    else:
                        buy_tok = pool.tok_a
                        s_full = get_next_sqrt_price_from_amount1_rounding_down(
                            sp, L, x)
                        s_next = s_full if s_full <= pool.hi else pool.hi
                        y_true = get_amount0_delta(sp, s_next, L, False)
                    if y_true != y:
                        return False, (f"v3 pool {pool.tok_a}/{pool.tok_b} op "
                                       f"x={x}: claimed {y} != replay {y_true}")
                    sp = s_next
                    ext_in[sell_tok] = ext_in.get(sell_tok, 0) + x
                    ext_out[buy_tok] = ext_out.get(buy_tok, 0) + y
    for tok in set(sold) | set(bought) | set(ext_in) | set(ext_out):
        lhs = sold.get(tok, 0) + ext_out.get(tok, 0)
        rhs = bought.get(tok, 0) + ext_in.get(tok, 0)
        if lhs != rhs:
            return False, (f"token {tok}: sold+ext_out {lhs} != "
                           f"bought+ext_in {rhs}")
    return True, "ok"


def user_surplus(orders, fills, buys):
    """Total user surplus (Fraction): how much better than limit each
    filled order executes, (exec_rate - limit_rate) * fill."""
    total = Fraction(0)
    for o in orders:
        f = fills.get(o.oid, 0)
        b = buys.get(o.oid, 0)
        if f > 0:
            total += Fraction(b * o.sell_amt - o.buy_min * f, o.sell_amt)
    return total


def matched_volume_a(orders, fills):
    """A-units sold by A-sellers (settlement size; fees are paid on it)."""
    return sum(fills.get(o.oid, 0) for o in orders if o.sell_tok == "A")


# ------------------------------------------------------------------
# float64 solver -- competent generic implementation, float arithmetic
# ------------------------------------------------------------------

def float64_settle(orders):
    a_sellers = [o for o in orders if o.sell_tok == "A"]
    b_sellers = [o for o in orders if o.sell_tok == "B"]
    if not a_sellers or not b_sellers:
        return {}, {}

    # feasibility: limits must overlap
    p_lo = max(o.buy_min / o.sell_amt for o in a_sellers)
    p_hi = min(o.sell_amt / o.buy_min for o in b_sellers)
    if p_lo > p_hi:
        return {}, {}

    # B-sellers fill fully (they define demand):
    #   sell S_j B, receive B_j A (their exact limit buy)
    fills, buys = {}, {}
    for o in b_sellers:
        fills[o.oid] = o.sell_amt
        buys[o.oid] = o.buy_min
    a_needed = sum(buys.values())
    b_sold = sum(fills.values())
    if a_needed == 0 or b_sold == 0:
        return {}, {}

    # A-sellers fill greedily to exactly cover the demand
    remaining = a_needed
    for o in a_sellers:
        if remaining <= 0:
            break
        # float64 sizing: int cast on a float division
        f = int(min(o.sell_amt, remaining))
        if f > 0:
            fills[o.oid] = f
            # float64 buy amount: int(limit_ratio * fill)
            buys[o.oid] = int(o.buy_min / o.sell_amt * f)
            remaining -= f
    # distribute the B surplus (float subtraction, int cast)
    total_b = sum(buys[o.oid] for o in a_sellers if o.oid in buys)
    excess = b_sold - total_b
    if excess > 0:
        first_a = a_sellers[0].oid
        if first_a in buys:
            buys[first_a] += int(excess)
    return fills, buys


# ------------------------------------------------------------------
# exact solver -- same structure, Fractions + constraint-preserving snaps
# ------------------------------------------------------------------

def exact_settle(orders):
    a_sellers = [o for o in orders if o.sell_tok == "A"]
    b_sellers = [o for o in orders if o.sell_tok == "B"]
    if not a_sellers or not b_sellers:
        return {}, {}

    p_lo = max(Fraction(o.buy_min, o.sell_amt) for o in a_sellers)
    p_hi = min(Fraction(o.sell_amt, o.buy_min) for o in b_sellers)
    if p_lo > p_hi:
        return {}, {}

    # B-sellers fill fully at their limit buy (they define demand)
    fills, buys = {}, {}
    for o in b_sellers:
        fills[o.oid] = o.sell_amt
        buys[o.oid] = o.buy_min
    a_needed = sum(buys.values())
    b_sold = sum(fills.values())
    if a_needed == 0 or b_sold == 0:
        return {}, {}
    if a_needed > sum(o.sell_amt for o in a_sellers):
        return {}, {}          # supply insufficient (shouldn't occur; batch
                               # generator trims B-side to keep this rare)

    # A-sellers fill greedily; the LAST filled order takes the remainder,
    # so conservation on A holds by construction
    remaining = a_needed
    for o in a_sellers:
        if remaining <= 0:
            break
        f = min(o.sell_amt, remaining)
        if f > 0:
            fills[o.oid] = f
            buys[o.oid] = -((-o.buy_min * f) // o.sell_amt)   # ceil(B*f/S)
            remaining -= f

    # feasibility of B conservation: the minimum B the A-sellers must
    # receive is sum(ceil(limit*f)) <= p_hi * a_needed <= b_sold (the
    # B-side's implied price is at least p_hi), so this holds whenever the
    # limits overlap; verify and distribute the exact surplus
    total_b = sum(buys[o.oid] for o in a_sellers if o.oid in buys)
    if total_b > b_sold:
        return {}, {}
    excess = b_sold - total_b
    if excess > 0:
        first_a = a_sellers[0].oid
        buys[first_a] += excess     # giving MORE B than a limit is always safe
    return fills, buys


# ------------------------------------------------------------------
# benchmark
# ------------------------------------------------------------------

def make_batch(rng, n_a, n_b, spread_bps):
    """Random tight-limit orders crossing a fair price.

    A-sellers accept at least fair*(1-eps) B per A; B-sellers accept at
    least fair^-1*(1-eps) A per B. Both sides can trade strictly inside
    their limits near the fair price, so surpluses are positive and
    limits bind when the solver squeezes fills to the boundary."""
    fair = Fraction(3, 2)
    orders = []
    oid = 0
    for _ in range(n_a):
        S = rng.randrange(10, 1000) * UNIT
        lim = fair * (1 - Fraction(rng.randrange(0, spread_bps), 10_000))
        B = (lim * S).__floor__() + rng.randrange(0, 3)
        orders.append(Order(oid, "A", "B", S, B)); oid += 1
    supply = sum(o.sell_amt for o in orders)
    for _ in range(n_b):
        S = rng.randrange(10, 1000) * UNIT
        lim = (1 / fair) * (1 - Fraction(rng.randrange(0, spread_bps), 10_000))
        B = (lim * S).__floor__() + rng.randrange(0, 3)
        if B > supply:                        # keep demand <= supply
            break
        orders.append(Order(oid, "B", "A", S, B)); oid += 1
        supply -= B
    return orders


def run_bench(n_batches=200, seed=7):
    rng = random.Random(seed)
    stats = {"f64_valid": 0, "f64_reverted": 0, "exact_valid": 0,
             "exact_reverted": 0, "f64_vol": 0, "exact_vol": 0,
             "f64_surplus": Fraction(0), "exact_surplus": Fraction(0)}
    fails = []
    for i in range(n_batches):
        orders = make_batch(rng, rng.randrange(1, 5), rng.randrange(1, 5),
                             rng.choice([2, 5, 10, 20]))

        f_f, f_b = float64_settle(orders)
        ok, reason = validate_settlement(orders, f_f, f_b)
        if ok:
            stats["f64_valid"] += 1
            stats["f64_vol"] += matched_volume_a(orders, f_f)
            stats["f64_surplus"] += user_surplus(orders, f_f, f_b)
        else:
            stats["f64_reverted"] += 1
            fails.append((i, reason))

        e_f, e_b = exact_settle(orders)
        ok_e, _ = validate_settlement(orders, e_f, e_b)
        if ok_e:
            stats["exact_valid"] += 1
            stats["exact_vol"] += matched_volume_a(orders, e_f)
            stats["exact_surplus"] += user_surplus(orders, e_f, e_b)
        else:
            stats["exact_reverted"] += 1

    n = n_batches
    print("=" * 66)
    print(f"  BATCH AUCTIONS: float64 vs exact solver (n={n})")
    print("=" * 66)
    print(f"float64 settlements VALID on-chain : {stats['f64_valid']}/{n}"
          f"  ({stats['f64_valid']*100//n}%)")
    print(f"float64 settlements REVERTED       : {stats['f64_reverted']}/{n}"
          f"  ({stats['f64_reverted']*100//n}%)")
    print(f"exact settlements VALID on-chain   : {stats['exact_valid']}/{n}"
          f"  ({stats['exact_valid']*100//n}%)")
    print(f"exact settlements REVERTED         : {stats['exact_reverted']}/{n}")
    print()
    print(f"matched volume (valid batches only):")
    print(f"  float64: {stats['f64_vol']/UNIT:>14,.2f} A-units")
    print(f"  exact  : {stats['exact_vol']/UNIT:>14,.2f} A-units")
    print(f"user surplus (valid batches only):")
    print(f"  float64: {float(stats['f64_surplus'])/UNIT:>14,.6f} A-units")
    print(f"  exact  : {float(stats['exact_surplus'])/UNIT:>14,.6f} A-units")
    print()
    print("economic meaning: each REVERT = the batch is lost to a")
    print(f"competitor (fee + reputation). float64 lost "
          f"{stats['f64_reverted']} of {n} auctions outright.")
    if fails:
        print("\nfirst 3 float64 failures:")
        for i, reason in fails[:3]:
            print(f"  batch {i}: {reason}")
    print("=" * 66)
    return stats


if __name__ == "__main__":
    run_bench()
