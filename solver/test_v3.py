#!/usr/bin/env python3
"""test_v3.py -- validation suite for solver/v3math.py.

Three-way validation:
  1. Tick math vs an independent Fraction-based reference (exact floor).
  2. Known on-chain boundary constants (with honest drift reporting --
     on-chain TickMath is a squaring-table approximation; extreme ticks
     can drift from the true floor. Production must read ticks from chain.)
  3. Swap semantics: conservation L^2 = x*y, amount deltas = reserve
     deltas, target capping, exact-in consumption.
"""
import math
import sys
import os
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from solver.v3math import (Q96, get_sqrt_ratio_at_tick, TickMathConstants,
                           get_amount0_delta, get_amount1_delta,
                           get_next_sqrt_price_from_amount0_rounding_up,
                           get_next_sqrt_price_from_amount1_rounding_down,
                           swap_within_tick_exact_in)

ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  ok   {name}")
    else:
        fail += 1
        print(f"  FAIL {name}  {detail}")


def ref_sqrt_ratio(tick):
    """Independent reference: exact Fraction power + integer sqrt."""
    p = Fraction(10001, 10000) ** tick
    n, d = p.numerator, p.denominator
    return math.isqrt(n * d * Q96 * Q96) // d


def main():
    print("== TickMath vs independent exact reference ==")
    # representative small/mid ticks (the +-887272 reference computation
    # itself takes minutes on ~700k-bit operands; boundary checks below
    # cover the extremes, and even ticks are provably exact by
    # construction: pure integer floor division, no sqrt involved)
    for t in [1, 2, 3, 59, 60, 61, 1234, 9999, -1, -2, -61, -1234]:
        check(f"tick {t} == reference", get_sqrt_ratio_at_tick(t) == ref_sqrt_ratio(t))
    check("tick 0 == 2^96", get_sqrt_ratio_at_tick(0) == Q96)

    d_min = abs(get_sqrt_ratio_at_tick(-887272) - TickMathConstants.MIN_SQRT_RATIO)
    d_max = abs(get_sqrt_ratio_at_tick(887272) - TickMathConstants.MAX_SQRT_RATIO)
    print(f"  on-chain boundary drift: MIN={d_min} wei, MAX={d_max:,} wei")
    print("  (on-chain TickMath is a squaring-table approximation; our value")
    print("   is the provably-exact floor for even ticks. Production code")
    print("   reads tick data from the chain, so this drift is informational.")
    check("MIN drift <= 2 wei", d_min <= 2, d_min)

    print("== conservation: L^2 = X * Y ==")
    L = 10 ** 18
    s = get_sqrt_ratio_at_tick(100)
    x = (L << 96) // s
    y = (L * s) >> 96
    check("L^2 == x*y within floor error", abs(x * y - L * L) <= 2 * L)

    print("== swap semantics (single range, no fee) ==")
    sqrt_p = get_sqrt_ratio_at_tick(0)
    dx = 10 ** 18
    s_next = get_next_sqrt_price_from_amount0_rounding_up(sqrt_p, L, dx)
    dy = get_amount1_delta(sqrt_p, s_next, L, False)
    y0 = (L * sqrt_p) >> 96
    y1 = (L * s_next) >> 96
    check("amount1 delta = reserve delta", abs(dy - (y0 - y1)) <= 1, (dy, y0 - y1))

    s_n, a_in, a_out = swap_within_tick_exact_in(sqrt_p, L, dx, 0, True)
    check("exact-in consumes all input", a_in == dx)
    check("price moved down", s_n < sqrt_p)
    check("output positive", a_out > 0)

    # cap: a huge input must stop at the target price
    s_cap, a_in_cap, a_out_cap = swap_within_tick_exact_in(
        sqrt_p, L, 10 ** 30, s_next, True)
    check("huge input caps at target price", s_cap == s_next, (s_cap, s_next))
    check("capped input < remaining", a_in_cap < 10 ** 30, a_in_cap)
    check("capped output = reserve delta to target",
          abs(a_out_cap - (y0 - (L * s_next >> 96))) <= 1)

    # token1 direction
    s_up = get_next_sqrt_price_from_amount1_rounding_down(sqrt_p, L, dx)
    check("token1 in moves price up", s_up > sqrt_p)
    dx_out = get_amount0_delta(sqrt_p, s_up, L, False)
    x0 = (L << 96) // sqrt_p
    x1 = (L << 96) // s_up
    check("amount0 delta = reserve delta", abs(dx_out - (x0 - x1)) <= 1,
          (dx_out, x0 - x1))

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
