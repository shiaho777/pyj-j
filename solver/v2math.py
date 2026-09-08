"""
v2math.py -- exact Uniswap v2 swap math.

Two layers:
  evm_*   -- EVM semantics: integer floor division, bit-for-bit what the
             chain executes. Any profit number from here is real.
  cont_*  -- continuous rational idealization (no floor), used to locate
             the continuous optimum before snapping to the integer grid.

Everything is Python ints / Fractions. No float64 anywhere.
"""
from fractions import Fraction

FEE_NUM = 997    # Uniswap v2 fee: 0.3% = 3/1000, so amountInWithFee = in*997
FEE_DEN = 1000


def evm_out(amount_in, reserve_in, reserve_out):
    """Exact EVM getAmountOut for Uniswap v2 (integer, floor division).

    Mirrors the Solidity formula exactly:
        amountInWithFee = amountIn * 997
        numerator       = amountInWithFee * reserveOut
        denominator     = reserveIn * 1000 + amountInWithFee
        amountOut       = numerator / denominator        (floor)
    """
    if amount_in <= 0 or reserve_in <= 0 or reserve_out <= 0:
        return 0
    aif = amount_in * FEE_NUM
    return (aif * reserve_out) // (reserve_in * FEE_DEN + aif)


def cont_out(amount_in, reserve_in, reserve_out):
    """Idealized continuous output as an exact Fraction (no floor)."""
    if amount_in <= 0:
        return Fraction(0)
    aif = amount_in * FEE_NUM
    return Fraction(aif * reserve_out, reserve_in * FEE_DEN + aif)


def evm_profit_two(x, pool_a, pool_b):
    """On-chain profit (integer base units) of a two-pool round trip.

    Buy pool_a's token1 with x of token0, sell the proceeds into pool_b
    for token0 back. pool_* are (reserve0, reserve1) integer tuples.
    """
    if x <= 0:
        return 0
    y = evm_out(x, pool_a[0], pool_a[1])
    if y == 0:
        return -x
    x_back = evm_out(y, pool_b[1], pool_b[0])
    return x_back - x


def cont_profit_two(x, pool_a, pool_b):
    """Continuous rational profit P(x) of the same round trip."""
    if x <= 0:
        return Fraction(0)
    y = cont_out(x, pool_a[0], pool_a[1])
    x_back = cont_out(y, pool_b[1], pool_b[0])
    return x_back - x
