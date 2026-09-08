"""
v3math.py -- exact Uniswap v3 core math (Q64.96 fixed point, integer).

Everything mirrors the EVM semantics of SqrtPriceMath.sol / TickMath.sol
but in Python arbitrary-precision integers, so there is no 512-bit
FullMath needed and no float64 drift. Rounding directions are preserved
bit-for-bit (roundUp/roundDown as in the Solidity).

Fixed point: sqrtPX96 = sqrt(price) * 2^96, price = token1/token0.
Liquidity L: L^2 = X * Y (amounts in raw units).

Derivations (real numbers, all verifiable from x = L/sqrtP, y = L*sqrtP):
  next price after token0 input dx:
      sqrtP' = L * sqrtP / (L + dx * sqrtP)
  next price after token1 input dy:
      sqrtP' = sqrtP + dy / L
  amount0 between prices A > B (moving down):
      dx = L * (1/B - 1/A) = L * (A - B) / (A*B)
  amount1 between prices A < B (moving up):
      dy = L * (B - A)

NOTE on TickMath: on-chain uses a squaring table; here we use exact
integer sqrt (isqrt), which matches floor(sqrt(price)*2^96). The on-chain
value can differ by +/-1 wei from isqrt in rare ticks (it rounds to
nearest via its iteration); production use should validate against
foundry mainnet forks. Everything else (price deltas, swap amounts) is
exact once the boundary prices are given.
"""

Q96 = 2 ** 96


def mul_div(a, b, d):
    """FullMath.mulDiv — trivial in Python big ints."""
    return (a * b) // d


def mul_div_round_up(a, b, d):
    if a == 0 or b == 0:
        return 0
    return (a * b + d - 1) // d


def div_rounding_up(a, b):
    return (a + b - 1) // b


def get_next_sqrt_price_from_amount0_rounding_up(sqrt_p_x96, liquidity, amount0):
    """SqrtPriceMath.getNextSqrtPriceFromAmount0RoundingUp(_, _, _, true).

    Price moves DOWN for token0 input. roundUp=true (exact-in): the price
    is rounded so that the derived output is never overestimated.
    """
    if amount0 == 0:
        return sqrt_p_x96
    numerator1 = liquidity << 96
    # sqrtP' = (sqrtP * numerator1) / (numerator1 + amount0 * sqrtP), round up
    product = mul_div_round_up(numerator1, sqrt_p_x96, numerator1 + amount0 * sqrt_p_x96)
    # EVM then rounds the uint160 cast (truncation); the roundUp division
    # above already guarantees >= true value, cast truncation is part of
    # on-chain semantics and is preserved by Python int
    return product


def get_next_sqrt_price_from_amount0_rounding_down(sqrt_p_x96, liquidity, amount0):
    if amount0 == 0:
        return sqrt_p_x96
    numerator1 = liquidity << 96
    return mul_div(numerator1, sqrt_p_x96, numerator1 + amount0 * sqrt_p_x96)


def get_next_sqrt_price_from_amount1_rounding_down(sqrt_p_x96, liquidity, amount1):
    """SqrtPriceMath.getNextSqrtPriceFromAmount1RoundingDown(_, _, _, false).

    Price moves UP for token1 input.
    """
    if amount1 == 0:
        return sqrt_p_x96
    return sqrt_p_x96 + (amount1 << 96) // liquidity


def get_next_sqrt_price_from_amount1_rounding_up(sqrt_p_x96, liquidity, amount1):
    if amount1 == 0:
        return sqrt_p_x96
    return sqrt_p_x96 + div_rounding_up(amount1 << 96, liquidity)


def get_amount0_delta(sqrt_a_x96, sqrt_b_x96, liquidity, round_up):
    """SqrtPriceMath.getAmount0Delta — token0 between two prices.

    amount0 = L * 2^96 * |A-B| / (A*B), computed as
    divRound(mulDiv(L<<96, |A-B|, A), B) exactly as on-chain.
    """
    if sqrt_a_x96 > sqrt_b_x96:
        sqrt_a_x96, sqrt_b_x96 = sqrt_b_x96, sqrt_a_x96
    numerator1 = liquidity << 96
    numerator2 = sqrt_b_x96 - sqrt_a_x96
    if round_up:
        return div_rounding_up(mul_div(numerator1, numerator2, sqrt_b_x96), sqrt_a_x96)
    return mul_div(numerator1, numerator2, sqrt_b_x96) // sqrt_a_x96


def get_amount1_delta(sqrt_a_x96, sqrt_b_x96, liquidity, round_up):
    """SqrtPriceMath.getAmount1Delta — token1 between two prices.

    amount1 = L * |A-B| / 2^96.
    """
    if sqrt_a_x96 > sqrt_b_x96:
        sqrt_a_x96, sqrt_b_x96 = sqrt_b_x96, sqrt_a_x96
    numerator2 = sqrt_b_x96 - sqrt_a_x96
    if round_up:
        return mul_div_round_up(liquidity, numerator2, Q96)
    return mul_div(liquidity, numerator2, Q96)


# ------------------------------------------------------------- TickMath

_TICK_CACHE = {}


def get_sqrt_ratio_at_tick(tick):
    """TickMath.getSqrtRatioAtTick via exact integer arithmetic.

    price = 1.0001^tick ; sqrtPX96 = floor(sqrt(price) * 2^96)

    Exact identity used:
        sqrt(price) * 2^96  squared  =  price * 2^192
    so with price = 10001^t / 10000^t  (t = |tick|):
        sqrtPX96 = isqrt(10001^t * 2^192 / 10000^t)
                 = isqrt(10001^t * 10000^t * 2^192) // 10000^t
    (using floor(sqrt(n/d)) = isqrt(n*d)//d).

    Half-size shortcut for even t: sqrt(1.0001^t) = 1.0001^(t/2) exactly,
    so sqrtPX96 = 10001^(t/2) * 2^96 // 10000^(t/2) with NO isqrt at all.
    For odd t: factor out one sqrt(1.0001):
        sqrt(price) = 1.0001^((t-1)/2) * sqrt(1.0001)
        sqrtPX96 = isqrt(10001^t * 10000^(t-1) * 2^192) // (10000^t)
    both forms keep the big-int operands at half the naive size.

    Results are cached; extreme ticks (~887k) take ~1s once, mid-range
    ticks are sub-millisecond. On-chain rounds via a squaring table and
    can differ by 1 on some ticks -- validate against a fork in production.
    """
    if tick in _TICK_CACHE:
        return _TICK_CACHE[tick]
    if tick < TickMathConstants.MIN_TICK or tick > TickMathConstants.MAX_TICK:
        raise ValueError("tick out of range")
    import math
    t = abs(tick)
    if t % 2 == 0:
        h = t // 2
        val = (10001 ** h) * Q96 // (10000 ** h)     # exact, no isqrt
    else:
        # sqrt(10001^t / 10000^t) = 10001^((t-1)/2) * sqrt(10001*10000)^1
        #                        / 10000^((t+1)/2) ... derive directly:
        # sqrtPX96^2 <= price*2^192 < (sqrtPX96+1)^2, so
        # sqrtPX96 = isqrt(10001^t * 10000^t * 2^192) // 10000^t
        val = math.isqrt((10001 ** t) * (10000 ** t) * Q96 ** 2) // (10000 ** t)
    if tick < 0:
        # 1/price: sqrt(1/price)*2^96 = 2^192 / sqrt(price*2^192)...
        # exact: sqrtPX96(-t) = isqrt(10000^t * 10001^t * 2^192) // 10001^t
        if t % 2 == 0:
            h = t // 2
            val = (10000 ** h) * Q96 // (10001 ** h)
        else:
            val = math.isqrt((10001 ** t) * (10000 ** t) * Q96 ** 2) // (10001 ** t)
    _TICK_CACHE[tick] = val
    return val


class TickMathConstants:
    MIN_TICK = -887272
    MAX_TICK = 887272
    MIN_SQRT_RATIO = 4295128740          # on-chain value at MIN_TICK
    MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342


def swap_within_tick_exact_in(sqrt_p_x96, liquidity, amount_remaining, sqrt_target_x96, zero_for_one):
    """One iteration of the v3 swap loop inside a single liquidity range.

    Mirrors SwapMath.computeSwapStep for exactInput (fee excluded here:
    fee is applied by the caller before this step).
    Returns (sqrt_p_next, amount_in_used, amount_out).
    """
    if zero_for_one:
        # token0 in, price moves down toward sqrt_target <= sqrt_p
        sqrt_p_next = get_next_sqrt_price_from_amount0_rounding_up(
            sqrt_p_x96, liquidity, amount_remaining)
        if sqrt_p_next < sqrt_target_x96:
            # price target reached first: cap at target
            sqrt_p_next = sqrt_target_x96
            amount_in = get_amount0_delta(sqrt_p_x96, sqrt_target_x96, liquidity, True)
            amount_out = get_amount1_delta(sqrt_p_x96, sqrt_target_x96, liquidity, False)
        else:
            amount_in = amount_remaining
            amount_out = get_amount1_delta(sqrt_p_x96, sqrt_p_next, liquidity, False)
        return sqrt_p_next, amount_in, amount_out
    else:
        # token1 in, price moves up toward sqrt_target >= sqrt_p
        sqrt_p_next = get_next_sqrt_price_from_amount1_rounding_down(
            sqrt_p_x96, liquidity, amount_remaining)
        if sqrt_p_next > sqrt_target_x96:
            sqrt_p_next = sqrt_target_x96
            amount_in = get_amount1_delta(sqrt_p_x96, sqrt_target_x96, liquidity, True)
            amount_out = get_amount0_delta(sqrt_p_x96, sqrt_target_x96, liquidity, False)
        else:
            amount_in = amount_remaining
            amount_out = get_amount0_delta(sqrt_p_x96, sqrt_p_next, liquidity, False)
        return sqrt_p_next, amount_in, amount_out
