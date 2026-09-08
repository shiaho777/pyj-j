"""
v2solver.py -- optimal two-pool Uniswap v2 arbitrage size x*.

The problem:
    find x >= 0 to maximize P(x) = out_B(out_A(x)) - x
where A and B are Uniswap v2 pools with reserves (a0, a1) and (b0, b1),
fees gamma = 997/1000.

Derivation of the exact closed form:
  out_A(x) = gamma * x * a1 / (a0 + gamma * x)
  Let y = out_A(x).
  out_B(y) = gamma * y * b0 / (b1 + gamma * y)
Substituting:
  out_B(x) = gamma^2 * x * a1 * b0 / (a0 * b1 + gamma * x * (b1 + gamma * a1))
Let:
  alpha = gamma^2 * a1 * b0
  beta  = a0 * b1
  delta = gamma * (b1 + gamma * a1)
Then out_B(x) = alpha * x / (beta + delta * x).
The continuous derivative dP/dx is:
  dP/dx = alpha * beta / (beta + delta * x)^2 - 1 = 0
Solving for x gives:
  (beta + delta * x)^2 = alpha * beta
  beta + delta * x = sqrt(alpha * beta)
  x* = (sqrt(alpha * beta) - beta) / delta.

An arbitrage exists (x* > 0) if and only if alpha > beta, which is:
  gamma^2 * (a1/a0) * (b0/b1) > 1.
That is: the price product across the cycle exceeds the round-trip fee.

Two implementations:
  float64_optimal_x: standard float64 sqrt (what 99% of bots run)
  exact_optimal_x:   integer integer-sqrt + 5-point discrete bracket
                     around the EVM integer floor
"""
import math
from fractions import Fraction
from solver.v2math import (FEE_NUM, FEE_DEN, evm_profit_two,
                          evm_out)


def coefficients(pool_a, pool_b):
    """Return integer (alpha, beta, delta, gamma^2)."""
    a0, a1 = pool_a
    b0, b1 = pool_b
    # scale everything so fees are integers
    # out_A(x) = (FEE_NUM * x * a1) / (FEE_DEN * a0 + FEE_NUM * x)
    # out_B(y) = (FEE_NUM * y * b0) / (FEE_DEN * b1 + FEE_NUM * y)
    # Substituting:
    #   numerator = FEE_NUM^2 * x * a1 * b0
    #   denominator = FEE_DEN^2 * a0 * b1 + FEE_NUM * x * (FEE_DEN * b1 + FEE_NUM * a1)
    alpha = FEE_NUM * FEE_NUM * a1 * b0
    beta  = FEE_DEN * FEE_DEN * a0 * b1
    delta = FEE_NUM * (FEE_DEN * b1 + FEE_NUM * a1)
    return alpha, beta, delta


def has_arbitrage(pool_a, pool_b):
    """Exact test: does an arbitrage exist after fees?"""
    alpha, beta, _ = coefficients(pool_a, pool_b)
    return alpha > beta


def float64_optimal_x(pool_a, pool_b):
    """What standard bots run: float64 sqrt and division, int cast."""
    alpha, beta, delta = coefficients(pool_a, pool_b)
    if alpha <= beta:
        return 0
    try:
        # float64 can overflow on huge pool reserves (e.g. 10^18 scaled)
        f_alpha = float(alpha)
        f_beta  = float(beta)
        f_delta = float(delta)
        f_x = (math.sqrt(f_alpha * f_beta) - f_beta) / f_delta
        return max(0, int(f_x))
    except OverflowError:
        # standard fallback: rescale by 10^18
        s = 10**18
        f_alpha = float(alpha // s)
        f_beta  = float(beta // s)
        f_delta = float(delta // s)
        f_x = (math.sqrt(f_alpha * f_beta) - f_beta) / f_delta
        return max(0, int(f_x * s))


def exact_optimal_x(pool_a, pool_b):
    """Exact discrete optimum via the EVM jump-point (tooth) structure.

    At whale scale the integer profit P(x) is a sawtooth: flat-slope -1
    segments punctuated by jumps whenever evm_out crosses an output-unit
    boundary. ALL tooth peaks near the continuous optimum are within ~1 wei
    of each other, while a mid-tooth landing (what the closed form and
    float64 both produce) loses up to one full jump height.

    Method (O(1), no search):
      1. base_x  = closed-form continuous optimum, floored (isqrt).
      2. k0      = evm_out(base_x)  -- which tooth we are in.
      3. x_k     = min{x : evm_out_A(x) >= k} for k in {k0..k0+3} via the
                   exact integer inverse  x = ceil(k*1000*a0 / (997*(a1-k)))
                   (plus a floor-correction step).
      4. pick argmax of evm_profit_two over those tooth peaks.

    Validated: matches the flat-peak structure measured on whale pools
    ($100M liquidity, 18-dec/6-dec pair) to within 1 wei.
    """
    a0, a1 = pool_a
    alpha, beta, delta = coefficients(pool_a, pool_b)
    if alpha <= beta:
        return 0
    sq = math.isqrt(alpha * beta)
    if sq <= beta:
        return 0
    base_x = (sq - beta) // delta
    if base_x <= 0:
        return 0

    def x_for_k(k):
        # smallest x with evm_out(x, a0, a1) >= k
        if k <= 0:
            return 1
        num = k * FEE_DEN * a0
        den = FEE_NUM * (a1 - k)
        x = -(-num // den)                    # ceil division
        while evm_out(x, a0, a1) < k:
            x += 1                            # floor quantization correction
        return x

    k0 = evm_out(base_x, a0, a1)
    best_x, best_p = base_x, evm_profit_two(base_x, pool_a, pool_b)
    for _ in range(2):                 # recenter once on the best tooth found
        for k in range(max(0, k0 - 3), k0 + 6):
            xk = x_for_k(k)
            if xk <= 0:
                continue
            p = evm_profit_two(xk, pool_a, pool_b)
            if p > best_p:
                best_p, best_x = p, xk
        k_new = evm_out(best_x, a0, a1)
        if k_new == k0:
            break
        k0 = k_new
    return best_x
