#!/usr/bin/env python3
"""
bench_arb.py -- benchmark: standard float64 arbitrage bot vs exact solver.

Scenario: 50 realistic two-pool WETH/USDC arbitrage opportunities.
Liquidity from $50k to $100M, imbalances from 0.61% to 5%.

Ground truth: TOOTH-AWARE. At scale, EVM integer profit P(x) is a sawtooth:
within a tooth the slope is exactly -1 (each extra wei in is a wei lost),
and profit jumps whenever evm_out crosses an output-unit boundary. The
discrete optimum is always a tooth peak (the last x before a jump down).
Ground truth enumerates tooth peaks around the continuous optimum via the
exact integer inverse x_for_k and picks the max -- verified equal to
brute-force scans on small pools, and structurally correct at whale scale
where naive brute force cannot see across 3.3e8-wei tooth spacing.

Honesty notes (do not overstate):
  - All tooth peaks near the optimum are within ~1 wei of each other. So
    the real comparison is "lands on a tooth peak" vs "lands mid-tooth".
  - A float64 bot that lands mid-tooth loses (tooth remainder) * (1 - marginal
    sell rate) in profit -- real wei, but we report the USD scale honestly.
"""
import math
import random
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PYJ = os.path.dirname(HERE)
sys.path.insert(0, PYJ)

from solver.v2math import evm_profit_two, evm_out, FEE_NUM, FEE_DEN
from solver.v2solver import float64_optimal_x, exact_optimal_x, has_arbitrage

WETH = 10**18
USDC = 10**6


def make_pool_pair(eth_price_usd, imbalance_pct, liquidity_usd):
    """Pool A (USDC cheap), Pool B (USDC dear). Arb: buy A, sell B."""
    pA = eth_price_usd * (1 + imbalance_pct / 200.0)
    pB = eth_price_usd * (1 - imbalance_pct / 200.0)
    eth_a = int((liquidity_usd / 2 / pA) * WETH)
    usdc_a = int((liquidity_usd / 2) * USDC)
    eth_b = int((liquidity_usd / 2 / pB) * WETH)
    usdc_b = int((liquidity_usd / 2) * USDC)
    return (eth_a, usdc_a), (eth_b, usdc_b)


def x_for_k(k, a0, a1):
    """Smallest x with evm_out(x, a0, a1) >= k (exact integer inverse)."""
    if k <= 0:
        return 1
    num = k * FEE_DEN * a0
    den = FEE_NUM * (a1 - k)
    x = -(-num // den)
    while evm_out(x, a0, a1) < k:
        x += 1
    return x


def ground_truth(pool_a, pool_b, hint_x):
    """Tooth-aware optimum: enumerate tooth peaks around hint_x."""
    a0, a1 = pool_a
    k0 = evm_out(hint_x, a0, a1)
    best_x, best_p = hint_x, evm_profit_two(hint_x, pool_a, pool_b)
    for k in range(max(0, k0 - 3), k0 + 6):
        xk = x_for_k(k, a0, a1)
        if xk <= 0:
            continue
        p = evm_profit_two(xk, pool_a, pool_b)
        if p > best_p:
            best_p, best_x = p, xk
    return best_x, best_p


def is_tooth_peak(x, pool_a, pool_b):
    """True tooth peak: x is the FIRST input achieving its output level,
    i.e. x == x_for_k(out(x)). Within a tooth profit falls by 1 per wei,
    so the maximum sits exactly at each tooth's left edge."""
    a0, a1 = pool_a
    if x <= 0:
        return False
    return x == x_for_k(evm_out(x, a0, a1), a0, a1)


def main():
    random.seed(42)
    configs = [
        (50_000, 1.0, 5.0),
        (1_000_000, 0.7, 3.0),
        (20_000_000, 0.65, 2.0),
        (100_000_000, 0.61, 1.5),
    ]
    cases = []
    for liq, lo, hi in configs:
        for _ in range(13):
            imb = random.uniform(lo, hi)
            price = random.uniform(2500, 3500)
            pA, pB = make_pool_pair(price, imb, liq)
            if has_arbitrage(pA, pB):
                cases.append((liq, imb, price, pA, pB))
    cases = cases[:50]

    stats = {
        "f64_subopt": 0,          # float64 profit < truth
        "f64_midtooth": 0,        # float64 x is not a tooth peak
        "exact_miss": 0,          # exact profit < truth
        "exact_worse_than_f64": 0,
        "f64_lost_wei": 0,
        "f64_lost_usd": 0.0,
        "profit_usd": 0.0,
    }
    worst = []

    for liq, imb, price, pA, pB in cases:
        x_ex = exact_optimal_x(pA, pB)
        p_ex = evm_profit_two(x_ex, pA, pB)
        x_f64 = float64_optimal_x(pA, pB)
        p_f64 = evm_profit_two(x_f64, pA, pB)

        # ground truth from the exact solver's hint, then cross-check with
        # a hint at the float64 x too (teeth far apart -> widen if needed)
        x_tr, p_tr = ground_truth(pA, pB, x_ex)
        x_tr2, p_tr2 = ground_truth(pA, pB, x_f64)
        if p_tr2 > p_tr:
            x_tr, p_tr = x_tr2, p_tr2

        lost_wei = p_tr - p_f64
        lost_usd = (lost_wei / WETH) * price
        stats["profit_usd"] += (p_tr / WETH) * price

        if lost_wei > 0:
            stats["f64_subopt"] += 1
            stats["f64_lost_wei"] += lost_wei
            stats["f64_lost_usd"] += lost_usd
        if not is_tooth_peak(x_f64, pA, pB):
            stats["f64_midtooth"] += 1
        if p_ex < p_tr:
            stats["exact_miss"] += 1
        if p_f64 > p_ex:
            stats["exact_worse_than_f64"] += 1

        worst.append((lost_usd, liq, imb, price, x_ex, p_ex, x_f64, p_f64,
                      x_tr, p_tr, lost_wei))

    n = len(cases)
    print("=" * 66)
    print("  TWO-POOL V2 ARBITRAGE: float64 bot vs exact solver (n=50)")
    print("=" * 66)
    print(f"float64 landed mid-tooth (suboptimal): {stats['f64_midtooth']}/{n}")
    print(f"float64 profit < truth:                {stats['f64_subopt']}/{n}")
    print(f"exact solver missed truth:             {stats['exact_miss']}/{n}")
    print(f"float64 beat exact solver:             {stats['exact_worse_than_f64']}/{n}")
    print()
    print(f"total arb profit at truth (50 arbs):   ${stats['profit_usd']:,.2f}")
    print(f"profit float64 left on the table:      {stats['f64_lost_wei']:,} wei")
    print(f"                                     = ${stats['f64_lost_usd']:,.4f}")
    print(f"  -> float64 captured "
          f"{(1 - stats['f64_lost_usd']/stats['profit_usd'])*100 if stats['profit_usd'] else 100:.4f}% of max profit")
    print("=" * 66)

    worst.sort(reverse=True)
    print("\nTop 3 float64 losses:")
    for lost_usd, liq, imb, price, x_ex, p_ex, x_f64, p_f64, x_tr, p_tr, lost_wei in worst[:3]:
        print(f"  pool ${liq:,} @ {imb:.2f}% imbalance:")
        print(f"    truth profit : {p_tr:,} wei  (${(p_tr/WETH)*price:,.4f})")
        print(f"    exact profit : {p_ex:,} wei  (${(p_ex/WETH)*price:,.4f})")
        print(f"    float64      : {p_f64:,} wei  (${(p_f64/WETH)*price:,.4f})")
        print(f"    loss         : {lost_wei:,} wei  (${lost_usd:,.6f})")
    return stats


if __name__ == "__main__":
    main()
