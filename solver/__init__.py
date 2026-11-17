"""
solver -- exact-arithmetic DeFi solvers.

Pure-integer / rational math that mirrors EVM semantics bit-for-bit.
The core is fully offline and dependency-free (stdlib only); the public
API below is the stable surface. Scanning / chain-replay tooling lives
in separate modules and is optional (needs network + RPC).

Public API
----------
Batch-auction settlement (CoW-style):
    Order(oid, sell_tok, buy_tok, sell_amt, buy_min)
    solve_batch(orders, pools=None) -> (fills, buys)
    validate_settlement(orders, fills, buys, pools=None) -> (ok, reason)

Uniswap v2 two-pool arbitrage:
    optimal_arbitrage_x(pool_a, pool_b) -> int      # O(1), tooth-top exact
    evm_out(amount_in, reserve_in, reserve_out) -> int
    evm_profit_two(x, pool_a, pool_b) -> int

Uniswap v3 tick math (Q64.96, EVM-exact):
    get_sqrt_ratio_at_tick(tick) -> int
    swap_within_tick_exact_in(...) -> (sqrt_p_next, amount_in, amount_out)

AMM liquidity layer:
    V2Pool(tok_a, tok_b, r_a, r_b)
    V3Pool(...)
    make_pools(fair_prices, depth_units, offset_bps, rng) -> dict

All amounts are integer base units. No float ever sizes a fill.
"""
from solver.cow import Order, validate_settlement          # noqa: F401
from solver.tournament import solve_exact as solve_batch   # noqa: F401
from solver.v2math import evm_out, evm_profit_two          # noqa: F401
from solver.v2solver import exact_optimal_x as optimal_arbitrage_x  # noqa: F401
from solver.v3math import (get_sqrt_ratio_at_tick,          # noqa: F401
                           swap_within_tick_exact_in)
from solver.amm import V2Pool, V3Pool, make_pools, max_route  # noqa: F401

__all__ = [
    "Order", "solve_batch", "validate_settlement",
    "optimal_arbitrage_x", "evm_out", "evm_profit_two",
    "get_sqrt_ratio_at_tick", "swap_within_tick_exact_in",
    "V2Pool", "V3Pool", "make_pools", "max_route",
]

__version__ = "1.0.0"