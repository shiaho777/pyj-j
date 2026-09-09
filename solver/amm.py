#!/usr/bin/env python3
"""
amm.py -- the external liquidity layer for the settlement simulator.

A Uniswap V2 pool as a settlement counterparty (the "interaction" layer
in CoW terms): orders whose remainder cannot match peer-to-peer route
through pool liquidity instead of dying on the book.

Everything is EVM-exact integer math. The pool leg of a settlement must
be executable on-chain, so the simulator executes it exactly as the EVM
would: reserves evolve sequentially through the batch, each routed order
moves the price for the next. The judge (cow.validate_settlement)
replays every recorded op from the initial reserves as its own
double-entry check -- the solver's claimed buys must equal what the pool
actually pays, or the settlement is invalid.
"""
from fractions import Fraction

from solver.v2math import evm_out

UNIT = 10 ** 18
BPS = 10_000


class V2Pool:
    """V2 pool for the unordered token pair (tok_a, tok_b).

    r_a, r_b are integer base-unit reserves. swap() executes EVM-exact
    and records the op so the judge can replay from init independently.
    """

    def __init__(self, tok_a, tok_b, r_a, r_b):
        self.tok_a, self.tok_b = tok_a, tok_b
        self.init = (r_a, r_b)          # judge replays from here
        self.r_a, self.r_b = r_a, r_b
        self.ops = []                   # (sell_tok, x, y) as executed

    def quote(self, sell_tok, x):
        if sell_tok == self.tok_a:
            return evm_out(x, self.r_a, self.r_b)
        return evm_out(x, self.r_b, self.r_a)

    def swap(self, sell_tok, x):
        y = self.quote(sell_tok, x)
        if sell_tok == self.tok_a:
            self.r_a += x
            self.r_b -= y
        else:
            self.r_b += x
            self.r_a -= y
        self.ops.append((sell_tok, x, y))
        return y

    def snapshot(self):
        """Fresh pool with the same initial reserves (for A/B runs)."""
        return V2Pool(self.tok_a, self.tok_b, *self.init)


def make_pools(fair, depth_units, offset_bps, rng):
    """One pool per unordered pair.

    Pool price = fair * (1 + off) with off uniform in +/- offset_bps:
    pool prices drift independently of the order flow, the way real
    AMM prices lag a moving consensus. Depth is in base units.
    """
    pools = {}
    for (a, b), f in fair.items():
        off = Fraction(rng.randrange(-offset_bps, offset_bps + 1), BPS)
        price = f * (1 + off)          # buy-token per sell-token, a->b
        r_a = depth_units * UNIT
        r_b = (price * r_a).__floor__()
        if r_b <= 0:
            r_b = 1
        pools[frozenset((a, b))] = V2Pool(a, b, r_a, r_b)
    return pools


def max_route(pool, sell_tok, S, B, rem):
    """Largest x <= rem such that the pool pays enough to satisfy the
    order's limit exactly:   evm_out(x, ...) * S >= B * x.

    V2 output-per-input is non-increasing in x up to integer dust, so
    the predicate holds on an initial segment and binary search applies.
    Every x the search accepts was tested directly, so the return value
    is always limit-valid (worst case it undershoots the true max by a
    few wei of dust inside the sawtooth boundary zone).
    """
    if rem <= 0 or min(S, B) <= 0:
        return 0
    lo, hi = 0, rem
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if pool.quote(sell_tok, mid) * S >= B * mid:
            lo = mid
        else:
            hi = mid - 1
    return lo
