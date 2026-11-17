# solver — exact-arithmetic DeFi solvers

Pure-integer / rational solvers that mirror EVM semantics bit-for-bit.
The core is **fully offline, stdlib-only** — no network, no third-party
packages, no float ever sizes a fill.

This is the "real workload" proof for the pyj exact-kernel thesis: in
batch auctions a float64 solver's rounding can *invalidate* a settlement
outright (one violated wei reverts the whole batch), so exactness is not
a nicety — it is the difference between winning and losing the auction.

## Install / import

The package is importable from the repo root with no build step:

```python
import solver
```

Or add the repo root to `PYTHONPATH`. No dependencies beyond the Python
standard library (`fractions`, `math`, `random`).

## Public API

### Batch-auction settlement (CoW-style)

```python
from solver import Order, solve_batch, validate_settlement

orders = [
    Order(0, "A", "B", sell_amt=10**18, buy_min=10**18),
    Order(1, "B", "A", sell_amt=10**18, buy_min=10**18),
]
fills, buys = solve_batch(orders)              # peers only
ok, reason = validate_settlement(orders, fills, buys)
assert ok
```

- `Order(oid, sell_tok, buy_tok, sell_amt, buy_min)` — all amounts are
  integer base units; partial fills allowed.
- `solve_batch(orders, pools=None) -> (fills, buys)` — three passes:
  bilateral best-counterpart rings, 3-cycles, then AMM backstop
  (single-hop + two-hop) when `pools` is supplied.
- `validate_settlement(orders, fills, buys, pools=None) -> (ok, reason)`
  — the integer-only "on-chain judge": limit + conservation checks, and
  with `pools`, a full EVM-exact replay of every recorded pool op.

### Uniswap v2 two-pool arbitrage

```python
from solver import optimal_arbitrage_x, evm_profit_two

pool_a = (reserve0, reserve1)     # integer base units
pool_b = (reserve0, reserve1)
x = optimal_arbitrage_x(pool_a, pool_b)   # O(1), tooth-top exact
profit = evm_profit_two(x, pool_a, pool_b)
```

The profit curve over integer input is a *sawtooth*, not a smooth
parabola; the optimum is a tooth top, found in O(1) with `isqrt` + an
analytic inverse — no search, no float. See `REPORT.md` battle 1.

### Uniswap v3 tick math (Q64.96)

```python
from solver import get_sqrt_ratio_at_tick, swap_within_tick_exact_in
```

EVM-exact `SqrtPriceMath` / `TickMath`. Known honest boundary: extreme
ticks carry a wei-level deviation from the on-chain square table
(documented in `v3math.py`); single-tick swaps only (multi-tick walking
lives in `tickwalk.py`, which needs RPC for live tick context).

### AMM liquidity layer

```python
from solver import V2Pool, V3Pool, make_pools, max_route
```

`V2Pool` / `V3Pool` execute EVM-exact and record ops so the judge can
replay them independently from initial reserves.

## Verification (all offline, deterministic)

```sh
make verify        # test_v3 (24 checks) + tournament + stress
```

- `test_v3.py` — 24 conservation / limit / truncation checks, hard exit code.
- `tournament.py` — 300 calibrated books; exact solver must be 100% valid
  (gate exits non-zero otherwise). Reports the float64 baseline's validity
  collapse for contrast.
- `stress.py` — 19-scenario stress matrix (batch size, limit tightness,
  size disparity, adversarial books, AMM offset/depth); gate exits non-zero
  if any scenario produces an invalid exact settlement.

## Honest limitations

- **Offline core only.** Scanning / chain-replay tooling (`scan.py`,
  `mainnet.py`, `cowpaper.py`, `profit.py`, `venues.py`, `l4.py`,
  `shadow.py`) needs network + RPC endpoints and is *not* part of the
  stable public API. Some of it reads a `/tmp` cache that is not
  reproducible across machines.
- **v3 single-tick** in the settlement solver; multi-tick walking is a
  separate module.
- **Dynamic-fee pools** are not modeled.
- The judge is a faithful integer model of the settlement contract, not
  the contract itself; on-chain integration is out of scope here.

## Layout

| module | role |
|---|---|
| `v2math.py` / `v2solver.py` | v2 EVM semantics + O(1) arbitrage |
| `v3math.py` | Q64.96 tick / swap math |
| `cow.py` | `Order`, integer judge, two-token settle |
| `amm.py` | `V2Pool` / `V3Pool` liquidity layer |
| `tournament.py` | `solve_exact` (the production solver) + benchmark |
| `stress.py` / `test_v3.py` / `bench_arb.py` | verification batteries |
| `scan.py` / `mainnet.py` / … | optional network tooling (not stable API) |

See `REPORT.md` for the full battle log and economic analysis.