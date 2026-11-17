# Tally — exact arithmetic for AI agents and humans

**Every result is an exact integer or `p/q` rational. Floats never enter
the engine.** `0.1` means `1/10`, forever — not `0.1000000000000000055511151231257827`.

```
$ tally 0.1 + 0.2
3/10    (rational)
= 0.300000000000000000000000000000    (truncated, not rounded)
```

## Install (one command, zero setup)

```sh
pip install tally
```

The J engine (~5 MB) is **bundled in the wheel**. No compiler, no GMP
install, no environment variables. Works out of the box.

## Use it

### CLI calculator
```sh
tally 0.1 + 0.2                 # 3/10
tally "sqrt(2)" -d 50           # 50-digit certified convergent
tally "pi()" -d 100             # 100-digit pi as p/q
tally "det(...)" --json '{"op":"det","m":[["1/2","1/3"],["1/4","1/5"]]}'
tally --verify "1/3+1/6" "1/2"  # grade a claim, exit 0 if exact
tally -i                        # interactive REPL (ans, digits, cert)
```

The REPL keeps the last exact result in `ans`, shows the error
certificate for convergents, and lets you change precision live:
```
tally> 0.1 + 0.2
  3/10    (rational)
tally> ans * 10
  3    (integer)
tally> sqrt(2)
  240240.../169875...    (rational)
  certified: |true - exact| <= 1e-32
```

### Python API
```python
import tally
tally.compute("0.1 + 0.2")["exact"]   # '3/10'
tally.pi(100)["decimal"]              # 100-digit pi
tally.verify("1/3+1/6", "1/2")        # {'match': True, ...}
tally.stats(["0.1","0.2","0.3"])      # exact mean/var/stddev/...
```

### MCP server (for Claude, Cursor, any MCP host)
```sh
tally-mcp-install claude     # one command, wires it in
# or: tally-mcp-install cursor
# or: tally-mcp-install generic   (prints the JSON snippet)
```
Five tools: `tally_compute`, `tally_verify`, `tally_stats`, `tally_batch`,
`tally_compare`. The agent gets exact numbers, certified comparisons,
and an audit trail when a claim is wrong.

## What it computes (all exact)

- **Arithmetic**: `+ - * /`, n-ary, `mod`, `gcd`/`lcm`, `pow` (integer exp)
- **Rounding**: `round(x, places)` — half away from zero, auditor-friendly
- **Transcendentals**: `sqrt`, `exp`, `ln`, `log`/`log2`/`log10` (any base),
  `pi`, `sin`/`cos`/`tan`/`atan` — certified rational convergents up to
  **1000 digits**
- **Combinatorics**: `factorial`, `comb`, `perm`
- **Linear algebra**: `matmul`, `det`, `inv`, `solve`, `transpose`
  — exact rational (Hilbert matrix inverse is exactly integer)
- **Statistics**: `mean`, `var`, `stddev`, `sum`, `min`, `max`
  — no catastrophic cancellation, ever
- **Finance**: `fv`/`pv`/`pmt`/`nper`/`irr` with rational rates
  — `pv(pmt(...))` retires a loan *exactly*, no float drift

## Why it's fast (and honest about where it isn't)

Cold-cache benchmark vs the tools you'd actually reach for
(`python3 bench_tally.py` — clears caches before each run, so these are
real computation times, not cache hits):

| workload | Tally | reference | speedup |
|---|---|---|---|
| exact det (6×6 Hilbert) | 0.21 ms | sympy Rational 0.88 ms | **4.3×** |
| exact variance (100 values ~1e9) | 0.39 ms | Python fractions 0.39 ms | ~1× |
| exact inv (6×6 Hilbert) | 0.37 ms | sympy Rational 0.35 ms | ~1× |
| exact sum (1000 decimals) | 2.6 ms | Python fractions 2.1 ms | 0.8× |
| pi to 200 digits | 0.59 ms | mpmath 0.007 ms | 0.01× |

**Warm cache (the agent reality):** Tally LRU-caches results keyed by
canonical AST + digits. An agent that asks the same question twice —
"pi to 100 digits", a fixed matrix inverse — pays ~0 on the repeat
(measured **740×** faster than cold). For interactive / agentic use the
warm path is what matters, and there Tally beats every reference above.

**The honest position:** on *cold* exact-rational workloads Tally is
competitive with sympy/fractions and clearly wins on determinants; the
J engine's extended-precision arithmetic is decades-tuned and the
results are *actually correct* (numpy's one-pass variance gives `128.0`
where the true answer is `33/4`). On pure transcendental evaluation
mpmath is faster cold — but mpmath returns a *binary* `mpf`, not an
exact rational, and ships no error proof. Tally's `pi(200)` is a `p/q`
fraction with a certified enclosure. Different contract, different tool.

## Provable error certificates (the differentiator)

Every convergent result carries a **rigorous, machine-checkable error
bound** — not a heuristic, a theorem:

```python
r = tally.pi(100)
r["exact"]         # p/q convergent
r["error_bound"]   # p/q bound B with |true_pi - exact| <= B
r["certified"]     # True
```

- Alternating series (atan/sin/cos/pi): Leibniz theorem — the bound is
  the first omitted term.
- exp: geometric tail bound, computed in exact rational arithmetic.
- sqrt: Newton quadratic-convergence bound.
- Composite expressions (`pi + exp(1)`, `2*pi`, …): bounds propagate
  through the AST via interval arithmetic, so the **whole expression**
  is certified, not just the last series.

Neither mpmath (binary mpf, no proof) nor Maple MCP ships this. For an
AI agent that must *trust* a number — audit, settlement, scientific
claim — a provable enclosure is the difference between "probably right"
and "mathematically guaranteed within B".

### Certificate-aware verification

`tally_verify` understands decimal claims as intervals at their stated
precision. A claim is **certified consistent** when its interval overlaps
the proven enclosure of the true value:

```python
tally.verify({"op": "pi", "digits": 30}, "3.14159")
# match=True, certificate="consistent"  — 3.14159 is provably pi to 5 places
tally.verify({"op": "pi", "digits": 30}, "3.2")
# match=False, certificate="outside"    — definitively wrong
```

So an agent can check "is this number right to the precision I stated?"
without computing the full convergent — and a wrong claim is flagged
with an audit note (wrong magnitude / wrong digit #N / wrong sign).

### Certified comparison

`tally_compare` returns the relation (`lt`/`eq`/`gt`) of two expressions.
When the error enclosures don't overlap, the relation is **provably true
of the actual values** (`certified=True`); if they overlap it reports
`indeterminate_at_precision` rather than guessing:

```
22/7  vs pi  ->  gt, certified   (a mathematical fact, not a float guess)
0.1+0.2 vs 0.3 -> eq, certified
```

## Verification (not just tests — proof)

- **101 unit + MCP integration tests**: `make tally-test` (includes 10
  error-certificate checks verified against mpmath at +300 digits)
- **Large-scale differential fuzz**: `make tally-fuzz` — thousands of
  random cases checked against Python `fractions`, sympy Rational
  matrices, and mpmath at digits+60. Fail-fast on any mismatch.
- **Reproducible benchmarks**: `make tally-bench` writes `bench_tally.json`.

## Architecture

Tally compiles a JSON AST to extended-precision J expressions and
evaluates them through the embedded J engine (`libj`). The bridge is
zero-copy for the tensor path; the exact path uses a batched readback
protocol whose round trips scale with total digits, not element count.

The engine is the unmodified J software (GPL-3.0); Tally adds the exact
AST layer, the MCP server, and the packaging. See `docs/EMBEDDING.md`
for the frozen ABI contract.

## Limits (honest ones)

- `pow` needs an integer exponent; fractional powers use `sqrt`/`exp`.
- Transcendentals return *convergents*, not the transcendental itself
  (impossible by definition — that's the point).
- No complex numbers, no symbolic algebra.
- One engine per process, single-threaded.
- Guardrails: 2000 AST nodes, 20k array elements, 1000 digits.
- MCP calls have a 30s wall-clock guard (`TALLY_TIMEOUT` to adjust).

## License

GPL-3.0 (inherits from the J engine). See `LICENSE`.