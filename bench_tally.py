#!/usr/bin/env python3
"""
bench_tally.py -- honest, reproducible benchmark: Tally vs the tools an
agent would actually reach for (Python fractions, sympy, mpmath).

Every workload is EXACT or certified-precision; float64 is not a
contestant because it is not correct on these workloads.

Run:  python3 bench_tally.py [repeats]
Output: markdown table + machine-readable JSON (bench_tally.json).
"""
import json
import statistics
import sys
import time
from fractions import Fraction

import tally_ast as T

REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def timed(fn, repeats=REPEATS):
    """COLD-cache timing: clear Tally's result + convergent caches before
    each run so the benchmark measures real computation, not cache hits.
    (Warm-cache speedups are reported separately in the README.)"""
    ts = []
    for _ in range(repeats):
        T._RESULT_CACHE.clear()
        T._PI_CACHE["g"] = -1
        T._LN2_CACHE["g"] = -1
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


ROWS = []


def row(workload, tally_ms, ref_ms, ref_name, note=""):
    speedup = ref_ms / tally_ms if tally_ms else float("inf")
    ROWS.append({"workload": workload, "tally_ms": round(tally_ms, 3),
                 "reference_ms": round(ref_ms, 3), "reference": ref_name,
                 "speedup": round(speedup, 2), "note": note})


# ---- 1. exact rational sum of 1000 decimal literals -------------------
data = [f"{i}.{i % 100:02d}" for i in range(1000)]

def tally_sum():
    T.evaluate({"op": "sum", "of": data})

def frac_sum():
    sum(Fraction(s) for s in data)

row("sum of 1000 decimal literals", timed(tally_sum), timed(frac_sum),
    "Python fractions", "exact rational result")

# ---- 2. exact variance, catastrophic-cancellation scale ---------------
xs = [str(10**9 + i) for i in range(100)]

def tally_var():
    T.evaluate({"op": "var", "of": xs})

def frac_var():
    fs = [Fraction(s) for s in xs]
    m = sum(fs) / len(fs)
    sum((x - m) ** 2 for x in fs) / len(fs)

row("variance of 100 values ~1e9", timed(tally_var), timed(frac_var),
    "Python fractions", "two-pass exact; numpy one-pass gives 128.0 here")

# ---- 3. exact 6x6 Hilbert determinant ---------------------------------
H = [[f"1/{i + j + 1}" for j in range(6)] for i in range(6)]

def tally_det():
    T.evaluate({"op": "det", "m": H})

import sympy
Hs = sympy.Matrix([[sympy.Rational(1, i + j + 1) for j in range(6)]
                   for i in range(6)])

def sympy_det():
    Hs.det()

row("det of 6x6 Hilbert matrix", timed(tally_det), timed(sympy_det),
    "sympy Rational", "exact rational determinant")

# ---- 4. exact 6x6 Hilbert inverse --------------------------------------
def tally_inv():
    T.evaluate({"op": "inv", "m": H})

def sympy_inv():
    Hs.inv()

row("inv of 6x6 Hilbert matrix", timed(tally_inv), timed(sympy_inv),
    "sympy Rational", "integer inverse, exact")

# ---- 5. pi to 200 digits -----------------------------------------------
def tally_pi():
    T.evaluate({"op": "pi", "digits": 200}, digits=200)

import mpmath

def mpmath_pi():
    mpmath.mp.dps = 210
    mpmath.nstr(mpmath.pi, 205)

row("pi to 200 digits", timed(tally_pi), timed(mpmath_pi),
    "mpmath (dps=210)", "Tally: rational convergent; mpmath: binary mpf")

# ---- 6. exp(1) to 200 digits -------------------------------------------
def tally_exp():
    T.evaluate({"op": "exp", "arg": 1, "digits": 200}, digits=200)

def mpmath_exp():
    mpmath.mp.dps = 210
    mpmath.nstr(mpmath.exp(1), 205)

row("exp(1) to 200 digits", timed(tally_exp), timed(mpmath_exp),
    "mpmath (dps=210)")

# ---- 7. sqrt(2) to 500 digits ------------------------------------------
def tally_sqrt():
    T.evaluate({"op": "sqrt", "arg": 2, "digits": 500}, digits=500)

def mpmath_sqrt():
    mpmath.mp.dps = 510
    mpmath.nstr(mpmath.sqrt(2), 505)

row("sqrt(2) to 500 digits", timed(tally_sqrt), timed(mpmath_sqrt),
    "mpmath (dps=510)")

# ---- 8. mortgage payment, exact rational rate --------------------------
def tally_pmt():
    T.evaluate({"op": "pmt", "rate": "0.005", "periods": 360, "pv": 300000})

def frac_pmt():
    r = Fraction(1, 200)
    n = 360
    g = (1 + r) ** n
    -(300000 * g) / ((g - 1) / r)

row("mortgage pmt (360 periods, exact)", timed(tally_pmt), timed(frac_pmt),
    "Python fractions", "rate 0.005 stays 1/200, never a float")

# ---- report ------------------------------------------------------------
print(f"| workload | Tally (ms) | reference (ms) | ref tool | speedup |")
print(f"|---|---|---|---|---|")
for r in ROWS:
    print(f"| {r['workload']} | {r['tally_ms']} | {r['reference_ms']} "
          f"| {r['reference']} | {r['speedup']}x |")
with open("bench_tally.json", "w") as f:
    json.dump({"repeats": REPEATS, "rows": ROWS}, f, indent=2)
print("\nwrote bench_tally.json")