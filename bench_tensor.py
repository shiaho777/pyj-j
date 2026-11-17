#!/usr/bin/env python3
"""
bench_tensor.py -- HONEST tensor-path benchmark.

Two distinct claims, never conflated:

  A. pyj/J vs numpy on raw array ops. numpy wins on its own turf
     (BLAS, SIMD, decades of tuning). We report the real ratio, not a
     cherry-picked internal comparison. The kernel's value is NOT
     out-running numpy on float64 matmul.

  B. The tape compiler vs the INTERPRETED tape (same engine, same
     numerics). This is the ~200x claim, and it is an internal
     optimization ratio -- explicitly labeled as such.

Run:  python3 bench_tensor.py
Writes bench_tensor.json.
"""
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyj

# macOS Accelerate BLAS emits spurious divide-by-zero/overflow warnings
# on some matmul calls even when inputs/outputs are finite and correct
# (verified against einsum: max diff 0.0). Silence them for clean output.
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*encountered in matmul.*")

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = []


def ms(fn, iters=20):
    fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3


# ---------------- A. raw ops vs numpy (honest: numpy wins) -------------
print("=== A. raw array ops: pyj/J vs numpy (float64) ===")
v = np.arange(1_000_000, dtype=np.float64)
pyj.set("bv", v)

t_np_sum = ms(lambda: v.sum(), 100)
def j_sum():
    pyj.do("bs=: +/ bv")
    return pyj.get("bs")
t_j_sum = ms(j_sum, 50)
ROWS.append({"op": "sum 1M float64", "numpy_ms": round(t_np_sum, 3),
             "pyj_ms": round(t_j_sum, 3),
             "numpy_faster_x": round(t_j_sum / t_np_sum, 2)})
print(f"  sum 1M:  numpy {t_np_sum:.3f} ms | pyj {t_j_sum:.3f} ms "
      f"| numpy {t_j_sum/t_np_sum:.1f}x faster")

n = 512
A = np.random.rand(n, n)
B = np.random.rand(n, n)
pyj.set("bA", A)
pyj.set("bB", B)
pyj.do("bmp=: +/ . *")

t_np_mm = ms(lambda: A @ B, 20)
def j_mm():
    pyj.do("bC=: bA bmp bB")
    return pyj.get("bC")
t_j_mm = ms(j_mm, 10)
ROWS.append({"op": f"matmul {n}x{n}", "numpy_ms": round(t_np_mm, 3),
             "pyj_ms": round(t_j_mm, 3),
             "numpy_faster_x": round(t_j_mm / t_np_mm, 2)})
print(f"  matmul {n}: numpy {t_np_mm:.3f} ms | pyj {t_j_mm:.3f} ms "
      f"| numpy {t_j_mm/t_np_mm:.1f}x faster")

# ---------------- B. compiler vs interpreted tape (internal) -----------
print("\n=== B. tape compiler vs interpreted tape (INTERNAL ratio) ===")
pyj.do(f"0!:0 <'{HERE}/ad.ijs'")
pyj.do(f"0!:0 <'{HERE}/adcomp.ijs'")

rng = np.random.default_rng(7)
N = 96
Xdata = np.hstack([rng.normal(size=(N, 2)), np.ones((N, 1))])
Ydata = np.eye(3)[np.repeat(np.arange(3), N // 3)]
W1 = rng.normal(size=(3, 16)) * 0.25
W2 = rng.normal(size=(16, 3)) * 0.25
pyj.set("X_v", Xdata)
pyj.set("Y_v", Ydata)
pyj.set("W1_v", W1)
pyj.set("W2_v", W2)
pyj.do("ADSETP (<'X_v';X_v) , (<'Y_v';Y_v) , (<'W1_v';W1_v) , (<'W2_v';W2_v)")
for s in ("cH=: 0 nmp 2", "cA=: ntanh cH", "cZ=: cA nmp 3", "cM=: nrmax1 cZ",
          "cS=: cZ nrsub1 cM", "cE=: nexp cS", "cSum=: nrsum1 cE",
          "cLge=: nlog cSum", "cLse=: cM nadd cLge", "cSh=: cZ nrsub1 cLse",
          "cG=: cSh nmul 1", "cLG=: nsum cG", "cLG2=: nsum cLG"):
    pyj.do(s)
LOSS = int(pyj.get("cLG2"))
pyj.do(f"{LOSS} ADGEN 1")

def rebuild():
    for s in ("cH=: 0 nmp 2", "cA=: ntanh cH", "cZ=: cA nmp 3", "cM=: nrmax1 cZ",
              "cS=: cZ nrsub1 cM", "cE=: nexp cS", "cSum=: nrsum1 cE",
              "cLge=: nlog cSum", "cLse=: cM nadd cLge", "cSh=: cZ nrsub1 cLse",
              "cG=: cSh nmul 1", "cLG=: nsum cG", "cLG2=: nsum cLG",
              f"gg=: ADGET {LOSS}"):
        pyj.do(s)

def compiled():
    pyj.do("adcB1 ''")
    pyj.get("adcg2")
    pyj.get("adcg3")

t_interp = ms(rebuild, 100)
t_comp = ms(compiled, 200)
ratio = t_interp / t_comp
ROWS.append({"op": "softmax fwd+bwd step (N=96)",
             "interpreted_ms": round(t_interp, 3),
             "compiled_ms": round(t_comp, 3),
             "compiler_speedup_x": round(ratio, 1),
             "note": "INTERNAL ratio: same engine, same numerics"})
print(f"  interpreted (rebuild+ADGET): {t_interp:.3f} ms")
print(f"  compiled (fused verb):       {t_comp:.3f} ms")
print(f"  compiler speedup:            {ratio:.0f}x  [internal, not vs numpy]")

with open("bench_tensor.json", "w") as f:
    json.dump({"rows": ROWS}, f, indent=2)
print("\nwrote bench_tensor.json")
print("\nHONEST SUMMARY: numpy is faster on raw float64 ops (claim A).")
print("The compiler's ~%.0fx is an INTERNAL win over the interpreted tape" % ratio)
print("(claim B) -- it is not a claim of beating numpy/torch.")