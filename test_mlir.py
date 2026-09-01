"""test_mlir.py — MLIR backend: export the AD tape, lower to LLVM, execute,
compare against the J engine's own forward values.

The exporter (adexport.py) supports float64 elementwise ops, matmul and
reductions. This suite validates:

  1. matmul + leading-axis sum
  2. tanh MLP forward (X@W1 -> tanh -> @W2)
  3. softmax block (rmax1/rsub1/exp/rsum1/rdiv1 chain needs rdiv1 — exported
     as div; rsum1 as trailing reduce)
  4. gated mix (where = mask*value)
  5. transpose, reshape, take, drop, gather (embedding lookup)

Execution requires the LLVM suite (mlir-opt, mlir-translate) and clang.
Set MLIR_BIN if they are not in /opt/homebrew/opt/llvm/bin. The test skips
itself when the toolchain is missing.

Usage:
  DYLD_LIBRARY_PATH=../jlibrary/bin PYJ_LIBPATH=../jlibrary/bin python3 test_mlir.py
"""
import sys, os, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pyj
from adt import Graph
from adexport import export_and_run

fails = 0
def check(label, got, want, tol=1e-6):
    global fails
    ok = np.allclose(got, want, atol=tol, rtol=1e-6)
    if not ok:
        fails += 1
        print(f"FAIL {label}: got {got!r} want {want!r}")
    else:
        print(f"ok   {label}")

pyj.do(f"0!:0 <'{HERE}/ad.ijs'")

MLIR_BIN = os.environ.get("MLIR_BIN", "/opt/homebrew/opt/llvm/bin")
HAVE_MLIR = all(shutil.which(os.path.join(MLIR_BIN, t)) or os.path.exists(os.path.join(MLIR_BIN, t))
                for t in ("mlir-opt", "mlir-translate")) and shutil.which("clang")
if not HAVE_MLIR:
    print("MLIR toolchain not found — skipping test_mlir (set MLIR_BIN to enable)")
    sys.exit(0)

def j_forward(loss_id):
    pyj.do(f"mlir_ref=: nval {loss_id}")
    return pyj.get("mlir_ref")

rng = np.random.default_rng(3)

# ---- 1: matmul + sum (leading axis) ----
X0 = rng.normal(size=(3, 4)); W0 = rng.normal(size=(4, 2))
g = Graph()
X = g.tensor(X0, "X"); W = g.tensor(W0, "W")
L = (X @ W).sum()
g.back(L)
ref = j_forward(L.id)
got = export_and_run(["X", "W"], L.id, {"X": X0, "W": W0})
check("MLIR matmul+sum", got, ref)

# ---- 2: 2-layer tanh MLP forward ----
X0 = rng.normal(size=(4, 3)); W1_0 = rng.normal(size=(3, 8)); W2_0 = rng.normal(size=(8, 2))
g = Graph()
X = g.tensor(X0, "X"); W1 = g.tensor(W1_0, "W1"); W2 = g.tensor(W2_0, "W2")
L = ((X @ W1).tanh() @ W2).sum()
g.back(L)
ref = j_forward(L.id)
got = export_and_run(["X", "W1", "W2"], L.id, {"X": X0, "W1": W1_0, "W2": W2_0})
check("MLIR tanh-MLP forward", got, ref)

# ---- 3: softmax row-normalization (rsub1/exp/rsum1/rdiv1 on 2D) ----
X0 = rng.normal(size=(5, 4)) * 3
g = Graph()
X = g.tensor(X0, "X")
m = X.rmax1()
e = (X.rsub1(m)).exp()
P = e.rdiv1(e.rsum1())
L = (P * 2.0).sum()
g.back(L)
ref = j_forward(L.id)
got = export_and_run(["X"], L.id, {"X": X0})
check("MLIR softmax chain", got, ref)

# ---- 4: gated mix ----
A0 = rng.normal(size=(6,)); B0 = rng.normal(size=(6,))
cond = (np.arange(6) % 2).astype(float)
g = Graph()
A = g.tensor(A0, "A"); B = g.tensor(B0, "B"); M = g.tensor(cond, "M")
Wv = A * M + B * (1.0 - M)
L = (Wv * Wv).sum()
g.back(L)
ref = j_forward(L.id)
got = export_and_run(["A", "B", "M"], L.id, {"A": A0, "B": B0, "M": cond})
check("MLIR gated mix", got, ref)

# ---- 5: shape ops ----
M0 = rng.normal(size=(2, 3)); E0 = rng.normal(size=(8,)); toks = np.array([0, 2, 2, 5])

g = Graph(); M = g.tensor(M0, "M")
L = M.transpose().sum(); g.back(L)
check("MLIR transpose", export_and_run(["M"], L.id, {"M": M0}), j_forward(L.id))

g = Graph(); M = g.tensor(M0, "M")
L = M.reshape(3, 2).sum(); g.back(L)
check("MLIR reshape", export_and_run(["M"], L.id, {"M": M0}), j_forward(L.id))

g = Graph(); M = g.tensor(M0, "M")
L = M.drop(1).sum(); g.back(L)
check("MLIR drop", export_and_run(["M"], L.id, {"M": M0}), j_forward(L.id))

g = Graph(); M = g.tensor(M0, "M")
L = M.take(2).sum(); g.back(L)
check("MLIR take", export_and_run(["M"], L.id, {"M": M0}), j_forward(L.id))

g = Graph(); E = g.tensor(E0, "E")
L = E.gather(toks).sum(); g.back(L)
check("MLIR gather", export_and_run(["E"], L.id, {"E": E0}), j_forward(L.id))

print()
print("MLIR TESTS:", "ALL OK" if fails == 0 else f"{fails} FAILURES")
sys.exit(0 if fails == 0 else 1)
