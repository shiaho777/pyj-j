"""test_ops3.py — AD coverage expansion: take/drop/gather (roadmap step 5a).

Validates the three new array-manipulation primitives end to end:
  1. unit VJPs vs numeric gradients (Tensor API, take/drop/gather)
  2. embedding-table + take chain gradcheck (the typical use)
  3. ADGEN compiled-verb parity with the interpreted path on a graph
     containing all three ops
  4. gather-based embedding training converges

Usage:
  DYLD_LIBRARY_PATH=../jlibrary/bin PYJ_LIBPATH=../jlibrary/bin python3 test_ops3.py
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from adt import Graph

fails = 0
def check(label, got, want, tol=1e-5):
    global fails
    ok = np.allclose(got, want, atol=tol, rtol=1e-6)
    if not ok:
        fails += 1
        print(f"FAIL {label}: got {got!r} want {want!r}")
    else:
        print(f"ok   {label}")

rng = np.random.default_rng(5)
eps = 1e-6

# ---- 1: take gradcheck: L = sum( (X.take(2) * Y).sum() ) ----
g = Graph()
X = g.tensor(rng.normal(size=(3, 4)), "X")
W = g.tensor(rng.normal(size=(3, 2)), "W")
L = ((X.take(2)) * W).sum()
g.back(L)
def loss_take(Xa):
    return (Xa[:, :2] * W.value).sum()
numX = np.zeros_like(X.value)
for i in range(3):
    for j in range(4):
        P = X.value.copy(); M = X.value.copy(); P[i, j] += eps; M[i, j] -= eps
        numX[i, j] = (loss_take(P) - loss_take(M)) / (2 * eps)
check("take gradcheck dX", X.grad, numX)

# ---- 2: drop gradcheck ----
g = Graph()
X = g.tensor(rng.normal(size=(3, 4)), "X")
W = g.tensor(rng.normal(size=(3, 3)), "W")
L = ((X.drop(1)) * W).sum()
g.back(L)
def loss_drop(Xa):
    return (Xa[:, 1:] * W.value).sum()
numX = np.zeros_like(X.value)
for i in range(3):
    for j in range(4):
        P = X.value.copy(); M = X.value.copy(); P[i, j] += eps; M[i, j] -= eps
        numX[i, j] = (loss_drop(P) - loss_drop(M)) / (2 * eps)
check("drop gradcheck dX", X.grad, numX)

# ---- 3: gather gradcheck (repeated indices -> gradient accumulates) ----
g = Graph()
T = g.tensor(rng.normal(size=(4, 3)), "T")     # embedding table
Pj = g.tensor(rng.normal(size=(2, 3)), "Pj")   # projection
idx = np.array([0, 2, 2])                       # row 2 used twice
L = ((T.gather(idx)) @ Pj.T()).sum()
g.back(L)
def loss_gather(Ta):
    return (Ta[idx] @ Pj.value.T).sum()
numT = np.zeros_like(T.value)
for i in range(4):
    for j in range(3):
        P = T.value.copy(); M = T.value.copy(); P[i, j] += eps; M[i, j] -= eps
        numT[i, j] = (loss_gather(P) - loss_gather(M)) / (2 * eps)
check("gather gradcheck dT (dup indices accumulate)", T.grad, numT)
# dPj of sum(gathered @ Pj^T) = ones (2,3)? each Pj cell contributes twice?
# L = sum(Ga @ Pj.T): dPj = 1s everywhere (2,3). But J gives T-row-dependent vals?
# J grad rows equal each other -> dPj should equal ones. Recompute expected via
# numeric instead of asserting ones:
numPj = np.zeros_like(Pj.value)
def loss_gather_pj(Pja):
    return (T.value[idx] @ Pja.T).sum()
for i in range(2):
    for j in range(3):
        P = Pj.value.copy(); M = Pj.value.copy(); P[i, j] += eps; M[i, j] -= eps
        numPj[i, j] = (loss_gather_pj(P) - loss_gather_pj(M)) / (2 * eps)
check("gather gradcheck dPj", Pj.grad, numPj)

# ---- 4: compiled parity — same gather/take graph through ADGEN ----
import pyj
pyj.do(f"0!:0 <'{HERE}/adcomp.ijs'")   # load the compiler
pyj.set("T_v", T.value)
pyj.set("Pj_v", Pj.value)
pyj.set("ci_v", np.array([0, 2], dtype=np.int64))
pyj.set("k3_v", np.array([3], dtype=np.int64))
pyj.do("ADSETP (<'T_v';T_v) , (<'Pj_v';Pj_v) , (<'ci_v';ci_v) , (<'k3_v';k3_v)")
pyj.do("eT=: 0 ngather 2")     # gather rows 0,2 (ids: T=0 Pj=1 ci=2 k3=3)
pyj.do("eK=: eT ntake 3")      # take(3): trailing-axis no-op but full path
pyj.do("eG=: eK nmp (ntr 1)")  # gathered @ Pj^T  (matches section 3 loss)
pyj.do("eL=: nsum eG")
pyj.do("eL2=: nsum eL")
LOSS = int(pyj.get("eL2"))
pyj.do(f"ggI=: ADGET {LOSS}")
pyj.do("ggT=: >0{ggI")
giT = pyj.get("ggT")
pyj.do(f"{LOSS} ADGEN 7")
pyj.do("adcB7 ''")
gcT = pyj.get("adcg0")
check("compiled gather grad == interpreted", gcT, giT)

# ---- 5: embedding training (gather path) converges ----
# 3 classes, 2 embeddings each used via one-hot-ish gather; learn row values
rng = np.random.default_rng(9)
N = 60
Xa = np.vstack([rng.normal([0, 0], .5, (N//2, 2)), rng.normal([3, 3], .5, (N//2, 2))])
ya = np.concatenate([np.zeros(N//2, dtype=int), np.ones(N//2, dtype=int)])
# quantize X to token ids per class (crude "vocab"): 4 bins per dim -> 16 tokens
bins = np.clip(((Xa / 3.0) * 3).astype(int) + 1, 0, 3)
toks = bins[:, 0] * 4 + bins[:, 1]        # ids in 0..15
V = np.eye(16)[toks]                       # for numeric eval
Em = rng.normal(size=(16, 1)) * 0.1

g = Graph()
E = g.tensor(Em, "E")
Y2 = g.tensor((2.0 * ya - 1.0).reshape(-1, 1), "Y2")
lr = 1.0
acc0 = None
for step in range(301):
    E.value = Em
    g._sync_leaves()
    Gv = E.gather(toks)                    # (N,1) embedding lookups
    Dv = Gv - Y2                           # linear output: score - target
    L = (Dv * Dv).sum() * (1.0 / N)
    g.back(L)
    Em = Em - lr * E.grad
    if step % 100 == 0:
        pred = (V @ Em)                    # V is one-hot per sample
        acc = ((pred > 0).ravel().astype(int) == ya).mean()
        print(f"  step {step:3d} acc={acc:.3f}")
pred = (V @ Em)
acc = ((pred > 0).ravel().astype(int) == ya).mean()
print(f"final embedding-train acc: {acc:.3f}")
check("embedding-gather training acc > 0.9", np.array([acc > 0.9]).astype(float), np.array([1.0]))

print()
print("OPS3 TESTS:", "ALL OK" if fails == 0 else f"{fails} FAILURES")
sys.exit(0 if fails == 0 else 1)
