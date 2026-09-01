"""test_ops4.py — rank dispatch: sum over an arbitrary axis (roadmap 5b).

nrsumr rotates axis k to the end and applies +/"1; the VJP replicates each
frame scalar across the reduced axis (ravel order) and un-rotates. The
compiled path mirrors this exactly.

  1. gradcheck vs numeric across shapes (1D-4D) and axes
  2. Tensor API gradcheck
  3. compiled-verb parity with the interpreted tape
  4. use case: batched row softmax over a 3D tensor (axis 2)

Usage:
  DYLD_LIBRARY_PATH=../jlibrary/bin PYJ_LIBPATH=../jlibrary/bin python3 test_ops4.py
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pyj
from adt import Graph

fails = 0
def check(label, got, want, tol=1e-5):
    global fails
    ok = np.allclose(got, want, atol=tol, rtol=1e-6)
    if not ok:
        fails += 1
        print(f"FAIL {label}")
    else:
        print(f"ok   {label}")

pyj.do(f"0!:0 <'{HERE}/ad.ijs'")
pyj.do(f"0!:0 <'{HERE}/adcomp.ijs'")
rng = np.random.default_rng(3)
eps = 1e-6

# ---- 1: numeric gradcheck over shapes/axes (raw tape path) ----
def tape_gradcheck(m, k):
    W = rng.normal(size=m.sum(k).shape)
    pyj.set("m_v", m); pyj.set("k_v", np.array(k, dtype=np.int64)); pyj.set("W_v", W)
    pyj.do("ADSETP (<'m_v';m_v) , (<'k_v';k_v) , (<'W_v';W_v)")
    pyj.do("s2=: 0 nrsumr 1")
    pyj.do("L2=: nsum (s2 nmul 2)")
    rc, out = pyj.do("gg2=: ADGET L2")
    assert rc == 0, out
    pyj.do("gm2=: >0{gg2")
    gJ = pyj.get("gm2")
    gN = np.zeros_like(m)
    it = np.nditer(m, flags=['multi_index'])
    while not it.finished:
        ix = it.multi_index
        P = m.copy(); M = m.copy()
        P[ix] += eps; M[ix] -= eps
        gN[ix] = ((P.sum(k)*W).sum() - (M.sum(k)*W).sum()) / (2*eps)
        it.iternext()
    return np.allclose(gJ, gN, atol=1e-4)

n_ok = 0
for shp, k in [((4,), 0), ((3, 5), 0), ((3, 5), 1), ((2, 3, 4), 0),
               ((2, 3, 4), 1), ((2, 3, 4), 2), ((2, 3, 4, 5), 1), ((2, 3, 4, 5), 3)]:
    m = rng.normal(size=shp)
    if tape_gradcheck(m, k):
        n_ok += 1
    else:
        fails += 1
        print(f"FAIL tape gradcheck {shp} axis {k}")
check("tape gradchecks (8 shape/axis combos)", np.array([n_ok]), np.array([8]))

# ---- 2: Tensor API gradcheck ----
g = Graph()
m = rng.normal(size=(2, 3, 4))
W0 = rng.normal(size=(2, 4))
M = g.tensor(m, "M")
W = g.tensor(W0, "W")
L = (M.sum_axis(1) * W).sum()
g.back(L)
def lossf(ma):
    return (ma.sum(1) * W0).sum()
numM = np.zeros_like(m)
it = np.nditer(m, flags=['multi_index'])
while not it.finished:
    ix = it.multi_index
    P = m.copy(); Mv = m.copy()
    P[ix] += eps; Mv[ix] -= eps
    numM[ix] = (lossf(P) - lossf(Mv)) / (2*eps)
    it.iternext()
check("Tensor API gradcheck sum_axis(1)", M.grad, numM)

# ---- 3: compiled parity ----
pyj.set("cp_m", m); pyj.set("cp_k", np.array(1, dtype=np.int64)); pyj.set("cp_W", W0)
pyj.do("ADSETP (<'cp_m';cp_m) , (<'cp_k';cp_k) , (<'cp_W';cp_W)")
pyj.do("cs=: 0 nrsumr 1")
pyj.do("cm=: cs nmul 2")
pyj.do("cl=: nsum cm")
LOSS = int(pyj.get("cl"))
pyj.do(f"ggI=: ADGET {LOSS}")
pyj.do("ggT=: >0{ggI")
giT = pyj.get("ggT")
pyj.do(f"{LOSS} ADGEN 4")
rc, out = pyj.do("adcB4 ''")
assert rc == 0, out
check("compiled sum_axis grad == interpreted", pyj.get("adcg0"), giT)

# ---- 4: use case — batched softmax over last axis of a 3D tensor ----
# B: (batch, row, col-class) -> softmax over axis 2, then sum over batch
g = Graph()
B0 = rng.normal(size=(2, 3, 4))
B = g.tensor(B0, "B")
mx = B.rmax1()                    # per-row max of the trailing axis
e = (B.rsub1(mx)).exp()
P = e.rdiv1(e.rsum1())            # softmax along the trailing axis
Ls = P.sum()
g.back(Ls)
# numeric: d(sum P)/dB
numB = np.zeros_like(B0)
it = np.nditer(B0, flags=['multi_index'])
while not it.finished:
    ix = it.multi_index
    P0 = B0.copy(); M0 = B0.copy()
    P0[ix] += eps; M0[ix] -= eps
    def soft(x):
        e = np.exp(x - x.max(-1, keepdims=True))
        return (e / e.sum(-1, keepdims=True)).sum()
    numB[ix] = (soft(P0) - soft(M0)) / (2*eps)
    it.iternext()
check("batched-softmax gradcheck (rmax1/rsub1/rdiv1 chain on rotated order)",
      B.grad, numB, tol=1e-4)

print()
print("OPS4 TESTS:", "ALL OK" if fails == 0 else f"{fails} FAILURES")
sys.exit(0 if fails == 0 else 1)
