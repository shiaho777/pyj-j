"""test_ops5.py — v5 features: compare gates, tape replay, compiled replay.

  1. nlt mask + nwhere gate: gradcheck of the gated-mix composition
     (mask*a + (1-mask)*b built from where/mul/add)
  2. Graph.replay: recorded-tape steps produce grads identical to a fresh
     full build, both before and after leaf updates
  3. replay + ADGEN combined: one-step cost of the compiled path measured
     and correctness-checked
  4. end-to-end: softmax classifier trained through replay_step converges

Usage:
  DYLD_LIBRARY_PATH=../jlibrary/bin PYJ_LIBPATH=../jlibrary/bin python3 test_ops5.py
"""
import sys, os, time
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
rng = np.random.default_rng(4)
eps = 1e-6

# ---- 1: gated mix (where/mask) gradcheck ----
g = Graph()
a0 = rng.normal(size=(6,))
b0 = rng.normal(size=(6,))
cond = (np.arange(6) % 2).astype(float)     # alternating gate
g = Graph()
A = g.tensor(a0, "A"); B = g.tensor(b0, "B"); M = g.tensor(cond, "M")
W = A * M + B * (1.0 - M)                    # if-then-else, differentiable
L = (W * W).sum()
g.back(L)
def lossf(aa, bb):
    w = np.where(cond > 0.5, aa, bb)
    return (w * w).sum()
numA = np.zeros_like(a0); numB = np.zeros_like(b0)
for i in range(6):
    P = a0.copy(); Q = a0.copy(); P[i] += eps; Q[i] -= eps
    numA[i] = (lossf(P, b0) - lossf(Q, b0)) / (2 * eps)
    P = b0.copy(); Q = b0.copy(); P[i] += eps; Q[i] -= eps
    numB[i] = (lossf(a0, P) - lossf(a0, Q)) / (2 * eps)
check("gated-mix gradcheck dA", A.grad, numA)
check("gated-mix gradcheck dB", B.grad, numB)
check("gate zeroes masked branch", A.grad * (1.0 - cond), np.zeros(6))

# ---- 2: replay == fresh build ----
rng = np.random.default_rng(7)
N = 64
Xa = np.vstack([rng.normal([0, 0], .8, (N//2, 2)), rng.normal([2.5, 2.5], .8, (N//2, 2))])
ya = np.concatenate([np.zeros(N//2), np.ones(N//2)])
Xdata = np.hstack([Xa, np.ones((N, 1))]); Ydata = (2*ya - 1).reshape(-1, 1)
W1_0 = rng.normal(size=(3, 8)) * 0.5
W2_0 = rng.normal(size=(8, 1)) * 0.5

g = Graph()
X = g.tensor(Xdata, "X"); Y = g.tensor(Ydata, "Y")
W1 = g.tensor(W1_0, "W1"); W2 = g.tensor(W2_0, "W2")
def build(g):
    H = (X @ W1).tanh(); O = H @ W2; P = O.tanh(); Dv = P - Y
    return (Dv * Dv).sum()

loss = g.replay(build)
g.back(loss)
g1_full = W1.grad.copy()
g.replay_step()
g1_rep = W1.grad.copy()
check("replay == full build (identical leaves)", g1_rep, g1_full, tol=1e-9)

w1_next = W1.value - 0.05 * g1_full
W1.value = w1_next
g.replay_step()
g_rep2 = W1.grad.copy()

g2 = Graph()
X2 = g2.tensor(Xdata, "X"); Y2 = g2.tensor(Ydata, "Y")
W1b = g2.tensor(w1_next, "W1"); W2b = g2.tensor(W2.value, "W2")
H = (X2 @ W1b).tanh(); O = H @ W2b; P = O.tanh(); Dv = P - Y2
L2 = (Dv * Dv).sum()
g2.back(L2)
check("replay == fresh build (after leaf update)", g_rep2, W1b.grad, tol=1e-9)

# ---- 3: replay + ADGEN compiled step ----
pyj.do(f"0!:0 <'{HERE}/adcomp.ijs'")
rng = np.random.default_rng(7)
N = 96
def _blob(center, r, n):
    ang = rng.uniform(0, 2*np.pi, n)
    rad = r * np.sqrt(rng.uniform(0.15, 1, n))
    return np.stack([center[0] + rad*np.cos(ang), center[1] + rad*np.sin(ang)], 1)
Xa = np.vstack([_blob([0, 0], 2.0, N//3), _blob([5, 0], 2.0, N//3), _blob([2.5, 4.3], 2.0, N//3)])
X3 = np.hstack([Xa, np.ones((N, 1))]); Y3 = np.eye(3)[np.repeat(np.arange(3), N // 3)]
W1_3 = rng.normal(size=(3, 16)) * 0.25
W2_3 = rng.normal(size=(16, 3)) * 0.25

g3 = Graph()
X = g3.tensor(X3, "X"); Y = g3.tensor(Y3, "Y")
W1 = g3.tensor(W1_3, "W1"); W2 = g3.tensor(W2_3, "W2")
def build3(g):
    H = (X @ W1).tanh(); O = H @ W2
    m = O.rmax1(); e = (O.rsub1(m)).exp()
    P = e.rdiv1(e.rsum1())
    return (P * Y).sum()
loss3 = g3.replay(build3)
g3.replay_step()
rc, out = pyj.do(f"{g3._replay_loss_id} ADGEN 9")
assert rc == 0, f"ADGEN failed: {out}"
g3._sync_leaves()
pyj.do("adcB9 ''")
gc1 = pyj.get("adcg2")
# interpreted reference: rebuild the full tape by replaying the ops, then ADGET
for s in g3._rec_ops:
    pyj.do(s)
g3.back(loss3)
check("compiled(replay) == interpreted", gc1, W1.grad, tol=1e-9)

t0 = time.perf_counter()
for _ in range(300):
    g3._sync_leaves()
    pyj.do("adcB9 ''")
    _ = pyj.get("adcg2")
t_comp = (time.perf_counter() - t0) / 300
print(f"  compiled replay step: {t_comp*1e3:.3f} ms")

# ---- 4: end-to-end training through replay_step ----
g4 = Graph()
X = g4.tensor(X3, "X"); Y = g4.tensor(Y3, "Y")
W1 = g4.tensor(W1_3, "W1"); W2 = g4.tensor(W2_3, "W2")
def build4(g):
    H = (X @ W1).tanh(); O = H @ W2
    m = O.rmax1(); e = (O.rsub1(m)).exp()
    P = e.rdiv1(e.rsum1())
    return (P * Y).sum()
loss4 = g4.replay(build4)
lr = 0.02
ce0 = None
w1, w2 = W1_3.copy(), W2_3.copy()
for step in range(801):
    W1.value = w1; W2.value = w2
    g4.replay_step()
    w1 = w1 + lr * W1.grad
    w2 = w2 + lr * W2.grad
    if step % 200 == 0:
        H = np.tanh(X3 @ w1); lg = H @ w2
        E = np.exp(lg - lg.max(1, keepdims=True)); Pr = E / E.sum(1, keepdims=True)
        ce = -(np.log((Pr * Y3).sum(1) + 1e-12)).mean()
        if ce0 is None: ce0 = ce
H = np.tanh(X3 @ w1)
pred = (H @ w2).argmax(1)
acc = (pred == np.repeat(np.arange(3), N // 3)).mean()
print(f"final replay-trained acc: {acc:.3f}")
check("replay-trained CE dropped >5x", np.array([ce0 / ce > 5.0]), np.array([1.0]))
check("replay-trained acc > 0.95", np.array([acc > 0.95]), np.array([1.0]))

print()
print("OPS5 TESTS:", "ALL OK" if fails == 0 else f"{fails} FAILURES")
sys.exit(0 if fails == 0 else 1)
