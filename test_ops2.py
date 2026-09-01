"""test_ops2.py — v3 ops (softmax/reshape/relu/transpose) gradchecks.

Usage:
  DYLD_LIBRARY_PATH=../jlibrary/bin PYJ_LIBPATH=../jlibrary/bin python3 test_ops2.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from adt import Graph

fails = 0
def check(label, got, want, tol=1e-5):
    global fails
    ok = np.allclose(got, want, atol=tol, rtol=1e-6)
    if ok:
        print(f"ok   {label}")
    else:
        fails += 1
        print(f"FAIL {label}:\n got {got!r}\n want {want!r}")

rng = np.random.default_rng(23)
eps = 1e-6

# ---- softmax gradcheck (with max-stabilization node in graph) ----
Xn = rng.normal(size=(4, 5))
Cn = np.eye(4, 5)
g = Graph()
X = g.tensor(Xn, "X")
C = g.tensor(Cn, "C")
P = X.softmax()
L = (P * C).sum()
g.back(L)
def loss_sm(Xa):
    E = np.exp(Xa - Xa.max(1, keepdims=True))
    Pn = E / E.sum(1, keepdims=True)
    return (Pn * Cn).sum()
num = np.zeros_like(Xn)
for i in range(Xn.shape[0]):
    for j in range(Xn.shape[1]):
        Pp = Xn.copy(); Mm = Xn.copy(); Pp[i,j] += eps; Mm[i,j] -= eps
        num[i,j] = (loss_sm(Pp) - loss_sm(Mm)) / (2*eps)
check("softmax gradcheck dX", X.grad, num)

# ---- reshape gradcheck: L = sum((reshape X (2,5)) * W) ----
Xn = rng.normal(size=(5, 2))
Wn = rng.normal(size=(2, 5))
g = Graph()
X = g.tensor(Xn, "X")
W = g.tensor(Wn, "W")
Rr = X.reshape(2, 5)
L = (Rr * W).sum()
g.back(L)
def loss_rs(Xa):
    return ((Xa.reshape(2, 5)) * Wn).sum()
num = np.zeros_like(Xn)
for i in range(Xn.shape[0]):
    for j in range(Xn.shape[1]):
        Pp = Xn.copy(); Mm = Xn.copy(); Pp[i,j] += eps; Mm[i,j] -= eps
        num[i,j] = (loss_rs(Pp) - loss_rs(Mm)) / (2*eps)
check("reshape gradcheck dX", X.grad, num)

# ---- transpose in chain: L = sum( (X @ W).T * C ) ----
Xn = rng.normal(size=(2, 3))
Wn = rng.normal(size=(3, 4))
Cn = rng.normal(size=(4, 2))
g = Graph()
X = g.tensor(Xn, "X")
W = g.tensor(Wn, "W")
C = g.tensor(Cn, "C")
L = (((X @ W).T()) * C).sum()
g.back(L)
def loss_tr(Xa):
    return ((Xa @ Wn).T * Cn).sum()
num = np.zeros_like(Xn)
for i in range(2):
    for j in range(3):
        Pp = Xn.copy(); Mm = Xn.copy(); Pp[i,j] += eps; Mm[i,j] -= eps
        num[i,j] = (loss_tr(Pp) - loss_tr(Mm)) / (2*eps)
check("transpose chain gradcheck dX", X.grad, num)

# ---- softmax classifier forward sanity vs numpy ----
Xn = rng.normal(size=(6, 4))
W1n = rng.normal(size=(4, 8)) * 0.5
b1n = rng.normal(size=(8)) * 0.1
W2n = rng.normal(size=(8, 3)) * 0.5
H = np.tanh(Xn @ W1n + b1n)
logits = H @ W2n
E = np.exp(logits - logits.max(1, keepdims=True))
Pn = E / E.sum(1, keepdims=True)
g = Graph()
X = g.tensor(Xn, "X")
W1 = g.tensor(W1n, "W1")
b1 = g.tensor(b1n, "b1")
W2 = g.tensor(W2n, "W2")
P = (X @ W1 + b1).tanh() @ W2
Psm = P.softmax()
mx = Psm.value_now() - Pn
check("softmax fwd vs numpy", mx, np.zeros_like(Pn), tol=1e-12)

# ---- full cross-entropy gradcheck: L = -(sum C*log P) with clipping avoided ----
# use sum(C * log(P)) and negate in Python
Yn = np.eye(6, 3)[rng.integers(0, 3, 6)]
g = Graph()
X = g.tensor(Xn, "X")
W1 = g.tensor(W1n, "W1")
b1 = g.tensor(b1n, "b1")
W2 = g.tensor(W2n, "W2")
Y = g.tensor(Yn, "Y")
Psm = (X @ W1 + b1).tanh() @ W2
Psm = Psm.softmax()
LG = ((Psm * Y).log()).sum()
g.back(LG)
def loss_ce(*params):
    Xa, W1a, b1a, W2a = params
    H = np.tanh(Xa @ W1a + b1a)
    lg = H @ W2a
    E = np.exp(lg - lg.max(1, keepdims=True))
    Pp = E / E.sum(1, keepdims=True)
    return (Yn * np.log(Pp)).sum()
numX = np.zeros_like(Xn)
for i in range(Xn.shape[0]):
    for j in range(Xn.shape[1]):
        Pp = Xn.copy(); Mm = Xn.copy(); Pp[i,j] += eps; Mm[i,j] -= eps
        numX[i,j] = (loss_ce(Pp, W1n, b1n, W2n) - loss_ce(Mm, W1n, b1n, W2n)) / (2*eps)
check("cross-entropy gradcheck dX", X.grad, numX)
numW1 = np.zeros_like(W1n)
for i in range(W1n.shape[0]):
    for j in range(W1n.shape[1]):
        Pp = W1n.copy(); Mm = W1n.copy(); Pp[i,j] += eps; Mm[i,j] -= eps
        numW1[i,j] = (loss_ce(Xn, Pp, b1n, W2n) - loss_ce(Xn, Mm, b1n, W2n)) / (2*eps)
check("cross-entropy gradcheck dW1", W1.grad, numW1)

print()
print("OPS2 TESTS: ALL OK" if fails == 0 else f"OPS2 TESTS: {fails} FAILURES")
sys.exit(1 if fails else 0)
