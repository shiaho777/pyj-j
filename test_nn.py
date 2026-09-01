"""test_nn.py — Tensor API (adt.py) end-to-end: unit grads, gradcheck, training.

Usage:
  DYLD_LIBRARY_PATH=../jlibrary/bin PYJ_LIBPATH=../jlibrary/bin python3 test_nn.py
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
        print(f"FAIL {label}: got {got!r} want {want!r}")

rng = np.random.default_rng(11)

# ---- 1: matmul grads ----
g = Graph()
X = g.tensor(rng.normal(size=(2, 3)), "X")
W = g.tensor(rng.normal(size=(3, 2)), "W")
L = (X @ W).sum()
g.back(L)
gX, gW = X.grad, W.grad
numX = np.zeros_like(X.value); numW = np.zeros_like(W.value)
eps = 1e-6
lossf = lambda A, B: (A @ B).sum()
for i in range(2):
    for j in range(3):
        P = X.value.copy(); M = X.value.copy(); P[i,j]+=eps; M[i,j]-=eps
        numX[i,j] = (lossf(P, W.value) - lossf(M, W.value)) / (2*eps)
for i in range(3):
    for j in range(2):
        P = W.value.copy(); M = W.value.copy(); P[i,j]+=eps; M[i,j]-=eps
        numW[i,j] = (lossf(X.value, P) - lossf(X.value, M)) / (2*eps)
check("matmul gradcheck dX", gX, numX)
check("matmul gradcheck dW", gW, numW)

# ---- 2: MLP layer with bias + tanh + mean-square loss ----
# L = sum( (tanh(X @ W + b) - Y)^2 )
g = Graph()
X = g.tensor(rng.normal(size=(4, 3)), "X")
W = g.tensor(rng.normal(size=(3, 2)), "W")
b = g.tensor(rng.normal(size=(2,)), "b")
Y = g.tensor(rng.normal(size=(4, 2)), "Y")
Z = X @ W + b          # badd broadcast
H = Z.tanh()
D = H - Y
L = (D * D).sum()
g.back(L)
def loss_fn(Xa, Wa, ba, Ya):
    return ((np.tanh(Xa @ Wa + ba) - Ya) ** 2).sum()
numW = np.zeros_like(W.value)
for i in range(3):
    for j in range(2):
        P = W.value.copy(); M = W.value.copy(); P[i,j]+=eps; M[i,j]-=eps
        numW[i,j] = (loss_fn(X.value,P,b.value,Y.value) - loss_fn(X.value,M,b.value,Y.value))/(2*eps)
numb = np.zeros_like(b.value)
for i in range(2):
    P = b.value.copy(); M = b.value.copy(); P[i]+=eps; M[i]-=eps
    numb[i] = (loss_fn(X.value,W.value,P,Y.value) - loss_fn(X.value,W.value,M,Y.value))/(2*eps)
check("MLP gradcheck dW", W.grad, numW)
check("MLP gradcheck db", b.grad, numb)

# ---- 3: relu network gradcheck (avoid kinks: shift inputs) ----
g = Graph()
X = g.tensor(rng.normal(0.5, 1.0, size=(3, 3)) + 2.0, "X")
W = g.tensor(rng.normal(size=(3, 2)), "W")
L = ((X @ W).relu()).sum()
g.back(L)
lossf2 = lambda A, B: np.maximum(A @ B, 0).sum()
numX = np.zeros_like(X.value)
for i in range(3):
    for j in range(3):
        P = X.value.copy(); M = X.value.copy(); P[i,j]+=eps; M[i,j]-=eps
        numX[i,j] = (lossf2(P, W.value) - lossf2(M, W.value)) / (2*eps)
check("relu gradcheck dX", X.grad, numX)

# ---- 4: transpose path: L = sum( (X @ W).T ) ----
g = Graph()
X = g.tensor(rng.normal(size=(2, 3)), "X")
W = g.tensor(rng.normal(size=(3, 2)), "W")
L = (X @ W).T().sum()
g.back(L)
lossf3 = lambda A, B: (A @ B).T.sum()
numX = np.zeros_like(X.value)
for i in range(2):
    for j in range(3):
        P = X.value.copy(); M = X.value.copy(); P[i,j]+=eps; M[i,j]-=eps
        numX[i,j] = (lossf3(P, W.value) - lossf3(M, W.value)) / (2*eps)
check("transpose gradcheck dX", X.grad, numX)

# ---- 5: training — 2-layer MLP with tanh + sigmoid MSE on blobs ----
rng = np.random.default_rng(42)
N = 64
Xa = np.vstack([rng.normal([0, 0], .8, (N//2, 2)), rng.normal([2.5, 2.5], .8, (N//2, 2))])
ya = np.concatenate([np.zeros(N//2), np.ones(N//2)])
Xdata = np.hstack([Xa, np.ones((N, 1))])          # bias column
Ydata = (2 * ya - 1).reshape(-1, 1)   # tanh targets: -1/+1

g = Graph()
X = g.tensor(Xdata, "X")
Y = g.tensor(Ydata, "Y")
W1 = g.tensor(rng.normal(size=(3, 8)) * 0.5, "W1")
W2 = g.tensor(rng.normal(size=(8, 1)) * 0.5, "W2")

# simpler: do it without helper closure
w1 = W1.value.copy(); w2 = W2.value.copy()
lr = 0.05
for step in range(600):
    W1.value = w1; W2.value = w2
    g._sync_leaves()
    H = (X @ W1).tanh()
    O = H @ W2
    P = O.tanh()
    Dv = P - Y
    L = (Dv * Dv).sum()
    g.back(L)
    w1 = w1 - lr * W1.grad
    w2 = w2 - lr * W2.grad
    if step % 50 == 0 or step == 199:
        import numpy as _np
        pv = _np.tanh(_np.tanh(Xdata @ w1) @ w2)
        print(f"step {step} loss {_np.mean((pv-Ydata)**2):.4f}")
pred = (np.tanh(np.tanh(Xdata @ w1) @ w2) > 0).astype(float).ravel()
acc = (pred == ya).mean()
check("training: 2-layer MLP acc > 0.9", np.array([acc > 0.9]).astype(float), np.array([1.0]))
print(f"final train acc: {acc:.2f}")

print()
print("NN TESTS: ALL OK" if fails == 0 else f"NN TESTS: {fails} FAILURES")
sys.exit(1 if fails else 0)
