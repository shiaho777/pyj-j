"""test_train2.py — 3-class softmax classifier trained end-to-end in the J engine.

Cross-entropy loss (log-softmax form), 2-layer MLP: 2 -> 16 -> 3, tanh hidden.
If this converges, the tape AD supports real classifier training.

Usage:
  DYLD_LIBRARY_PATH=../jlibrary/bin PYJ_LIBPATH=../jlibrary/bin python3 test_train2.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from adt import Graph

rng = np.random.default_rng(7)

# 3-class spiral-ish blobs
N = 96
def blob(center, r, n):
    ang = rng.uniform(0, 2*np.pi, n)
    rad = r * np.sqrt(rng.uniform(0.15, 1, n))
    return np.stack([center[0] + rad*np.cos(ang), center[1] + rad*np.sin(ang)], 1)

Xa = np.vstack([blob([0, 0], 2.0, N//3), blob([5, 0], 2.0, N//3), blob([2.5, 4.3], 2.0, N//3)])
ya = np.repeat(np.arange(3), N//3)
Xdata = np.hstack([Xa, np.ones((len(Xa), 1))])     # bias column
Ydata = np.eye(3)[ya]                              # one-hot

g = Graph()
X = g.tensor(Xdata, "X")
Y = g.tensor(Ydata, "Y")
W1 = g.tensor(rng.normal(size=(3, 16)) * 0.25, "W1")
W2 = g.tensor(rng.normal(size=(16, 3)) * 0.25, "W2")

def forward(w1, w2):
    """Rebuild the graph each step. Cross-entropy via log-sum-exp (stable, no log(0)):
    L = -sum(Y * (logits - (rowmax + log(sum(exp(logits - rowmax))))))"""
    W1.value = w1; W2.value = w2
    g._sync_leaves()
    H = (X @ W1).tanh()
    logits = H @ W2
    m = logits.rmax1()
    e = (logits.rsub1(m)).exp()
    s = e.rsum1()                    # per-row sum (N,)
    lse = m + s.log()                # logsumexp = rowmax + log(sum(exp(logits-rowmax)))
    shifted = logits.rsub1(lse)      # = log softmax (N,3)
    LG = ((shifted * Y).sum()) * (1.0 / len(Xdata))   # MEAN log-softmax term; CE = -LG
    return LG

# mean loss => mean-scale gradients; lr=0.3 with plain GD saturates the tanh
# hidden layer (|w1| grows past 10 => |H| ~ 1, dead units) before the classes
# separate. lr=0.1 keeps every weight finite and the margin healthy.
lr = 0.1
w1, w2 = W1.value.copy(), W2.value.copy()
for step in range(601):
    LG = forward(w1, w2)
    g.back(LG)
    # LG = mean(Y*logP) is the NEGATIVE loss; ascend it
    if step < 3:
        print(f"  step{step} |gW1|={np.abs(W1.grad).max():.5f} |gW2|={np.abs(W2.grad).max():.5f}")
    w1 = w1 + lr * W1.grad
    w2 = w2 + lr * W2.grad
    if step % 100 == 0:
        # note: Apple Accelerate's matmul raises spurious FP flags on clean
        # inputs (verified: fresh, finite arrays still emit the warning); the
        # finite checks below are the real guard.
        assert np.isfinite(w1).all() and np.isfinite(w2).all(), f"weights exploded at step {step}"
        H = np.tanh(Xdata @ w1)
        lg = H @ w2
        E = np.exp(lg - lg.max(1, keepdims=True))
        P = E / E.sum(1, keepdims=True)
        ce = -np.mean(np.log((P * Ydata).sum(1) + 1e-12))
        print(f"step {step:3d}  mean CE={ce:.4f}  |w1|max={np.abs(w1).max():.2f}")

H = np.tanh(Xdata @ w1)
lg = H @ w2
pred = lg.argmax(1)
acc = (pred == ya).mean()
print(f"final train acc: {acc:.3f}")

ok = acc > 0.95
print("TRAIN2 TESTS:", "ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
