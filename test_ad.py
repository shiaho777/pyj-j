"""pyj AD correctness tests: tape-based reverse-mode AD in J, driven from Python.

Usage:
  DYLD_LIBRARY_PATH=../jlibrary/bin PYJ_LIBPATH=../jlibrary/bin python3 test_ad.py
"""
import sys, os, faulthandler
faulthandler.enable()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyj

HERE = os.path.dirname(os.path.abspath(__file__))

fails = 0
def check(label, got, want, tol=1e-9):
    global fails
    ok = np.allclose(got, want, atol=tol, rtol=1e-7)
    if not ok:
        fails += 1
        print(f"FAIL {label}: got {got!r} want {want!r}")
    else:
        print(f"ok   {label}")

# ---- load AD library into J ----
pyj.do(f"0!:0 <'{HERE}/ad.ijs'")

def run_graph(inputs: dict, builder, order):
    """Set leaves, build graph via builder(dict name->id), backprop, return grads dict."""
    names = ' '.join(f"'{k}'" for k in order)
    pyj.do(f"ADSET {names}")
    ids = {k: i for i, k in enumerate(order)}
    loss_id = builder(ids)
    rc, _ = pyj.do(f"gg=: ADGET {loss_id}")
    grads = {}
    for i, k in enumerate(order):
        grads[k] = pyj.do(f"ggx=: >{i}{{gg")[0] and pyj.get("ggx")
    return grads

def grads_of(order):
    out = []
    for i in range(len(order)):
        pyj.do(f"ggx=: >{i}{{gg")
        out.append(pyj.get("ggx"))
    return out

# ---- test 1: x^4 at x=3 -> 108 ----
pyj.set("x", np.array(3.0))
pyj.do("ADSET <'x'")
pyj.do("r=: nsq (0 nmul 0)")
pyj.do("gg=: ADGET r")
pyj.do("ggx=: >0{gg")
check("d/dx x^4 at 3", pyj.get("ggx"), np.array(108.0))

# ---- test 2: sum(x*x) at x=3 -> 6 ----
pyj.do("t=: nsum (0 nmul 0)")
pyj.do("gg=: ADGET t")
pyj.do("ggx=: >0{gg")
check("d/dx sum(x*x) at 3", pyj.get("ggx"), np.array(6.0))

# ---- test 3: matmul VJP L = sum(X mp W) ----
X = np.array([[1., 2., 3.], [4., 5., 6.]])
W = np.array([[.5, 1.], [1., .5], [2., 1.]])
pyj.set("X", X); pyj.set("W", W)
pyj.do("ADSET 'X';'W'")
pyj.do("L=: nsum (0 nmp 1)")
pyj.do("gg=: ADGET L")
pyj.do("gX=: >0{gg"); pyj.do("gW=: >1{gg")
check("dL/dX", pyj.get("gX"), np.ones((2, 2)) @ W.T)
check("dL/dW", pyj.get("gW"), X.T @ np.ones((2, 2)))

# ---- test 4: tanh(x^2) at 1.5 ----
pyj.set("y4", np.array(1.5))
pyj.do("ADSET <'y4'")
pyj.do("h=: ntanh (nsq 0)")
pyj.do("gg=: ADGET h")
pyj.do("ggx=: >0{gg")
t = np.tanh(1.5**2)
check("d/dx tanh(x^2) at 1.5", pyj.get("ggx"), np.array(2*1.5*(1-t*t)))

# ---- test 5: max VJP ----
pyj.set("v5", np.array([-1., 2., -3., 4.]))
pyj.do("ADSET <'v5'")
pyj.do("m5=: nmax 0")
pyj.do("gg=: ADGET m5")
pyj.do("ggx=: >0{gg")
check("d/dx max(x)", pyj.get("ggx"), (np.array([-1., 2., -3., 4.]) == 4).astype(float))

# ---- test 6: sigmoid chain sig(x)=1/(1+^-x), L=sum ----
x6 = np.array([1.0, 2.0])
pyj.set("x6", x6)
pyj.do("ADSET <'x6'")
pyj.do("c1=: nleaf 1")          # constant leaf
pyj.do("e6=: nexp (nneg 0)")    # ^-x
pyj.do("d6=: c1 nadd e6")       # 1+^-x
pyj.do("s6=: c1 ndiv d6")       # sigmoid
pyj.do("l6=: nsum s6")
pyj.do("gg=: ADGET l6")
pyj.do("ggx=: >0{gg")
sig = 1/(1+np.exp(-x6))
check("d/dx sum(sigmoid(x))", pyj.get("ggx"), sig*(1-sig))

# ---- test 7: numerical gradcheck on a small MLP layer ----
# L = sum( tanh( X mp W ) ) with random X (2x3), W (3x2)
rng = np.random.default_rng(7)
Xr = rng.normal(size=(2, 3)); Wr = rng.normal(size=(3, 2))
pyj.set("X", Xr); pyj.set("W", Wr)
pyj.do("ADSET 'X';'W'")
pyj.do("L7=: nsum (ntanh (0 nmp 1))")
pyj.do("gg=: ADGET L7")
pyj.do("gX=: >0{gg"); pyj.do("gW=: >1{gg")
gX = pyj.get("gX"); gW = pyj.get("gW")

def loss_fn(Xa, Wa):
    return np.tanh(Xa @ Wa).sum()

eps = 1e-6
numX = np.zeros_like(Xr)
for i in range(Xr.shape[0]):
    for j in range(Xr.shape[1]):
        P = Xr.copy(); M = Xr.copy(); P[i, j] += eps; M[i, j] -= eps
        numX[i, j] = (loss_fn(P, Wr) - loss_fn(M, Wr)) / (2*eps)
numW = np.zeros_like(Wr)
for i in range(Wr.shape[0]):
    for j in range(Wr.shape[1]):
        P = Wr.copy(); M = Wr.copy(); P[i, j] += eps; M[i, j] -= eps
        numW[i, j] = (loss_fn(Xr, P) - loss_fn(Xr, M)) / (2*eps)
check("gradcheck dL/dX (tanh layer)", gX, numX, tol=1e-5)
check("gradcheck dL/dW (tanh layer)", gW, numW, tol=1e-5)

# ---- test 8: 2-layer MLP gradcheck: L = sum( tanh( tanh(X mp W1) mp W2 ) ) ----
W2r = rng.normal(size=(2, 2))
pyj.set("W2", W2r)
pyj.do("ADSET 'X';'W';'W2'")
pyj.do("A7=: 0 nmp 1")          # X mp W1
pyj.do("B7=: ntanh A7")
pyj.do("C7=: B7 nmp 2")         # tanh(...) mp W2
pyj.do("L8=: nsum (ntanh C7)")
pyj.do("gg=: ADGET L8")
pyj.do("gX=: >0{gg"); pyj.do("gW=: >1{gg"); pyj.do("gW2=: >2{gg")
gX2 = pyj.get("gX"); gW2_ = pyj.get("gW"); gW22 = pyj.get("gW2")

def loss2(Xa, Wa, Wb):
    return np.tanh(np.tanh(Xa @ Wa) @ Wb).sum()

numX2 = np.zeros_like(Xr)
for i in range(2):
    for j in range(3):
        P = Xr.copy(); M = Xr.copy(); P[i, j] += eps; M[i, j] -= eps
        numX2[i, j] = (loss2(P, Wr, W2r) - loss2(M, Wr, W2r)) / (2*eps)
numW1 = np.zeros_like(Wr)
for i in range(3):
    for j in range(2):
        P = Wr.copy(); M = Wr.copy(); P[i, j] += eps; M[i, j] -= eps
        numW1[i, j] = (loss2(Xr, P, W2r) - loss2(Xr, M, W2r)) / (2*eps)
numW2 = np.zeros_like(W2r)
for i in range(2):
    for j in range(2):
        P = W2r.copy(); M = W2r.copy(); P[i, j] += eps; M[i, j] -= eps
        numW2[i, j] = (loss2(Xr, Wr, P) - loss2(Xr, Wr, M)) / (2*eps)
check("gradcheck dL/dX (2-layer)", gX2, numX2, tol=1e-5)
check("gradcheck dL/dW1 (2-layer)", gW2_, numW1, tol=1e-5)
check("gradcheck dL/dW2 (2-layer)", gW22, numW2, tol=1e-5)

# ---- test 9: training loop — logistic regression on blobs, gradient descent ----
# Leaves: Xt (data, constant), yt (labels, constant), wt (weights, trainable).
# Graph: z=Xt mp wt; L = sum( (sigmoid(z) - yt)^2 ),  sigmoid(z) = 1 % (1 + ^-z)
# Rebuild the graph each step (tape is cheap to rebuild) with fresh leaf values.
rng = np.random.default_rng(42)
N, D = 64, 2
Xa0 = np.vstack([rng.normal([0, 0], .8, (N//2, D)), rng.normal([2.5, 2.5], .8, (N//2, D))])
ya = np.concatenate([np.zeros(N//2), np.ones(N//2)])
Xa = np.hstack([Xa0, np.ones((N, 1))])   # append bias column
pyj.set("Xt", Xa)
pyj.set("yt", ya.reshape(-1, 1))
pyj.set("wt", np.array([[0.3], [0.3], [0.3]]))

BUILD = """
wtv=: wt
ADSET 'wtv';'Xt';'yt'
NB. leaves: node0=wtv, node1=Xt, node2=yt
z9=: 1 nmp 0
e9=: nexp (nneg z9)
one9=: nleaf 1
d9=: one9 nadd e9
p9=: one9 ndiv d9
r9=: p9 nsub 2
q9=: nsq r9
L9=: nsum q9
"""

def train_step(w, lr):
    pyj.set("wt", w)
    for s in BUILD.strip().split("\n"):
        pyj.do(s)
    pyj.do("gg=: ADGET L9")
    pyj.do("gw=: >0{gg")
    return w - lr * pyj.get("gw")

w = np.array([[0.3], [0.3], [0.3]])
for step in range(80):
    w = train_step(w, 0.5)
w_final = w
z = (Xa @ w_final).ravel()
acc = ((1/(1+np.exp(-z)) > .5).astype(float) == ya).mean()
check("training: logistic accuracy > 0.9", np.array([acc > 0.9]).astype(float), np.array([1.0]))
print(f"final weights: {w_final.ravel()}, train acc: {acc:.2f}")

print()
print("AD TESTS: ALL OK" if fails == 0 else f"AD TESTS: {fails} FAILURES")
sys.exit(1 if fails else 0)
