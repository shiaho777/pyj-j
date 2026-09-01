"""test_compile.py — tape compiler (adcomp.ijs / ADGEN) validation + benchmark.

Roadmap step 4a: compile the AD tape into a single fused J verb (adcB<n>)
that recomputes the forward pass and runs the backward VJP walk as
straight-line code, then verify:

  1. compiled grads == interpreted ADGET grads on the same tape (softmax
     classifier forward: mp/tanh/rmax1/rsub1/exp/rsum1/log chain)
  2. compiled grads == numeric gradients (central differences)
  3. a real training loop driven by the compiled verb converges
  4. benchmark: compiled step vs interpreted (tape rebuild + ADGET) step

Usage:
  DYLD_LIBRARY_PATH=../jlibrary/bin PYJ_LIBPATH=../jlibrary/bin python3 test_compile.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyj

HERE = os.path.dirname(os.path.abspath(__file__))
pyj.do(f"0!:0 <'{HERE}/ad.ijs'")
pyj.do(f"0!:0 <'{HERE}/adcomp.ijs'")

fails = 0
def check(label, got, want, tol=1e-9):
    global fails
    ok = np.allclose(got, want, atol=tol, rtol=1e-7)
    if not ok:
        fails += 1
        print(f"FAIL {label}: got {got!r} want {want!r}")
    else:
        print(f"ok   {label}")

rng = np.random.default_rng(7)
N = 96
def _blob(center, r, n):
    ang = rng.uniform(0, 2*np.pi, n)
    rad = r * np.sqrt(rng.uniform(0.15, 1, n))
    return np.stack([center[0] + rad*np.cos(ang), center[1] + rad*np.sin(ang)], 1)
Xa = np.vstack([_blob([0, 0], 2.0, N//3), _blob([5, 0], 2.0, N//3), _blob([2.5, 4.3], 2.0, N//3)])
Xdata = np.hstack([Xa, np.ones((N, 1))])
Ydata = np.eye(3)[np.repeat(np.arange(3), N // 3)]
W1_0 = rng.normal(size=(3, 16)) * 0.25
W2_0 = rng.normal(size=(16, 3)) * 0.25

# ---------- build the tape once (softmax classifier forward) ----------
pyj.set("X_v", Xdata)
pyj.set("Y_v", Ydata)
pyj.set("W1_v", W1_0)
pyj.set("W2_v", W2_0)
pyj.do("ADSETP (<'X_v';X_v) , (<'Y_v';Y_v) , (<'W1_v';W1_v) , (<'W2_v';W2_v)")
pyj.do("cH=: 0 nmp 2")            # X mp W1        (ids: X=0 Y=1 W1=2 W2=3)
pyj.do("cA=: ntanh cH")
pyj.do("cZ=: cA nmp 3")           # tanh(H) mp W2
pyj.do("cM=: nrmax1 cZ")
pyj.do("cS=: cZ nrsub1 cM")
pyj.do("cE=: nexp cS")
pyj.do("cSum=: nrsum1 cE")
pyj.do("cLge=: nlog cSum")
pyj.do("cLse=: cM nadd cLge")     # log-sum-exp = rowmax + log(sum(exp(Z-rowmax)))
pyj.do("cSh=: cZ nrsub1 cLse")
pyj.do("cG=: cSh nmul 1")         # shifted * Y
pyj.do("cLG=: nsum cG")           # = sum(Y*logP)  (leading-axis; (3,))
pyj.do("cLG2=: nsum cLG")         # full scalar reduction
LOSS = int(pyj.get("cLG2"))

def interp_grads():
    """interpreted path: ADGET on the existing tape"""
    pyj.do(f"gg=: ADGET {LOSS}")
    g = []
    for i in range(4):
        pyj.do(f"ggx=: >{i}{{gg")
        g.append(pyj.get("ggx"))
    return g

gi = interp_grads()

# ---------- compile: adcB1 (fused fwd+bwd), adcF1 (fwd only) ----------
pyj.do(f"{LOSS} ADGEN 1")

# ---------- 1: compiled == interpreted ----------
pyj.do("adcB1 ''")
gW1c = pyj.get("adcg2")
gW2c = pyj.get("adcg3")
check("compiled gX == interp", pyj.get("adcg0"), gi[0])
check("compiled gY == interp", pyj.get("adcg1"), gi[1])
check("compiled gW1 == interp", gW1c, gi[2])
check("compiled gW2 == interp", gW2c, gi[3])

# ---------- 2: compiled == numeric ----------
def fwd_loss(w1, w2):
    H = np.tanh(Xdata @ w1)
    lg = H @ w2
    lg = lg - lg.max(1, keepdims=True)
    E = np.exp(lg)
    P = E / E.sum(1, keepdims=True)
    return -(np.log((P * Ydata).sum(1) + 1e-12)).sum()

eps = 1e-6
numW1 = np.zeros_like(W1_0)
for i in range(3):
    for j in range(16):
        P = W1_0.copy(); M = W1_0.copy(); P[i, j] += eps; M[i, j] -= eps
        numW1[i, j] = (fwd_loss(P, W2_0) - fwd_loss(M, W2_0)) / (2 * eps)
check("compiled dCE/dW1 == numeric", -gW1c, numW1, tol=1e-5)

# ---------- 3: training loop on the compiled verb ----------
# sum-loss gradients (scale ~N per step); lr must be small. This init also
# needs a few hundred steps for the tanh layer to break symmetry.
lr = 0.02
w1, w2 = W1_0.copy(), W2_0.copy()
ce0 = None
for step in range(801):
    pyj.set("W1_v", w1)
    pyj.set("W2_v", w2)
    pyj.do("adcB1 ''")
    g1 = pyj.get("adcg2")
    g2 = pyj.get("adcg3")
    w1 = w1 + lr * g1
    w2 = w2 + lr * g2
    if step % 200 == 0:
        ce = fwd_loss(w1, w2) / N
        if ce0 is None:
            ce0 = ce
        print(f"  step {step:3d}  mean CE={ce:.4f}")
H = np.tanh(Xdata @ w1)
pred = (H @ w2).argmax(1)
acc = (pred == np.repeat(np.arange(3), N // 3)).mean()
print(f"final train acc: {acc:.3f}")
# ratio asserts are fragile when CE reaches ~1e-4; just require a solid drop
check("compiled-path training: CE dropped >5x", np.array([ce0 / (fwd_loss(w1, w2) / N) > 5.0]), np.array([1.0]))
check("compiled-path train acc > 0.95", np.array([acc > 0.95]), np.array([1.0]))

# ---------- 4: benchmark ----------
n_iters = 200
t0 = time.perf_counter()
for _ in range(n_iters):
    gi_ = interp_grads()
t_interp = (time.perf_counter() - t0) / n_iters

t0 = time.perf_counter()
for _ in range(n_iters):
    pyj.do("adcB1 ''")
    g1 = pyj.get("adcg2")
    g2 = pyj.get("adcg3")
t_comp = (time.perf_counter() - t0) / n_iters

# full interpreted training step = rebuild tape (per-node JDo) + ADGET
def rebuild_tape():
    pyj.set("W1_v", w1)
    pyj.set("W2_v", w2)
    for s in ("cH=: 0 nmp 2", "cA=: ntanh cH", "cZ=: cA nmp 3", "cM=: nrmax1 cZ",
              "cS=: cZ nrsub1 cM", "cE=: nexp cS", "cSum=: nrsum1 cE",
              "cLge=: nlog cSum", "cLse=: cM nadd cLge", "cSh=: cZ nrsub1 cLse",
              "cG=: cSh nmul 1", "cLG=: nsum cG", "cLG2=: nsum cLG",
              f"gg=: ADGET {LOSS}"):
        pyj.do(s)

t0 = time.perf_counter()
for _ in range(n_iters):
    rebuild_tape()
t_interp_full = (time.perf_counter() - t0) / n_iters

print()
print(f"benchmark (per step, softmax fwd+bwd, N={N}):")
print(f"  interpreted: tape rebuild + ADGET   {t_interp_full*1e3:8.2f} ms")
print(f"  interpreted: ADGET only             {t_interp*1e3:8.2f} ms")
print(f"  compiled:    fused adcB1 call       {t_comp*1e3:8.2f} ms   ({t_interp_full/t_comp:.1f}x vs full, {t_interp/t_comp:.1f}x vs ADGET)")

print()
print("COMPILE TESTS:", "ALL OK" if fails == 0 else f"{fails} FAILURES")
sys.exit(0 if fails == 0 else 1)
