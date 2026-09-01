# pyj

Run the J language engine inside the Python process. numpy arrays cross into J
without conversion, and the bridge layers a reverse-mode autodiff engine —
written entirely in J — on top: tape recording, VJP rules, and a compiler that
fuses a whole model into a single J verb.

I built this to explore a question: array languages were designed for exactly
the math that machine learning wants (implicit broadcasting, rank-polymorphic
reduction, verbs that compose), so what gets unlocked if J sits *inside* the
Python ecosystem instead of asking people to leave it?

```python
import pyj, numpy as np

pyj.set("a", np.array([1.0, 2.0, 3.0, 4.0]))
pyj.do("s=: +/ a")            # J sentence, executed by the engine
pyj.get("s")                  # -> 10.0, numpy float64

pyj.set("m", np.arange(24.0).reshape(2, 3, 4))
pyj.do('t=: +/"1 m')          # rank-1 sum: vmap for free
pyj.get("t")                  # shape (2, 3)

pyj.do("bpv=: !32x")          # 32! exactly, via GMP — past int64
```

## Autodiff

`ad.ijs` is ~250 lines of J implementing reverse-mode AD over a closed set of
24 primitives (arithmetic, matmul, tanh/exp/log, sum/max, rank-1 ops,
reshape/transpose/take/drop/gather). `adt.py` wraps it in a tensor API:

```python
from adt import Graph
g = Graph()
X = g.tensor(Xa, "X")
W = g.tensor(Wa, "W")
L = (((X @ W).tanh() - Y) ** 2).sum()
g.back(L)
W.grad                        # numpy array — computed by J
```

Every forward pass and every gradient in the loop below executes in the J
engine, not numpy:

```python
for step in range(600):
    W1.value = w1; W2.value = w2
    g._sync_leaves()
    L = build_loss(g)         # e.g. log-sum-exp cross-entropy
    g.back(L)
    w1 = w1 + lr * W1.grad
```

The test suites include central-difference gradchecks (matmul, MLP with bias,
softmax, cross-entropy, embedding lookup with duplicate indices) and training
runs — logistic regression, a 2-layer MLP, a 3-class softmax classifier, an
embedding model — each converging to accuracy 1.00.

## The tape compiler

Interpreting the tape means a boxed walk per backward pass: `select.`
dispatch and row lookups for every node. `adcomp.ijs` instead reads the tape
and emits the source of one explicit J verb — straight-line forward code, then
straight-line VJP code, the way you would have written it by hand. Constants
bake in at compile time; dead subgraphs get pruned.

Measured on a 96-sample softmax forward+backward:

| | per step |
|---|---|
| rebuild tape + ADGET | 4.05 ms |
| compiled verb | 0.02 ms |

That's ~180x, and it's the path I'd take further: the generated verb is
ordinary J, so it can eventually be replayed, cached, or shipped to a real
backend.

## The bridge

Data crosses by pointer, not by format. Writing is `JSetM` (`io.c` `setterm`):
one `memcpy` from the numpy buffer into JE memory. Reading is `JGetM`: J
hands back raw shape/data pointers into JE memory, and one `memcpy` fills a
fresh numpy array. Types map 1:1 at 64 bits — bool/int64/float64/complex128
to B01/INT/FL/CMPX. On a 10M-element sum, 8 ms goes to crossing the bridge
(~10 GB/s) and 1 ms to the engine.

## Building

You need a J engine runtime (libj + profile.ijs), GMP, and Python with numpy.

```sh
git clone https://github.com/jsoftware/jsource   # or copy bin/ from a J release
cd jsource/make2 && ./build_libj.sh && ./build_jconsole.sh && ./cpbin.sh
cp ../jlibrary/bin/* <this repo>/jlibrary/bin/   # libj, profile.ijs

brew install gmp    # macOS; JE dlopens libgmp at startup
./build.sh
```

Then:

```sh
DYLD_LIBRARY_PATH=$PWD/jlibrary/bin PYJ_LIBPATH=$PWD/jlibrary/bin python3 test_pyj.py
```

`test_pyj.py` (bridge), `test_ad.py` (AD core), `test_nn.py` (tensor API),
`test_ops2.py` (rank-1/softmax), `test_ops3.py` (take/drop/gather + compiler
parity + embedding training), `test_train2.py` (softmax classifier),
`test_compile.py` (compiler validation + benchmark).

Tested on macOS arm64. Linux should work with the `.so` suffix (the build
script handles it) but hasn't been verified on a clean machine yet.

## Notes from the trenches

Things I learned building this on J 9.8 beta, kept here because they'll bite
again:

- Top-level control words are rejected in embedded mode — everything lives in
  explicit `3 : 0` verbs. The stdlib isn't loaded either, so `empty` and `LF`
  don't exist.
- Boolean literals auto-type to B01. Node id lists must be forced to INT with
  `(2#0)+x,y` or `{` indexing breaks.
- `+/` reduces the **leading** axis — that's numpy's `sum(axis=0)`, not the
  last axis.
- After `JInit2`/`JSM`, one warmup `JSetM` is required before name lookup
  works; without it, later `JDo`/`JGetM` fail silently.
- `}` amend *replaces*, never accumulates. Embedding gradients (duplicate
  indices) need an `(i.n)="(0 1) idx` boolean-table matmul instead.
- `(N,1) * (1,1)` is a length error: J extends rank-0 scalars, but doesn't
  extend length-1 frames the way numpy broadcasts.

## Roadmap

- [x] Zero-copy numpy bridge
- [x] Closed-set reverse-mode AD, gradchecked
- [x] Tensor API, end-to-end training runs
- [x] Tape-to-verb compiler
- [x] take/drop/gather (embedding lookup)
- [ ] General rank dispatch (`"`), comparison and choose
- [ ] Tape replay caching; MLIR/StableHLO as an execution backend

## License

GPL-3.0 — see [LICENSE](LICENSE). The embedded J engine is Jsoftware's
jsource, used at runtime under the same license; nothing from it is
redistributed here. Not affiliated with Jsoftware.
