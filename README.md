# pyj — J as a tensor engine inside Python

`pyj` embeds the [J language](https://www.jsoftware.com) engine (`libj`) directly
in the CPython process, so numpy arrays cross into J with zero conversion and
J's primitives — `+/ .*`, `"` (rank), `d.` (derivatives), `x:` (extended
precision) — become callable operators from Python.

On top of the bridge it layers a **reverse-mode autodiff engine written in J**:
a tape of 24 closed-set primitives, VJP rules, a tape-to-J-verb compiler that
fuses a whole model into one straight-line J function (~180x faster than the
interpreted tape), and a numpy-style Tensor API. All gradients and all forward
math execute inside the J engine.

This is a pilot for "AI-native array language": **parasitize the Python
ecosystem instead of replacing it.**

## Status

- macOS arm64 (Apple Silicon): built and fully tested. Linux: expected to work
  (build.sh handles `.so`), not yet CI-verified.
- 7 test suites, 45+ checks, all green: bridge types, unit VJPs, central-
  difference gradchecks (MLP / softmax / cross-entropy / embedding),
  end-to-end training runs (logistic, 2-layer MLP, 3-class softmax
  classifier, embedding lookup — all reach accuracy 1.00).
- Benchmarks (`bench_pyj.py`, `test_compile.py`):
  - bridge overhead ~0.5 µs/op (1k-element `pyj.set`);
  - 10M-element sum: 8 ms crossing the bridge (~10 GB/s) + 1 ms in J;
  - compiled AD verb: 0.02 ms/step vs 4.05 ms/step interpreted (96-sample
    softmax forward+backward) — **~180x**.

## Files

| file | what it is |
|---|---|
| `pyj.c` | CPython extension: JInit2/JSM/JDo + numpy <-> JSetM/JGetM zero-serialization bridge |
| `ad.ijs` | reverse-mode AD in pure J: tape + closed-set VJP table (24 primitives) |
| `adcomp.ijs` | tape -> J verb compiler: emits a single fused forward+backward function |
| `adt.py` | Tensor factory API (operator overloading; the recommended entry point) |
| `test_*.py` | 7 suites: bridge, AD, Tensor API, rank-1 ops, compiler, end-to-end training |
| `bench_pyj.py` | bridge overhead benchmark |

## Build

```sh
# 1) get a J engine runtime into ./jlibrary/bin (libj + profile.ijs):
#    - build from source:  git clone https://github.com/jsoftware/jsource
#      cd jsource/make2 && ./build_libj.sh && ./build_jconsole.sh && ./cpbin.sh
#    - or copy the bin/ folder of a released J distribution
# 2) GMP (the engine dlopens it):            brew install gmp
# 3) python3 with numpy, then:
./build.sh
```

## Usage

```python
import pyj, numpy as np

pyj.set("a", np.array([1.0, 2.0, 3.0, 4.0]))
pyj.do("s=: +/ a")           # J sentences execute directly
pyj.get("s")                 # -> 10.0 (numpy float64)

# J's rank mechanism is free vmap:
pyj.set("m", np.arange(24.0).reshape(2, 3, 4))
pyj.do("t=: +/\"1 m")        # sum along last axis
pyj.get("t")                 # shape (2, 3)

# extended precision beyond int64 via GMP:
pyj.do("bpv=: !32x")         # 32! as exact big integer
```

### Tensor API (autodiff)

```python
from adt import Graph
g = Graph()
X = g.tensor(Xa, "X")        # leaves: data & parameters
W = g.tensor(Wa, "W")
H = (X @ W).tanh()           # build with operators
L = ((H - Y) * (H - Y)).sum()
g.back(L)                    # reverse mode through the J tape
W.grad                       # numpy gradient

# or compile the tape into one fused J verb (train-time path):
#   loss_id ADGEN 1   ->  adcB1 ''  ->  adcgL, adcg0.. (grads)
```

Array manipulation ops with full VJPs: `.take(n) .drop(n) .reshape(...) .gather(idx)`
(embedding lookup via gather; duplicate indices accumulate correctly).

## Data path (zero-serialization)

- **write**: C-contiguous numpy array -> `JSetM` (`io.c:1081` -> `setterm`):
  one `memcpy` into JE memory, no format encoding.
- **read**: `JGetM` (`io.c:1041`): returns raw shape/data pointers into JE
  memory; one `memcpy` into a fresh numpy array.
- dtypes: bool / int64 / float64 / complex128 map to J B01/INT/FL/CMPX
  (all 64-bit, zero loss). BOX/XNUM/RAT/SPARSE do not cross; `":` them first.

## Design notes & J9.8-beta pitfalls (learned the hard way)

See the "实现注记" sections of `README` history / code comments. Highlights:

- top-level control words are banned in embedded mode — everything lives in
  explicit `3 : 0` verbs; stdlib (`empty`, `LF`) is absent — define your own;
- boolean literals auto-type to B01 — force INT ids with `(2#0)+x,y`;
- `+/` reduces the **leading** axis (numpy `sum(axis=0)`), rank semantics
  differences are pinned by tests;
- JSetM takes double-indirect shape/data; a warmup JSetM is required after
  JInit2/JSM or later JDo/JGetM silently fail;
- `}` amend **replaces**, never accumulates — scatter-accumulate (embedding
  gradients) uses an `(i.n)="(0 1) idx` boolean table matmul instead.

## Roadmap

1. ~~zero-copy bridge~~ ✅
2. ~~closed-set reverse-mode AD~~ ✅ (24 primitives, gradchecked)
3. ~~Tensor API + end-to-end training~~ ✅ (softmax classifier acc=1.00)
4. ~~tape -> fused J verb compiler~~ ✅ (~180x vs interpreted tape)
5. AD coverage: ~~take/drop/gather~~ ✅; remaining: general rank dispatch `"`,
   compare/choose. Next: MLIR/StableHLO device backend, tape replay caching.

## License

This project links against and embeds the J engine, which is GPL-3.0 licensed
(Jsoftware). See `license.txt` (upstream jsource) — this repository's code is
distributed under GPL-3.0 as well. J and jsource are (c) Jsoftware; this is an
independent pilot project, not affiliated with Jsoftware.
