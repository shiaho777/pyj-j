# Embedding the pyj kernel — ABI contract & 10-minute guide

The kernel is the unmodified J engine (`libj`, ~5 MB) plus a thin bridge.
This document is the stable contract downstream apps build against.

## The ABI contract (frozen)

### Raw C ABI (5 calls)

Any host language can drive the engine through these five symbols
(declared in `jsrc/jlib.h`, exported by `libj.dylib` / `libj.so`):

| call | signature | semantics |
|---|---|---|
| `JInit2` | `JS JInit2(char* libpath)` | load the engine; `libpath` is the dir holding `libj` + `profile.ijs` |
| `JSM` | `void JSM(JS, void** callbacks)` | install output/session callbacks |
| `JDo` | `int JDo(JS, char* sentence)` | execute one J sentence; 0 = success |
| `JSetM` | `int JSetM(JS, char* name, ll* type, ll* rank, ll* shape, ll* data)` | store a raw array as J noun `name` (one memcpy in) |
| `JGetM` | `int JGetM(JS, char* name, ll* type, ll* rank, ll* shape, ll* data)` | fetch J noun `name` as raw pointers (one memcpy out) |
| `JFree` | `int JFree(JS)` | release the instance |

(`ll` = `long long`.)

**Invariants that must hold for any conforming host:**

1. **One instance per process, single-threaded.** `JDo` carries interpreter
   state; concurrent calls are undefined. Serialize at the host layer.
2. **Zero serialization on the hot path.** Data crosses as raw ravel bytes
   via `JSetM`/`JGetM` — no text protocols, no `3!:1` encode.
3. **Warmup required.** After `JInit2`/`JSM`, one `JSetM` must run before
   name lookup works; otherwise later `JDo`/`JGetM` fail silently.
4. **Types map 1:1 at 64 bits.** bool/int64/float64/complex128 ↔ B01/INT/FL/CMPX.

### Python module surface (`pyj`)

The CPython extension wraps the same ABI:

```python
pyj.do(sentence)   # -> (rc: int, [output lines]); rc 0 = success
pyj.set(name, arr) # store a numpy array as J noun `name`
pyj.get(name)      # fetch J noun `name` as a numpy array
pyj.free()         # release the J instance
```

Supported dtypes: `bool`, `int64`, `float64`, `complex128` (zero-loss),
plus char arrays surfaced as `uint8` on read.

## 10-minute embedding (Python host)

```sh
# 1. build the engine once (needs a C compiler + GMP)
git clone --depth 1 https://github.com/jsoftware/jsource.git
cd jsource/make2 && ./build_libj.sh && ./build_jconsole.sh && ./cpbin.sh
mkdir -p /path/to/pyj/jlibrary/bin
cp ../jlibrary/bin/* /path/to/pyj/jlibrary/bin/

# 2. build the bridge
cd /path/to/pyj && ./build.sh

# 3. run
DYLD_LIBRARY_PATH=$PWD/jlibrary/bin PYJ_LIBPATH=$PWD/jlibrary/bin \
  python3 -c "import pyj, numpy as np; pyj.set('a', np.arange(4.0)); \
              pyj.do('s=: +/ a'); print(pyj.get('s'))"   # -> 6.0
```

## Zero-build embedding (vendored C host)

`vendor/` is what a downstream app receives — no jsource, no Python, no
engine build step:

```sh
cd vendor && make && DYLD_LIBRARY_PATH=. ./vendor_demo   # macOS
cd vendor && make && LD_LIBRARY_PATH=.    ./vendor_demo   # Linux
```

`vendor/app.c` (~100 lines) trains a classifier in-process through the raw
ABI and serves predictions. Every numeric step runs inside `libj`.

## Exact arithmetic as a service (Tally MCP)

For agents that need exact numbers rather than a tensor kernel, run the
MCP server — no J knowledge required:

```sh
DYLD_LIBRARY_PATH=jlibrary/bin PYJ_LIBPATH=jlibrary/bin python3 tally_server.py
```

See `TALLY.md` for the three tools and the AST.

## Versioning

- The raw 5-call ABI and the `pyj` module surface are **frozen**; breaking
  either is a regression per `VISION.md`.
- The Tally AST is versioned (`v1`); new ops are additive. Limits
  (node count, array size, digits) are guardrails, documented in `TALLY.md`.
- The `solver` public API (`solver/__init__.py`, `__version__`) is stable;
  network/scan tooling is explicitly *not* part of it.