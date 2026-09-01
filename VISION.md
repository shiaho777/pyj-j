# Vision: J as an embeddable tensor kernel

This document fixes the project's direction. It exists so that future
decisions can be checked against a written standard instead of re-litigating
the goal in every discussion.

## The bet

We are not trying to make the J language popular. We are treating the J
engine (`libj`) as an **embeddable tensor kernel** — the same role SQLite
plays for databases and Lua plays for game scripting:

- a few megabytes, no dependencies (beyond GMP), loads in-process,
  starts instantly, ships inside someone else's application;
- a full array language as the *internal* compute model, with reverse-mode
  autodiff and a tape compiler living inside it;
- host languages (today Python) talk to it over a thin bridge.

The claim underneath: a closed-set array language is not just a language,
it is a ready-made IR with an interpreter attached. That makes it uniquely
suited to be the compute core of AI tooling — provided it is never asked to
be popular.

## Why this niche might exist

Nobody embeds a tensor runtime. MLIR/LLVM is hundreds of megabytes with
churning APIs; BLAS stacks are fragmented library collections; training
runtimes are welded to Python. Meanwhile:

1. **libj is a proven array VM.** 30+ years of engineering, rank/frame
   semantics in the interpreter core (not a library abstraction), JIT-ish
   special-code paths. Our tape compiler reaching ~180x by emitting
   straight-line J is evidence: *an array language kernel is already an
   optimizer for array programs.*
2. **Embeddability is the product.** Kernels do not need ecosystems; they
   need small size, stability, and a crisp C ABI. SQLite has ~2% name
   recognition and is in every phone.
3. **Exact arithmetic is genuinely scarce** in the numeric-AI era (GMP
   backends inside array ops) — reference computations, verifying
   model-generated numeric code, hybrid symbolic/numeric work.

## What this project is (and is not)

**Is:**

- `libj` + in-process bridge (`pyj.c`) + autodiff/AD-tape written in J
  (`ad.ijs`) + tape→verb compiler (`adcomp.ijs`) + Python tensor API
  (`adt.py`) + MLIR exporter (`adexport.py`).
- A layer cake where every layer above the kernel is replaceable.

**Is not:**

- A fork or rewrite of J — the engine is used unmodified, so upstream
  updates keep flowing.
- A J-language advocacy project. We do not advertise the language; we
  productize the kernel.
- A general-purpose J-on-Python compatibility layer. Host-facing APIs are
  designed for tensor workloads, not for running J scripts.

## Architecture invariants

Breaking any of these is a regression, regardless of features gained:

1. **The kernel stays unmodified.** No patches to jsource. Anything that
   requires editing J internals is rejected by design.
2. **Zero-serialization bridge.** numpy ↔ J stays at one `memcpy` per
   direction (JSetM/JGetM). No text protocols, no 3!:1 encode on the hot path.
3. **J-side logic stays in J.** AD rules, the tape compiler, and the VJP
   table live in `.ijs` files. They must keep working from bare `jconsole`
   with no Python involved — this is the portability proof and it doubles
   as our debugging channel.
4. **Every numerics claim is gradchecked or parity-tested.** New ops enter
   the closed set only with a numeric-gradient test or an interpreted-vs-
   compiled (or MLIR-vs-J) parity test.
5. **Test suites run from a clean checkout on CI.** No state, no manual
   dylib surgery, no `install_name_tool` outside `build.sh`.

## Success criteria (in order)

1. **Correctness** — all suites green on CI, gradients verified against
   numeric differentiation on every primitive.
2. **Embedded-ness** — a downstream application can vendor `libj + pyj`
   (≈5.5 MB) and ship. Measured by: build works from a script on a clean
   machine, Linux + macOS, no root, no network after bootstrap.
3. **A real workload** — one non-demo application runs its numerics through
   the kernel. This is the only test of the niche. Candidates: exact-integer
   verification of AI-generated numeric code; an MLIR reference backend;
   an embedded training loop in a non-Python host (re-uses ad.ijs verbatim).
4. **Performance headroom** — the compiler path stays ≥10x over the
   interpreted tape on real workloads.

## Non-goals

- Making J idiomatic for humans; documenting the language; community
  building around J syntax.
- Beating numpy/torch on their own turf (huge models, GPU farms).
- StableHLO/IREE for its own sake — only when a downstream consumer needs
  that exact format.

## Current gap to the bet

- Linux is untested (CI is macOS-only) — invariant 5 is violated today.
- Exporter coverage stops before take/drop/gather/reshape/transpose.
- No real workload yet (success criterion 3) — everything so far is
  self-validated demos.
