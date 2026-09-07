# Tally — caught-in-the-act cards

Each card: one question, the float64/default-tool answer, and the
exact Tally answer. Source: recon/shootout.py + recon/check2.py.

### 01. One third

| | answer |
|---|---|
| default tool (float64/bc) | `0` |
| **Tally (exact)** | `0.333333333333333333333333333333  (truncated, 30 digits)` |

> bc with default scale=0 returns **0** for 1/3. A model quoting bc verbatim propagates the zero downstream.

### 02. Square root of 2

| | answer |
|---|---|
| default tool (float64/bc) | `1` |
| **Tally (exact)** | `1.414213562373095048801688724209  (truncated, 30 digits)` |

> bc without -l has scale=0: sqrt(2) returns **1**. Tally returns a certified rational convergent.

### 03. 0.1 + 0.2

| | answer |
|---|---|
| default tool (float64/bc) | `0.30000000000000004` |
| **Tally (exact)** | `0.300000000000000000000000000000  (truncated)` |

> float64 cannot represent 1/10. Tally parses "0.1" as exactly 1/10 — before any float can touch it.

### 04. Ten tenths equal one?

| | answer |
|---|---|
| default tool (float64/bc) | `False 0.9999999999999999` |
| **Tally (exact)** | `1  (exactly 1)` |

> Float accumulation fails equality tests — a reconciliation job takes the else-branch. Tally: 10×1/10 = 1, exactly.

### 05. Compound interest, 360 months

| | answer |
|---|---|
| default tool (float64/bc) | `446774.4314006109` |
| **Tally (exact)** | `44677443140061322124…  (exact rational, truncated)` |

> float64 is off from significant digit #15: …06109 vs the exact …06132. ~1e-8 per account — invisible on one statement, material at ledger scale. Tally compounds the exact rational 241/240.

### 06. 1/det(Hilbert_4) — must be an integer

| | answer |
|---|---|
| default tool (float64/bc) | `6047999.999999422` |
| **Tally (exact)** | `6048000` |

> The reciprocal of the Hilbert-4 determinant is exactly 6,048,000. float64 says …999999.42.

### 09. 1 − cos(10⁻⁸)

| | answer |
|---|---|
| default tool (float64/bc) | `0.0` |
| **Tally (exact)** | `≈ 5.00000000000000042×10⁻¹⁷ (exact rational)` |

> float64 rounds cos(1e-8) to exactly 1.0, so the answer collapses to zero. Tally's rational Taylor series keeps the result non-zero — and exact.

### 10. Variance of [10⁹ … 10⁹+9]

| | answer |
|---|---|
| default tool (float64/bc) | `128.0` |
| **Tally (exact)** | `33r4  (= 8.25, exactly)` |

> The one-pass formula LLMs generate most often cancels catastrophically: 128.0 instead of 8.25. Tally's two-pass exact arithmetic gets 33/4.

### 11. The constant e

| | answer |
|---|---|
| default tool (float64/bc) | `2.718281828459045` |
| **Tally (exact)** | `2.718281828459045235360287471352662497757…  (40 certified digits)` |

> float64 stops at 16 digits. Tally's rational series is certified to the requested precision.

### 12. C(100, 50) — terminal wins this one

| | answer |
|---|---|
| default tool (float64/bc) | `100891344545564193334812497256` |
| **Tally (exact)** | `100891344545564193334812497256` |

> Python's bignum is exact for pure integers — full credit. Tally matches it; the differentiation is everywhere else.

### 13. 2.3 × 100 (a money multiply)

| | answer |
|---|---|
| default tool (float64/bc) | `229.99999999999997` |
| **Tally (exact)** | `230` |

> float64 says 229.99999999999997; the ledger says 230. Tally parses "2.3" as exactly 23/10.

### 13b. float(2^53 + 1)

| | answer |
|---|---|
| default tool (float64/bc) | `9007199254740992.0` |
| **Tally (exact)** | `9007199254740993` |

> Integers past 2^53 are not representable in float64 — the +1 silently vanishes. ID / counter pipelines lose it without any error raised.

### 14. numpy in a slim container

| | answer |
|---|---|
| default tool (float64/bc) | `ModuleNotFoundError: No module named 'numpy'` |
| **Tally (exact)** | `exact rational matmul, zero deps (sum of squares = 68)` |

> Strip site-packages (slim container / restricted sandbox) and the float toolchain is simply unavailable. Tally's engine has zero dependencies.
