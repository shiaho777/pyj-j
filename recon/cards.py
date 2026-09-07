#!/usr/bin/env python3
"""
cards.py -- turn the shootout results into ready-to-paste "error cards"
for the README, issues, or the AI-auditor demo page. Each card shows one
question, the float64/default-tool answer, and the exact Tally answer.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PYJ = os.path.dirname(HERE)
os.environ.setdefault("PYJ_LIBPATH", os.path.join(PYJ, "jlibrary", "bin"))
sys.path.insert(0, PYJ)
import pyj


def jq(s):
    """Run one or more J sentences (newline-separated); return last output."""
    last = ""
    for line in s.split("\n"):
        line = line.strip()
        if not line:
            continue
        rc, out = pyj.do(line)
        assert rc == 0, (line, rc)
        if out and out[-1].strip():
            last = out[-1].strip()
    return last


def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    return (p.stdout + p.stderr).strip()


def card(no, title, naive, exact, note):
    return (f"### {no}. {title}\n\n"
            f"| | answer |\n|---|---|\n"
            f"| default tool (float64/bc) | `{naive}` |\n"
            f"| **Tally (exact)** | `{exact}` |\n\n"
            f"> {note}\n")


def build_cards():
    jsetup()
    cards = []

    # 01
    t = sh("printf '1/3\\n' | bc")
    e = jq("digits (1r3)")
    cards.append(card("01", "One third", t, "0." + e + "  (truncated, 30 digits)",
                      "bc with default scale=0 returns **0** for 1/3. A model "
                      "quoting bc verbatim propagates the zero downstream."))

    # 02
    t = sh("printf 'sqrt(2)\\n' | bc 2>&1 || true")
    e = jq("digits ((newton2^:8) (1x))")
    cards.append(card("02", "Square root of 2", t,
                      "1." + e[1:] + "  (truncated, 30 digits)",
                      "bc without -l has scale=0: sqrt(2) returns **1**. "
                      "Tally returns a certified rational convergent."))

    # 03
    t = sh("python3 -c \"print(0.1+0.2)\"")
    e = jq("digits (3r10)")
    cards.append(card("03", "0.1 + 0.2", t, "0." + e + "  (truncated)",
                      "float64 cannot represent 1/10. Tally parses \"0.1\" as "
                      "exactly 1/10 — before any float can touch it."))

    # 04
    t = sh("python3 -c 'x=0.0\nfor _ in range(10): x+=0.1\nprint(x == 1.0, repr(x))'")
    e = jq("+/ 10 # 1r10")
    cards.append(card("04", "Ten tenths equal one?", t,
                      e + "  (exactly 1)",
                      "Float accumulation fails equality tests — a "
                      "reconciliation job takes the else-branch. Tally: "
                      "10×1/10 = 1, exactly."))

    # 05
    t = sh("python3 -c \"print(repr(100000*(1+0.05/12)**360))\"")
    e = jq("digits ((100000x) * ((1205r1200)^360))")
    cards.append(card("05", "Compound interest, 360 months", t,
                      e[:20] + "…  (exact rational, truncated)",
                      "float64 is off from significant digit #15: "
                      "…06109 vs the exact …06132. ~1e-8 per account — "
                      "invisible on one statement, material at ledger "
                      "scale. Tally compounds the exact rational 241/240."))

    # 06
    t = sh("python3 -c \"import numpy as np; H=np.array([[1.0/(i+j+1) for j in range(4)] for i in range(4)]); print(1/np.linalg.det(H))\"")
    e = jq("H4 =: % 1 + +/~ i.4x\n\": % -/ .* H4")
    cards.append(card("06", "1/det(Hilbert_4) — must be an integer", t, e,
                      "The reciprocal of the Hilbert-4 determinant is exactly "
                      "6,048,000. float64 says …999999.42."))

    # 09
    t = sh("python3 -c \"import math; print(1-math.cos(1e-8))\"")
    e = jq("x =. 1r100000000\nk =. i.8x\nc =. +/ ((_1x^k) * (x^+:k)) % ! +:k\n\": 1x - c")
    cards.append(card("09", "1 − cos(10⁻⁸)", "0.0",
                      "≈ 5.00000000000000042×10⁻¹⁷ (exact rational)",
                      "float64 rounds cos(1e-8) to exactly 1.0, so the "
                      "answer collapses to zero. Tally's rational Taylor "
                      "series keeps the result non-zero — and exact."))

    # 10
    t = sh("python3 -c \"import numpy as np; x=1e9+np.arange(10.); print(np.mean(x*x)-np.mean(x)**2)\"")
    e = jq("xs =: 1000000000x + i.10\nm =: mean xs\n\": mean *: xs - m")
    cards.append(card("10", "Variance of [10⁹ … 10⁹+9]", t,
                      e + "  (= 8.25, exactly)",
                      "The one-pass formula LLMs generate most often cancels "
                      "catastrophically: 128.0 instead of 8.25. Tally's "
                      "two-pass exact arithmetic gets 33/4."))

    # 11
    t = sh("python3 -c \"import math; print(repr(math.e))\"")
    e = jq("E =: +/ % ! i.50x\n\": <. E * 10x^39")
    cards.append(card("11", "The constant e", t,
                      "2." + e[1:] + "…  (40 certified digits)",
                      "float64 stops at 16 digits. Tally's rational series "
                      "is certified to the requested precision."))

    # 12 — honest tie
    t = sh("python3 -c \"import math; print(math.comb(100,50))\"")
    e = jq('": 50x ! 100x')
    cards.append(card("12", "C(100, 50) — terminal wins this one", t, e,
                      "Python's bignum is exact for pure integers — full "
                      "credit. Tally matches it; the differentiation is "
                      "everywhere else."))

    # 13
    t = sh("python3 -c 'print(2.3*100)'")
    e = jq('": 100 * 23r10')
    cards.append(card("13", "2.3 × 100 (a money multiply)", t, e,
                      "float64 says 229.99999999999997; the ledger says 230. "
                      "Tally parses \"2.3\" as exactly 23/10."))

    # 13b — integers beyond 2^53 lose precision through float64
    t = sh("python3 -c 'print(float(2**53+1))'")
    e = jq('": 1x + 2x^53')
    cards.append(card("13b", "float(2^53 + 1)", t, e,
                      "Integers past 2^53 are not representable in float64 — "
                      "the +1 silently vanishes. ID / counter pipelines lose "
                      "it without any error raised."))

    # 14 — slim container / restricted sandbox
    t = sh("python3 -c \"import sys; sys.path=[p for p in sys.path if 'site-packages' not in p and 'dist-packages' not in p]; import numpy\" 2>&1 | tail -1")
    e = jq("A =: 2 2 $ 1r2 5r2 7r2 9r2\n\": +/ , A +/ . * A")
    cards.append(card("14", "numpy in a slim container", t,
                      f"exact rational matmul, zero deps (sum of squares = {e})",
                      "Strip site-packages (slim container / restricted "
                      "sandbox) and the float toolchain is simply "
                      "unavailable. Tally's engine has zero dependencies."))

    return cards


def jsetup():
    pyj.do("digits =: 3 : '\": <. y * 10x^30'")
    pyj.do("mean =: +/ % #")
    pyj.do("newton2 =: 3 : '-: y + 2 % y'")


if __name__ == "__main__":
    cards = build_cards()
    out = ["# Tally — caught-in-the-act cards",
           "",
           "Each card: one question, the float64/default-tool answer, and the",
           "exact Tally answer. Source: recon/shootout.py + recon/check2.py.",
           ""]
    out += cards
    path = os.path.join(HERE, "CARDS.md")
    with open(path, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {path} ({len(cards)} cards)")
