#!/usr/bin/env python3
"""
tally_fuzz.py -- large-scale differential verification of the Tally kernel.

Every result Tally produces is checked against an INDEPENDENT reference:
  * rational arithmetic / stats / finance -> Python fractions (exact)
  * exact linear algebra                  -> sympy Rational matrices (exact)
  * transcendental convergents            -> mpmath at digits+40 precision

Usage:
  python3 tally_fuzz.py            # ~2000 cases, ~1 min
  python3 tally_fuzz.py 20000      # scale up (cases per family)
  python3 tally_fuzz.py 2000 42    # fixed seed for reproducibility

Exit code 0 iff every case matches. A mismatch prints the full AST, the
Tally result and the reference value, then aborts (fail-fast: an exact
engine with one wrong answer is broken, not "mostly right").
"""
import json
import random
import sys
import time
from fractions import Fraction

import tally_ast as T

SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 20261116
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rng = random.Random(SEED)

CHECKS = 0
T0 = time.perf_counter()


def fail(family, ast, got, want, extra=""):
    print(f"\nMISMATCH in {family} (case #{CHECKS}, seed {SEED})")
    try:
        shown = json.dumps(ast)
    except TypeError:
        shown = str(ast)
    print("AST :", shown[:400])
    print("got :", str(got)[:200])
    print("want:", str(want)[:200])
    if extra:
        print(extra)
    sys.exit(1)


def ev(ast, digits=30):
    global CHECKS
    CHECKS += 1
    return T.evaluate(ast, digits=digits)


def rand_frac(maxnum=10**12, maxden=10**9):
    n = rng.randint(-maxnum, maxnum)
    d = rng.randint(1, maxden)
    return Fraction(n, d)


def frac_str(fr):
    return str(fr.numerator) if fr.denominator == 1 else \
        f"{fr.numerator}/{fr.denominator}"


def rand_dec():
    """Random decimal literal with up to 18 fractional digits."""
    n = rng.randint(-10**15, 10**15)
    places = rng.randint(0, 18)
    return f"{n / 10**places:.{places}f}" if places else str(n)


# ------------------------------------------------------------- families

def fam_arithmetic(n):
    for _ in range(n):
        a, b = rand_frac(), rand_frac()
        sa, sb = frac_str(a), frac_str(b)
        op = rng.choice(["add", "sub", "mul", "div", "mod"])
        if op == "div" and b == 0:
            continue
        if op == "mod" and b == 0:
            continue
        ast = {"op": op, "args": [sa, sb]} if op in ("add", "sub", "mul", "div") \
            else {"op": "mod", "a": sa, "b": sb}
        want = {"add": a + b, "sub": a - b, "mul": a * b,
                "div": a / b, "mod": a % b}[op]
        r = ev(ast)
        if Fraction(r["exact"]) != want:
            fail("arithmetic", ast, r["exact"], frac_str(want))
    # n-ary chains
    for _ in range(n // 3):
        xs = [rand_frac() for _ in range(rng.randint(2, 8))]
        r = ev({"op": "add", "args": [frac_str(x) for x in xs]})
        if Fraction(r["exact"]) != sum(xs):
            fail("n-ary add", xs, r["exact"], frac_str(sum(xs)))


def fam_decimal_literals(n):
    """JSON decimal literals must mean their exact decimal value."""
    for _ in range(n):
        s = rand_dec()
        want = Fraction(s)
        r = ev({"op": "add", "args": [s, "0"]})
        if Fraction(r["exact"]) != want:
            fail("decimal literal", s, r["exact"], frac_str(want))


def fam_integers(n):
    import math
    for _ in range(n):
        k = rng.randint(0, 300)
        r = ev({"op": "factorial", "arg": k})
        if int(r["exact"]) != math.factorial(k):
            fail("factorial", k, r["exact"][:60], str(math.factorial(k))[:60])
        a, b = rng.randint(1, 10**15), rng.randint(1, 10**15)
        r = ev({"op": "gcd", "args": [a, b]})
        if int(r["exact"]) != math.gcd(a, b):
            fail("gcd", (a, b), r["exact"], math.gcd(a, b))
        nn, kk = rng.randint(0, 200), rng.randint(0, 200)
        if kk <= nn:
            r = ev({"op": "comb", "n": nn, "k": kk})
            if int(r["exact"]) != math.comb(nn, kk):
                fail("comb", (nn, kk), r["exact"], math.comb(nn, kk))


def fam_stats(n):
    for _ in range(n):
        xs = [rand_frac(10**9, 10**6) for _ in range(rng.randint(2, 30))]
        data = [frac_str(x) for x in xs]
        mean = sum(xs) / len(xs)
        var = sum((x - mean) ** 2 for x in xs) / len(xs)
        r = ev({"op": "mean", "of": data})
        if Fraction(r["exact"]) != mean:
            fail("mean", data[:5], r["exact"], frac_str(mean))
        r = ev({"op": "var", "of": data})
        if Fraction(r["exact"]) != var:
            fail("var", data[:5], r["exact"], frac_str(var))
        r = ev({"op": "sum", "of": data})
        if Fraction(r["exact"]) != sum(xs):
            fail("sum", data[:5], r["exact"], frac_str(sum(xs)))


def fam_linalg(n):
    import sympy
    for _ in range(n):
        dim = rng.randint(2, 5)
        # integer matrices keep sympy fast; occasional rational entries
        def entry():
            if rng.random() < 0.25:
                return frac_str(rand_frac(100, 9))
            return str(rng.randint(-20, 20))
        A = [[entry() for _ in range(dim)] for _ in range(dim)]
        As = sympy.Matrix([[sympy.Rational(x) for x in row] for row in A])
        r = ev({"op": "det", "m": A})
        want = As.det()
        if Fraction(r["exact"]) != Fraction(want):
            fail("det", A, r["exact"], str(want))
        if As.det() != 0:
            r = ev({"op": "inv", "m": A})
            Ai = As.inv()
            got = [[Fraction(x) for x in row] for row in r["exact"]]
            wantm = [[Fraction(Ai[i, j]) for j in range(dim)] for i in range(dim)]
            if got != wantm:
                fail("inv", A, got, wantm)
            b = [str(rng.randint(-10, 10)) for _ in range(dim)]
            r = ev({"op": "solve", "a": A, "b": b})
            xs = As.solve(sympy.Matrix([sympy.Rational(x) for x in b]))
            got = [Fraction(x) for x in r["exact"]]
            wantv = [Fraction(xs[i]) for i in range(dim)]
            if got != wantv:
                fail("solve", (A, b), got, wantv)


def fam_transcendental(n):
    import mpmath
    for _ in range(n):
        digits = rng.choice([20, 40, 80, 150])
        mpmath.mp.dps = digits + 60
        op = rng.choice(["sqrt", "exp", "ln", "sin", "cos", "atan", "pi"])
        if op == "pi":
            ast, ref = {"op": "pi", "digits": digits}, mpmath.pi
        elif op == "sqrt":
            x = rand_frac(10**8, 10**4)
            x = abs(x) + Fraction(1, 10**6)
            ast = {"op": "sqrt", "arg": frac_str(x), "digits": digits}
            ref = mpmath.sqrt(mpmath.mpf(x.numerator) / x.denominator)
        elif op == "exp":
            x = rand_frac(100, 10)
            ast = {"op": "exp", "arg": frac_str(x), "digits": digits}
            ref = mpmath.exp(mpmath.mpf(x.numerator) / x.denominator)
        elif op == "ln":
            x = abs(rand_frac(10**8, 10**4)) + Fraction(1, 10**6)
            ast = {"op": "ln", "arg": frac_str(x), "digits": digits}
            ref = mpmath.log(mpmath.mpf(x.numerator) / x.denominator)
        elif op == "atan":
            x = rand_frac(10**4, 10**3)
            ast = {"op": "atan", "arg": frac_str(x), "digits": digits}
            ref = mpmath.atan(mpmath.mpf(x.numerator) / x.denominator)
        else:
            x = rand_frac(10**4, 10**2)
            ast = {"op": op, "arg": frac_str(x), "digits": digits}
            f = mpmath.sin if op == "sin" else mpmath.cos
            ref = f(mpmath.mpf(x.numerator) / x.denominator)
        r = ev(ast, digits=digits)
        got = Fraction(r["exact"])
        refs = mpmath.nstr(ref, digits + 40)
        err = abs(got - Fraction(refs))
        tol = Fraction(1, 10**digits)
        if err > tol:
            fail(op, ast, r["exact"][:80], refs[:80],
                 f"err={float(err):.3e} tol={float(tol):.3e}")


def fam_finance(n):
    for _ in range(n):
        rate = rand_frac(100, 1000)          # |rate| <= 0.1
        if rate <= -1:
            continue
        periods = rng.randint(1, 120)
        pv = rand_frac(10**6, 100)
        pmt = rand_frac(10**4, 100)
        g = (1 + rate) ** periods
        ann = periods if rate == 0 else (g - 1) / rate
        want_fv = -(pv * g + pmt * ann)
        r = ev({"op": "fv", "rate": frac_str(rate), "periods": periods,
                "pv": frac_str(pv), "payment": frac_str(pmt)})
        if Fraction(r["exact"]) != want_fv:
            fail("fv", (rate, periods, pv, pmt), r["exact"], frac_str(want_fv))
        # round-trip: pv of the fv-payment stream recovers pv exactly
        r2 = ev({"op": "pv", "rate": frac_str(rate), "periods": periods,
                 "payment": frac_str(pmt), "fv": r["exact"]})
        if Fraction(r2["exact"]) != pv:
            fail("fv->pv roundtrip", (rate, periods, pv, pmt),
                 r2["exact"], frac_str(pv))


def fam_verify(n):
    """verify() must accept exact claims and reject perturbed ones."""
    global CHECKS
    for _ in range(n):
        a, b = rand_frac(), rand_frac()
        ast = {"op": "mul", "args": [frac_str(a), frac_str(b)]}
        truth = a * b
        v = T.verify(ast, frac_str(truth))
        CHECKS += 1
        if not v.get("match"):
            fail("verify true claim", ast, v, frac_str(truth))
        perturbed = truth + Fraction(1, 10**20)
        v = T.verify(ast, frac_str(perturbed))
        CHECKS += 1
        if v.get("match"):
            fail("verify wrong claim accepted", ast, v, frac_str(perturbed))


FAMILIES = [
    ("arithmetic", fam_arithmetic),
    ("decimal-literals", fam_decimal_literals),
    ("integers", fam_integers),
    ("stats", fam_stats),
    ("linalg", fam_linalg),
    ("transcendental", fam_transcendental),
    ("finance", fam_finance),
    ("verify", fam_verify),
]


def main():
    print(f"tally_fuzz: seed={SEED}, ~{N} cases/family")
    for name, fn in FAMILIES:
        before = CHECKS
        t = time.perf_counter()
        fn(N)
        dt = time.perf_counter() - t
        print(f"  {name:18s} {CHECKS - before:6d} checks  {dt:7.1f}s  OK")
    total = time.perf_counter() - T0
    print(f"\nALL {CHECKS} DIFFERENTIAL CHECKS PASSED "
          f"({total:.1f}s, seed {SEED})")


if __name__ == "__main__":
    main()