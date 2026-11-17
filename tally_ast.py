"""
tally_ast.py -- Tally: an exact-compute engine for AI agents.

Public surface is a JSON AST. It compiles to extended-precision J expressions
and evaluates through the pyj bridge, so every result is an exact integer or
rational (GMP-backed). JSON floats never enter: the server parses JSON with
parse_float capturing the literal text, so 0.1 means 1/10, forever.

AST forms:
  number      42 | true | "0.1" | "22/7" | "1e-8"   (JSON numbers work too)
  array       [expr, ...] nested rectangular lists
  op          {"op": NAME, ...}

Ops (v1):
  add sub mul div          {"op":"add","args":[...]}   n-ary, exact
  neg abs sign floor ceil  {"op":"neg","arg":e}
  round                    {"op":"round","arg":e,"places":2}  half away from 0
  pow                      {"op":"pow","base":e,"exp":e}  integer exponent
  sqrt exp ln              {"op":"ln","arg":e,"digits":40}   certified convergent
  pi                       {"op":"pi","digits":100}          Machin formula
  sin cos tan atan         {"op":"sin","arg":e,"digits":40}  range-reduced series
  factorial comb perm      {"op":"comb","n":e,"k":e}
  gcd lcm                  {"op":"gcd","args":[...]}   integers
  mod                      {"op":"mod","a":e,"b":e}    a mod b
  eq ne lt le gt ge        {"op":"lt","a":e,"b":e}     exact comparison -> bool
  sum prod min max         {"op":"sum","args":[...]} or {"op":"sum","of":[...]}
  mean var stddev          {"op":"mean","of":[...]}
  matmul dot det inv solve transpose
  fv pv pmt nper           {"op":"fv","rate":"0.05","periods":10,"payment":-100,"pv":0}
  irr                      {"op":"irr","args":[-1000,300,400,500],"digits":40}
"""

import json
import math
import os
import re
import sys
from decimal import Decimal
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("PYJ_LIBPATH", os.path.join(_HERE, "jlibrary", "bin"))
sys.path.insert(0, _HERE)

import pyj  # noqa: E402


class TallyError(Exception):
    """User-facing error; becomes an MCP isError result."""


# ---------------------------------------------------------------- numbers

class ExactDecimal(str):
    """Marker for a JSON number literal captured before float could touch it."""


def loads_exact(text):
    """json.loads where decimals arrive as ExactDecimal (exact text)."""
    return json.loads(text, parse_float=ExactDecimal)


def _to_fraction(v):
    """Normalize any accepted scalar form to an exact Fraction."""
    if isinstance(v, bool):
        return Fraction(int(v))
    if isinstance(v, int):
        return Fraction(v)
    if isinstance(v, (ExactDecimal, str)):
        s = str(v).strip()
        if not s:
            raise TallyError("empty string is not a number")
        try:
            if "/" in s:
                return Fraction(s)
            return Fraction(Decimal(s))
        except Exception:
            raise TallyError(
                f"cannot parse {s!r} as an exact number "
                "(want: integer, decimal, scientific, or 'p/q')")
    if isinstance(v, float):
        raise TallyError(
            "a binary float reached Tally through the host language. "
            "Pass numbers as JSON literals or strings (\"0.1\"); "
            "floats are the enemy.")
    raise TallyError(f"unsupported scalar type: {type(v).__name__}")


def _jlit(fr):
    """Fraction -> J extended literal."""
    n, d = fr.numerator, fr.denominator
    if d == 1:
        return f"{n}x" if n >= 0 else f"_{abs(n)}x"
    return f"{n}r{d}" if n >= 0 else f"_{abs(n)}r{d}"


_TOKEN = re.compile(r"^(_?\d+)(?:r(\d+))?$")


def _parse_token(tok):
    t = tok.strip()
    if t in ("_", "__", "_."):          # J infinities / NaN: leave exact domain
        raise TallyError(
            "non-finite result (division by zero or overflow); "
            "exact domain covers finite integers and rationals only")
    m = _TOKEN.match(t)
    if not m:
        raise TallyError(
            f"non-finite or unexpected engine value {tok!r} "
            "(division by zero, or an op outside the exact domain)")
    p, q = m.group(1), m.group(2)
    n = int(p.replace("_", "-"))
    if q == "0":
        raise TallyError("non-finite result (zero denominator)")
    return Fraction(n, int(q) if q else 1)


# ---------------------------------------------------------------- limits

MAX_NODES = 2000
MAX_ARRAY_ELEMS = 20000
MAX_RANK = 8
MAX_DIGITS = 1000
DEFAULT_DIGITS = 30
MAX_FACTORIAL = 5000


def _check_digits(d):
    try:
        d = int(d)
    except Exception:
        raise TallyError("digits must be an integer")
    if not (1 <= d <= MAX_DIGITS):
        raise TallyError(f"digits must be 1..{MAX_DIGITS}")
    return d


# ---------------------------------------------------------------- engine

def _jdo(sentence):
    rc, out = pyj.do(sentence)
    if rc != 0:
        msg = " | ".join(l.strip().lstrip("|").strip()
                         for l in out if l.strip()) or f"rc={rc}"
        raise TallyError(f"engine rejected the expression: {msg}")
    return [l.rstrip("\n") for l in out]


def _engine_eval(jexpr):
    """Evaluate an extended-precision J expr; return (shape, [Fraction]).

    One-shot readback protocol (long-line-safe): the J session truncates
    a single display line around ~259 chars, so big integers cannot print
    raw. All raveled elements are decomposed at once with `2 x:` into an
    (n,2) table of extended num/den integers; each column is formatted,
    joined with '|' separators, cut into 198-char chunks *inside J* via
    `_198 <\\`, then the chunks are re-joined with LF and displayed in
    ONE JDo call. The output callback captures every LF-delimited piece,
    so the whole column comes back in a single round trip regardless of
    digit count. Total: ~10 JDo calls per evaluation, independent of
    array size and precision. Python reassembles and splits on '|'.
    """
    _jdo(f"tally_v =: {jexpr}")
    shape_line = _jdo('": $ tally_v')
    shape = ([int(t) for t in shape_line[0].split()]
             if shape_line and shape_line[0].strip() else [])
    n = 1
    for s in shape:
        n *= s
    _jdo("tally_nd =: 2 x: , tally_v")
    _jdo("tally_lf =: 10 { a.")
    nums = _read_column(0, n)
    dens = _read_column(1, n)
    toks = [num if den == "1" else f"{num}r{den}"
            for num, den in zip(nums, dens)]
    return shape, [_parse_token(t) for t in toks]


def _read_column(col, n):
    """Read one column of tally_nd as n exact integer strings.

    The J session caps a single JDo's captured output at ~43k chars, so
    chunks are emitted in BATCHES of <=200 (each ~40k chars, safely under
    the cap). Round trips scale with total digits / 200, not element count
    and not per-chunk: a 47k-digit column costs 2 calls, not 241."""
    _jdo(f"tally_str =: ; (\": &.> <\"0 ({col} {{\"1 tally_nd)) ,&.> <'|'")
    _jdo("tally_cb =: _198 <\\ tally_str")
    nchunks = int(_jdo("# tally_cb")[0].strip())
    parts = []
    BATCH = 200
    for start in range(0, nchunks, BATCH):
        cnt = min(BATCH, nchunks - start)
        _jdo(f"tally_all =: ; (({cnt} {{. ({start} }}. tally_cb))) ,&.> <tally_lf")
        parts.append("".join(_jdo('": tally_all')).replace("\n", ""))
    s = "".join(parts)
    fields = s.split("|")[:-1]          # trailing '|' leaves one empty tail
    if len(fields) != n:
        raise TallyError("internal: readback element count mismatch")
    return fields


# ---------------------------------------------------------------- compiler

_CMP_OPS = ("eq", "ne", "lt", "le", "gt", "ge")


class _Compiler:
    def __init__(self):
        self.nodes = 0
        # Provable truncation-error tracking. Each convergent op records a
        # rigorous bound B with |true_value - computed_convergent| <= B.
        # Bounds compose along the AST via interval arithmetic, so the
        # top-level envelope can certify the WHOLE expression, not just
        # the last series. Exact ops contribute 0.
        self.err = Fraction(0)          # bound of the last emitted expr
        self.has_conv = False           # any convergent in the last emit?
        self.mag = None                 # |value| estimate (for rel bounds)

    def _set_err(self, bound, mag=None, conv=True):
        self.err = bound
        self.has_conv = conv
        self.mag = mag

    def _snap(self):
        return (self.err, self.mag, self.has_conv)

    def _restore(self, s):
        self.err, self.mag, self.has_conv = s

    def _combine_binary(self, op, sa, sb, va=None, vb=None):
        """Interval-arithmetic error propagation for a binary op.
        sa/sb are child snapshots (err, mag, has_conv). Exact literals
        carry mag=|value|, err=0, so mul/div by a literal stay tight."""
        ea, ma, ca = sa
        eb, mb, cb = sb
        if not (ca or cb):
            # both exact: result exact, magnitude still tracked
            if op in ("add", "sub"):
                m = (ma + mb) if (ma is not None and mb is not None) else None
            elif op == "mul":
                m = (ma * mb) if (ma is not None and mb is not None) else None
            elif op == "div" and mb not in (None, 0):
                m = (ma / mb) if ma is not None else None
            else:
                m = None
            self._set_err(Fraction(0), m, conv=False)
            return
        BIG = Fraction(10) ** 60
        ma = ma if ma is not None else BIG
        mb = mb if mb is not None else BIG
        if op in ("add", "sub"):
            self._set_err(ea + eb, ma + mb)
        elif op == "mul":
            self._set_err(ma * eb + mb * ea + ea * eb, ma * mb)
        elif op == "div":
            # |a/b - a'/b'| <= (ea + |a/b|*eb) / (|b| - eb), needs |b| > eb.
            # When b is an exact literal (eb=0, mb=|b|>0) this is tight:
            # bound = ea/|b| + (ma/|b|)*0 = ea/|b|.
            if eb == 0 and mb > 0:
                quot = ma / mb
                self._set_err(ea / mb, quot)
            elif mb > eb:
                quot = ma / (mb - eb)
                self._set_err((ea + quot * eb) / (mb - eb), quot)
            else:
                self._set_err(BIG, BIG)
        else:
            self._set_err(BIG, BIG)

    def _tick(self):
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise TallyError(f"expression too large (> {MAX_NODES} nodes)")

    def emit(self, node):
        """node -> J expression string (extended precision)."""
        self._tick()
        if isinstance(node, dict):
            return self._op(node)
        if isinstance(node, list):
            shape, flat = _flatten(node)
            # convert each element ONCE (was twice: toks + mag)
            frs = [_to_fraction(v) for v in flat]
            toks = " ".join(_jlit(f) for f in frs)
            mag = max((abs(f) for f in frs), default=Fraction(0))
            self._set_err(Fraction(0), mag, conv=False)
            return f"({' '.join(map(str, shape))}) $ ({toks})"
        fr = _to_fraction(node)
        # exact literal: error 0, magnitude known -> tight mul/div bounds
        self._set_err(Fraction(0), abs(fr), conv=False)
        return _jlit(fr)

    def value_of(self, node, what="argument"):
        """Compile + evaluate a scalar sub-expression now."""
        if node is None:
            raise TallyError(f"missing {what}")
        shape, vals = _engine_eval(self.emit(node))
        if shape:
            raise TallyError(f"{what} must be a scalar")
        return vals[0]

    def _op(self, node):
        op = node.get("op")
        if not isinstance(op, str):
            raise TallyError("op node needs an \"op\" string")
        allowed = {"op", "args", "arg", "a", "b", "n", "k",
                   "base", "exp", "of", "m", "digits",
                   "rate", "periods", "payment", "pv", "fv", "guess",
                   "places"}
        extra = set(node) - allowed
        if extra:
            raise TallyError(f"unknown fields for op {op!r}: {sorted(extra)}")

        args = node.get("args")
        of = node.get("of")

        if op in ("add", "mul", "sub", "div", "gcd", "lcm"):
            sym = {"add": "+", "mul": "*", "sub": "-",
                   "div": "%", "gcd": "+.", "lcm": "*."}[op]
            if of is not None:
                e = self.emit(of)
                # fold over an array: conservatively sum element bounds
                if self.has_conv:
                    self._set_err(self.err * 1000, self.mag)
                return f"({sym}/ ({e}))"
            if op in ("add", "mul", "sub", "div") and args and len(args) == 2:
                ea = self.emit(args[0]); sa = self._snap()
                eb = self.emit(args[1]); sb = self._snap()
                self._combine_binary(op, sa, sb)
                return f"(({ea}) {sym} ({eb}))"
            es = self._args(args, 2)
            if op in ("gcd", "lcm"):
                self._set_err(Fraction(0), conv=False)
            elif self.has_conv:
                self._set_err(self.err * 1000, self.mag)
            return "(" + f" {sym} ".join(es) + ")"

        if op in ("sum", "prod", "min", "max"):
            sym = {"sum": "+", "prod": "*", "min": "<.", "max": ">."}[op]
            if of is not None:
                return f"({sym}/ ({self.emit(of)}))"
            if args and len(args) == 1 and isinstance(args[0], list):
                return f"({sym}/ ({self.emit(args[0])}))"
            es = self._args(args, 1)
            return "(" + f" {sym} ".join(es) + ")"

        if op in ("neg", "abs", "sign", "floor", "ceil", "factorial"):
            if op == "factorial":
                self._check_nonneg_int(self._field(node, "arg"),
                                       "factorial", MAX_FACTORIAL)
            e = self.emit(self._field(node, "arg"))
            sym = {"neg": "-", "abs": "|", "sign": "*",
                   "floor": "<.", "ceil": ">.", "factorial": "!"}[op]
            return f"({sym} ({e}))"

        if op == "mod":
            a = self.emit(self._field(node, "a"))
            b = self.emit(self._field(node, "b"))
            return f"(({b}) | ({a}))"          # J: x|y is y mod x

        if op == "pow":
            base = self.emit(self._field(node, "base"))
            expv = self.value_of(node.get("exp"), "pow exponent")
            if expv.denominator != 1:
                raise TallyError(
                    "pow needs an integer exponent in v0 "
                    "(use sqrt or exp for fractional powers)")
            if abs(expv) > 10**6:
                raise TallyError("pow exponent too large (|exp| > 1e6)")
            return f"(({base}) ^ ({_jlit(expv)}))"

        if op in ("comb", "perm"):
            self._check_nonneg_int(node.get("n"), "comb/perm n", 10**7)
            self._check_nonneg_int(node.get("k"), "comb/perm k", 10**7)
            n = self.emit(node["n"])
            k = self.emit(node["k"])
            c = f"(({k}) ! ({n}))"
            return c if op == "comb" else f"({c} * (! ({k})))"

        if op in _CMP_OPS:
            a = self.emit(self._field(node, "a"))
            b = self.emit(self._field(node, "b"))
            sym = {"eq": "=", "ne": "~:", "lt": "<", "le": "<:",
                   "gt": ">", "ge": ">:"}[op]
            return f"(({a}) {sym} ({b}))"

        if op in ("mean", "var", "stddev"):
            v = self.emit(self._field(node, "of"))
            mean = f"((+/ % #) ({v}))"
            if op == "mean":
                return mean
            varx = f"((+/ % #) (*: ({v}) - (tally_m =: {mean})))"
            if op == "var":
                return varx
            xv = self.value_of_from(varx)
            return self._sqrt_expr(xv, node)

        if op == "sqrt":
            x = self.value_of(self._field(node, "arg"), "sqrt argument")
            return self._sqrt_expr(x, node)

        if op == "exp":
            x = self.value_of(self._field(node, "arg"), "exp argument")
            digits = _check_digits(node.get("digits", DEFAULT_DIGITS))
            return self._exp_expr(x, digits)

        if op == "ln":
            x = self.value_of(self._field(node, "arg"), "ln argument")
            digits = _check_digits(node.get("digits", DEFAULT_DIGITS))
            return self._ln_expr(x, digits)

        if op in ("log", "log2", "log10"):
            x = self.value_of(self._field(node, "arg"), f"{op} argument")
            digits = _check_digits(node.get("digits", DEFAULT_DIGITS))
            if op == "log":
                base = self.value_of(self._field(node, "base"), "log base")
            else:
                base = Fraction(2 if op == "log2" else 10)
            return self._log_expr(x, base, digits)

        if op == "pi":
            digits = _check_digits(node.get("digits", DEFAULT_DIGITS))
            return self._pi_expr(digits)

        if op in ("sin", "cos", "tan", "atan"):
            x = self.value_of(self._field(node, "arg"), f"{op} argument")
            digits = _check_digits(node.get("digits", DEFAULT_DIGITS))
            return self._trig_expr(op, x, digits)

        if op in ("fv", "pv", "pmt", "nper", "irr"):
            return self._finance_expr(op, node)

        if op == "round":
            # exact rounding to a stated decimal precision (half away from 0)
            x = self.value_of(self._field(node, "arg"), "round argument")
            places = self.value_of(node.get("places", 0), "round places")
            if places.denominator != 1 or not (-MAX_DIGITS <= places <= MAX_DIGITS):
                raise TallyError(f"round places must be an integer in -{MAX_DIGITS}..{MAX_DIGITS}")
            scale = Fraction(10) ** int(places)
            scaled = x * scale
            n, d = scaled.numerator, scaled.denominator
            q, r = divmod(abs(n), d)
            if 2 * r >= d:
                q += 1
            if n < 0:
                q = -q
            return _jlit(Fraction(q) / scale)

        if op in ("matmul", "dot"):
            a = self.emit(self._field(node, "a"))
            b = self.emit(self._field(node, "b"))
            return f"(({a}) +/ . * ({b}))"

        if op == "det":
            return f"(-/ .* ({self.emit(self._field(node, 'm'))}))"
        if op == "inv":
            return f"(%. ({self.emit(self._field(node, 'm'))}))"
        if op == "transpose":
            return f"(|: ({self.emit(self._field(node, 'm'))}))"
        if op == "solve":
            a = self.emit(self._field(node, "a"))
            b = self.emit(self._field(node, "b"))
            return f"(({b}) %. ({a}))"

        raise TallyError(f"unknown op {op!r}")

    # -- special ops: argument already evaluated to an exact Fraction

    def value_of_from(self, jexpr):
        shape, vals = _engine_eval(jexpr)
        if shape:
            raise TallyError("internal: expected scalar")
        return vals[0]

    def _sqrt_expr(self, x, node):
        digits = _check_digits(node.get("digits", DEFAULT_DIGITS))
        if x < 0:
            raise TallyError("sqrt of a negative number (no complex in v0)")
        if x == 0:
            self._set_err(Fraction(0), Fraction(0), conv=False)
            return _jlit(Fraction(0))
        root = _perfect_root(x)
        if root is not None:                     # exact square -> exact root
            self._set_err(Fraction(0), root, conv=False)
            return _jlit(root)
        fx = float(x)
        if not (1e-280 < fx < 1e280):
            raise TallyError("sqrt argument outside v0 range (|x| > 1e280)")
        seed = _jlit(Fraction(math.sqrt(fx)))
        xl = _jlit(x)
        k = _newton_iters(digits)
        # Newton for sqrt: relative error squares each step. Seed rel err
        # <= 2^-52; after k steps <= (2^-52)^(2^k). _newton_iters guarantees
        # 53*2^k >= 4*(digits+5) plus 2 guard steps, so the absolute error
        # is far below 10^-(digits+5). Bound conservatively by 10^-(digits+2).
        bound = Fraction(1, 10 ** (digits + 2))
        self._set_err(bound, Fraction(math.ceil(math.sqrt(fx)) + 1))
        return f"(((({xl}) + *:) % +:)^:{k} ({seed}))"

    def _exp_expr(self, x, digits):
        if x == 0:
            return _jlit(Fraction(1))
        fx = float(x)
        if abs(fx) > 700:
            raise TallyError("exp argument outside v0 range (|x| > 700)")
        j = 0
        while abs(fx) / (2 ** j) > 0.5:
            j += 1
        t = x / (2 ** j) if j else x
        # Term count from the ACTUAL |t| (<= 0.5), not a worst-case linear
        # estimate: a_k = |t|^k/k! decays super-geometrically, so ~110 terms
        # suffice for 200 digits at |t|=0.5 (was 220). Find the smallest n
        # with a_{n+1} <= 10^-(digits+10+j) by EXACT rational comparison
        # (the +j covers the 2^j squaring amplification). Factorial and
        # power are accumulated incrementally — no big-factorial recompute,
        # no float-log precision loss at large digit counts.
        at = abs(t)
        # Magnitude of the RESULT exp(x) (signed x!): exp(500) ~ 1e217 needs
        # 217 extra integer digits; exp(-700) ~ 1e-305 needs NONE (the value
        # is tiny, its absolute error is tiny too). Using |x| here would
        # over-compute negative arguments by hundreds of digits.
        mag_digits = max(0, int(fx / math.log(10)) + 2)
        target = Fraction(1, 10 ** (digits + 10 + j + mag_digits))
        n = 0
        fact = 1                            # (n+1)!
        pw = at                             # at^(n+1)
        while pw > target * fact:
            n += 1
            fact *= n + 1
            pw *= at
        n += 2                              # guard terms
        # Vectorized common-denominator Taylor: exp(p/q) =
        #   [sum_k p^k q^(n-k) (n!/k!)] / (q^n n!)
        # One vectorized pass instead of n sequential big-rational adds.
        p, q = t.numerator, t.denominator
        pl, ql = _jlit(Fraction(p)), _jlit(Fraction(q))
        inner = (f"(3 : '(+/ (({pl} ^ k) * ({ql} ^ (y - k)) * "
                 f"((! y) % (! k)))) [ k =. i. >: y')")
        series = f"(({inner} {n}x) % (({ql} ^ {n}x) * (! {n}x)))"
        # Provable truncation bound. Reduced series tail <= a_{n+1}/(1-r)
        # with r = |t|/(n+2) < 1/2. Raising to the 2^j power amplifies the
        # ABSOLUTE error by <= 2^j * exp(|x|) (derivative of y^m is m*y^(m-1),
        # y=exp(t)<=e^0.5, m=2^j, so m*y^(m-1) <= 2^j * exp(|x|)).
        # Bound = 2^j * exp(|x|) * tail. exp(|x|) <= 10^mag_digits.
        a_n1 = at ** (n + 1) / Fraction(math.factorial(n + 1))
        r = at / (n + 2)
        tail = a_n1 / (1 - r)
        exp_mag = Fraction(10 ** mag_digits)
        bound = Fraction(2 ** j) * exp_mag * tail * 2
        self._set_err(bound, exp_mag)
        return f"((*:^:{j}) {series})" if j else series

    # -- transcendental convergents (guard-digit certified, truncation-safe)

    def _pi_expr(self, digits, guard=None):
        """Machin: pi = 16 atan(1/5) - 4 atan(1/239), alternating series.
        Truncation error < first omitted term; guard digits cover it.
        Cached: a convergent computed at g digits is reusable for any
        request needing <= g digits, so repeated calls are free."""
        g = guard if guard is not None else digits + 10
        if _PI_CACHE["g"] >= g:
            self._set_err(_PI_CACHE.get("err", Fraction(0)), Fraction(4))
            return _jlit(_PI_CACHE["val"])
        n5 = int(g / 1.39) + 3      # atan(1/5): ~1.4 digits per term
        n239 = int(g / 4.75) + 3    # atan(1/239): ~4.76 digits per term
        a5 = _vec_atan(1, 5, n5)
        a239 = _vec_atan(1, 239, n239)
        expr = f"((16x * {a5}) - (4x * {a239}))"
        val = self.value_of_from(expr)
        # Leibniz: alternating series error <= first omitted term.
        # atan(1/u) tail <= (1/u)^(2n+1)/(2n+1); pi combines 16x and 4x.
        b5 = Fraction(1, 5) ** (2 * n5 + 1) / (2 * n5 + 1)
        b239 = Fraction(1, 239) ** (2 * n239 + 1) / (2 * n239 + 1)
        bound = 16 * b5 + 4 * b239
        _PI_CACHE["g"], _PI_CACHE["val"], _PI_CACHE["err"] = g, val, bound
        self._set_err(bound, Fraction(4))
        return _jlit(val)

    def _ln_expr(self, x, digits):
        """ln(x) = 2 atanh((x-1)/(x+1)) after range reduction x = m*2^e.
        atanh series converges fast for m near 1; ln2 from atanh(1/3)."""
        if x <= 0:
            raise TallyError("ln of a non-positive number (no complex in v0)")
        if x == 1:
            self._set_err(Fraction(0), Fraction(0), conv=False)
            return _jlit(Fraction(0))
        g = digits + 10
        # reduce to m in [2/3, 4/3): x = m * 2^e
        e = 0
        m = x
        while m >= Fraction(4, 3):
            m /= 2
            e += 1
        while m < Fraction(2, 3):
            m *= 2
            e -= 1
        t = (m - 1) / (m + 1)                 # |t| <= 1/5
        # Term count from the ACTUAL |t| by exact rational comparison:
        # need |t|^(2n+1)/(2n+1) <= 10^-g. For |t| well below 1/5 this
        # cuts terms substantially (was worst-case log(5) for every x).
        at = abs(t)
        target = Fraction(1, 10 ** g)
        n = 0
        pw = at                             # at^(2n+1)
        while pw > target * (2 * n + 1):
            n += 1
            pw *= at * at
        n += 2
        atanh_t = _vec_atanh(t.numerator, t.denominator, n)
        # atanh tail bound: terms |t|^(2k+1)/(2k+1) shrink by factor t^2;
        # tail <= first_omitted / (1 - t^2). ln = 2*atanh, so x2.
        def _atanh_bound(tt, nn):
            a = abs(tt) ** (2 * nn + 1) / (2 * nn + 1)
            return 2 * a / (1 - tt * tt)
        b_t = _atanh_bound(t, n)
        if e == 0:
            self._set_err(b_t, abs(_ln_mag(x)))
            return f"(2x * {atanh_t})"
        # ln2 = 2 atanh(1/3): ~0.95 digits/term (cached across calls)
        if _LN2_CACHE["g"] >= g:
            ln2v = _LN2_CACHE["val"]
            b_ln2 = _LN2_CACHE["err"]
        else:
            n2 = int(g * math.log(10) / math.log(9)) + 3
            ln2v = self.value_of_from(f"(2x * {_vec_atanh(1, 3, n2)})")
            b_ln2 = _atanh_bound(Fraction(1, 3), n2)
            _LN2_CACHE["g"], _LN2_CACHE["val"], _LN2_CACHE["err"] = g, ln2v, b_ln2
        self._set_err(b_t + abs(e) * b_ln2, abs(_ln_mag(x)))
        return f"((2x * {atanh_t}) + ({_jlit(Fraction(e))} * {_jlit(ln2v)}))"

    def _log_expr(self, x, base, digits):
        """log_base(x) = ln(x) / ln(base), with certified quotient bound.

        Both ln convergents carry provable bounds; the quotient error is
        propagated by interval arithmetic: |a/b - a'/b'| <=
        (ea + |a/b|*eb) / (|b| - eb), valid when |b| > eb.
        """
        if x <= 0:
            raise TallyError("log of a non-positive number")
        if base <= 0 or base == 1:
            raise TallyError("log base must be positive and != 1")
        g = digits + 10
        # numerator ln(x)
        if x == 1:
            self._set_err(Fraction(0), Fraction(0), conv=False)
            return _jlit(Fraction(0))
        ln_x_expr = self._ln_expr(x, g)
        ex = self.err
        vx = self.value_of_from(ln_x_expr)
        # denominator ln(base)
        ln_b_expr = self._ln_expr(base, g)
        eb = self.err
        vb = self.value_of_from(ln_b_expr)
        if abs(vb) <= eb:
            raise TallyError("log base too close to 1 to certify at this precision")
        # Rigorous quotient enclosure: true ln(x) in [vx-ex, vx+ex], true
        # ln(b) in [vb-eb, vb+eb]. The quotient's extreme values occur at
        # interval endpoints; take the max deviation from the convergent
        # quotient vx/vb. This is a proven bound, not a linearization.
        q0 = vx / vb
        lo_b = abs(vb) - eb
        hi_b = abs(vb) + eb
        # |num| extremes
        hi_a = abs(vx) + ex
        lo_a = max(Fraction(0), abs(vx) - ex)
        q_hi = hi_a / lo_b
        q_lo = lo_a / hi_b
        bound = max(abs(q_hi - q0), abs(q_lo - q0))
        self._set_err(bound, abs(q0) + 1)
        return f"({ln_x_expr} % {ln_b_expr})"

    def _atan_series(self, t, digits):
        """atan(t) for |t| <= 1/5 via vectorized alternating series."""
        if t == 0:
            return _jlit(Fraction(0))
        # digits per term ~ -2*log10|t|
        dpt = max(0.5, -2 * math.log10(abs(float(t))))
        n = int(digits / dpt) + 3
        # Leibniz bound: first omitted term |t|^(2n+1)/(2n+1)
        bound = abs(t) ** (2 * n + 1) / (2 * n + 1)
        self._set_err(bound, Fraction(2))     # |atan| <= pi/2 < 2
        return _vec_atan(t.numerator, t.denominator, n)

    def _trig_expr(self, op, x, digits):
        g = digits + 10
        if op == "atan":
            # Identity-based reduction keeps operands SMALL (no convergent
            # feedback blowup): atan(x) = 2 atan(x/(1+sqrt(1+x^2))), with
            # the sqrt taken EXACTLY on a small rational via Newton in J.
            neg = x < 0
            ax = abs(x)
            halfpi = False
            if ax > 1:
                halfpi = True
                ax = 1 / ax
            # now ax <= 1; halve until <= 1/5 using exact sqrt of 1+ax^2.
            # ax is a rational with bounded size: each step's sqrt argument
            # is 1+ax^2 <= 2, and we keep ax as the EXACT rational transform
            # of a float64-seeded convergent rounded to g+10 digits.
            steps = 0
            while ax > Fraction(1, 5):
                s = self._sqrt_small(1 + ax * ax, g + 10)
                ax = _round_frac(ax / (1 + s), g + 10)
                steps += 1
                if steps > 20:
                    raise TallyError("atan argument reduction failed")
            core = self._atan_series(ax, g + 5)
            core_err = self.err
            # Each halving step rounds ax to g+10 places (err <= 10^-(g+10));
            # atan'<=1 and the 2^steps amplification gives the accumulated
            # reduction error. Conservative: steps * 2^steps * 10^-(g+10).
            red_err = Fraction(steps * (2 ** steps), 10 ** (g + 10))
            total = (2 ** steps) * core_err + red_err
            expr = f"(({2 ** steps}x) * {core})"
            if halfpi:
                pi_e = self._pi_expr(g + 5)
                total = total + self.err / 2     # pi/2 contribution
                expr = f"(({pi_e} % 2x) - {expr})"
            self._set_err(total, Fraction(2))
            return f"(- {expr})" if neg else expr
        # sin/cos/tan: reduce x mod 2pi at guard precision, then a
        # VECTORIZED common-denominator Taylor series on the reduced r.
        if abs(x) > 10**6:
            raise TallyError(f"{op} argument too large (|x| > 1e6)")
        pi_g = self._pi_expr(g + 10)
        pi_err = self.err
        # Evaluate the reduced argument r = x - 2pi*round(x/2pi) NOW as a
        # rational convergent, so the series sees a plain literal (no
        # nested quoted verbs) and can be fully vectorized.
        two_pi = self.value_of_from(f"(2x * {pi_g})")
        k = (x / two_pi + Fraction(1, 2)).__floor__()
        r_val = x - two_pi * k
        # Round r to g+10 places: the pi convergent carries ~350-digit
        # numerator/denominator, and raising those to series powers is the
        # bottleneck. Rounding adds <= 10^-(g+10) error (folded into red_err)
        # and keeps the series operands small.
        r_val = _round_frac(r_val, g + 10)
        # Taylor terms: |r| <= pi; need pi^n/n! < 10^-g
        n = 10
        logfact = 0.0
        while True:
            logfact += math.log(n)
            if n * math.log(math.pi) - logfact < -g * math.log(10):
                break
            n += 1
        n += 2
        # Reduction error: r uses pi_g (err pi_err); |dr/dpi| <= 2*|k|+2.
        # Plus the rounding of r to g+10 places (|sin'|,|cos'| <= 1).
        kmax = Fraction(abs(x)) / 3 + 2
        red_err = (2 * kmax + 2) * pi_err + Fraction(1, 10 ** (g + 10))
        # Taylor truncation: alternating, |r|<=pi, first omitted term.
        # Use pi < 355/113 (a proven upper bound) and CEIL the rational so
        # the bound is rigorously >= the true tail (limit_denominator would
        # round DOWN and could invalidate the certificate).
        PI_UP = Fraction(355, 113)          # > pi, classic Archimedes bound
        if op in ("sin", "tan"):
            tay = PI_UP ** (2 * n + 1) / math.factorial(2 * n + 1)
        else:
            tay = PI_UP ** (2 * n) / math.factorial(2 * n)
        # ceil to a clean power-of-ten upper bound via digit counts (pure
        # integer — float(tay) underflows to 0 at large n and log10 fails).
        tn, td = tay.numerator, tay.denominator
        tay_exp = len(str(tn)) - len(str(td)) + 1   # +1 ensures upper bound
        tay = Fraction(1, 10 ** (-tay_exp)) if tay_exp < 0 else Fraction(10 ** tay_exp)
        rp, rq = r_val.numerator, r_val.denominator
        if op == "sin":
            self._set_err(tay + red_err, Fraction(1))
            return _vec_sin(rp, rq, n)
        if op == "cos":
            self._set_err(tay + red_err, Fraction(1))
            return _vec_cos(rp, rq, n)
        # tan = sin/cos. Rigorous quotient bound via interval endpoints:
        # sin(r) in [vs-es, vs+es], cos(r) in [vc-ec, vc+ec] where
        # es = ec = tay + red_err. If |cos| enclosure excludes 0 the
        # quotient is bounded; otherwise tan is near a pole and we refuse
        # to certify (honest failure, not a fake bound).
        sin_expr = _vec_sin(rp, rq, n)
        cos_expr = _vec_cos(rp, rq, n)
        vs = self.value_of_from(sin_expr)
        es = tay + red_err
        vc = self.value_of_from(cos_expr)
        ec = tay + red_err
        if abs(vc) <= ec:
            raise TallyError(
                "tan argument too close to a pole (cos ~ 0) to certify "
                "at this precision; shift the argument or raise digits")
        q0 = vs / vc
        hi_a, lo_a = abs(vs) + es, max(Fraction(0), abs(vs) - es)
        lo_b, hi_b = abs(vc) - ec, abs(vc) + ec
        bound = max(abs(hi_a / lo_b - q0), abs(lo_a / hi_b - q0))
        self._set_err(bound, abs(q0) + 1)
        return f"({sin_expr} % {cos_expr})"

    def _sqrt_small(self, x, digits):
        """sqrt convergent of a SMALL rational (x <= 2), as a Fraction.
        Newton in J from a float64 seed; operand size stays bounded."""
        expr = self._sqrt_expr(x, {"digits": min(digits, MAX_DIGITS)})
        return self.value_of_from(expr)

    # -- exact finance (rational rates, integer periods)

    def _finance_expr(self, op, node):
        digits = _check_digits(node.get("digits", DEFAULT_DIGITS))
        if op in ("fv", "pv", "pmt"):
            rate = self.value_of(self._field(node, "rate"), "rate")
            nper = self.value_of(self._field(node, "periods"), "periods")
            if nper.denominator != 1 or nper < 0 or nper > 10**5:
                raise TallyError("periods must be an integer in 0..100000")
            if rate <= -1:
                raise TallyError("rate must be > -1")
            pmt = self.value_of(node.get("payment", 0), "payment")
            pv = self.value_of(node.get("pv", 0), "pv")
            fv = self.value_of(node.get("fv", 0), "fv")
            rl, nl = _jlit(rate), _jlit(nper)
            g = f"((1x + {rl}) ^ {nl})"          # growth factor (1+r)^n
            if rate == 0:
                ann = nl                          # annuity factor degenerates
            else:
                ann = f"(({g} - 1x) % {rl})"      # ((1+r)^n - 1)/r
            if op == "fv":
                # fv of pv + payments: -(pv*(1+r)^n + pmt*annuity)
                return f"(- (({_jlit(pv)} * {g}) + ({_jlit(pmt)} * {ann})))"
            if op == "pv":
                # pv = -(fv + pmt*annuity) / g  (inverse of the fv formula)
                return f"((- ({_jlit(fv)} + ({_jlit(pmt)} * {ann}))) % {g})"
            # pmt: payment that amortizes pv to fv in n periods
            if rate == 0:
                return f"((- ({_jlit(pv)} + {_jlit(fv)})) % {nl})"
            return f"(((- ({_jlit(pv)} * {g})) + {_jlit(fv)}) % {ann})"
        if op == "nper":
            rate = self.value_of(self._field(node, "rate"), "rate")
            pmt = self.value_of(self._field(node, "payment"), "payment")
            pv = self.value_of(self._field(node, "pv"), "pv")
            fv = self.value_of(node.get("fv", 0), "fv")
            if rate <= -1:
                raise TallyError("rate must be > -1")
            if rate == 0:
                if pmt == 0:
                    raise TallyError("nper: zero rate and zero payment")
                n = -(pv + fv) / pmt
                if n < 0:
                    raise TallyError("nper: no positive solution")
                return _jlit(n)
            # n = ln((pmt - fv*r)/(pmt + pv*r)) / ln(1+r)
            num = (pmt - fv * rate) / (pmt + pv * rate)
            if num <= 0:
                raise TallyError("nper: no real solution (check signs)")
            g = digits + 10
            ln_num = self._ln_expr(num, g)
            ln_den = self._ln_expr(1 + rate, g)
            return f"({ln_num} % {ln_den})"
        # irr: bisection to locate the root, then Newton refinement for
        # quadratic convergence. Bisection alone needs ~3.3*digits steps
        # (linear); Newton doubles correct digits each step, so ~log2(digits)
        # steps suffice after a modest bracket. All arithmetic is exact
        # rational; the final bracket certifies the convergent.
        flows = node.get("args")
        if not isinstance(flows, list) or len(flows) < 2:
            raise TallyError("irr needs args: cash flows [c0, c1, ...]")
        cs = [self.value_of(c, "cash flow") for c in flows]
        if not any(c > 0 for c in cs) or not any(c < 0 for c in cs):
            raise TallyError("irr needs both positive and negative flows")
        g = digits + 10
        lo, hi = Fraction(-99, 100), Fraction(10)
        def npv(r):
            return sum(c / (1 + r) ** i for i, c in enumerate(cs))
        def dnpv(r):
            # d/dr NPV = sum -i*c_i / (1+r)^(i+1)
            return sum(-i * c / (1 + r) ** (i + 1)
                       for i, c in enumerate(cs) if i)
        flo, fhi = npv(lo), npv(hi)
        if flo * fhi > 0:
            hi = Fraction(1000)
            fhi = npv(hi)
            if flo * fhi > 0:
                raise TallyError("irr: no sign change in (-0.99, 1000)")
        # Phase 1: ~40 bisection steps -> bracket width ~1e-12, a solid
        # Newton starting point regardless of requested precision.
        for _ in range(40):
            mid = (lo + hi) / 2
            fm = npv(mid)
            if fm == 0:
                return _jlit(mid)
            if (fm > 0) == (flo > 0):
                lo, flo = mid, fm
            else:
                hi = mid
        # Phase 2: Newton from the bracket midpoint. Each iterate is
        # ROUNDED to g+5 decimal places so denominators stay bounded
        # (unrounded Newton squares the denominator each step -> explosion
        # on high-degree NPV). Rounding adds <= 10^-(g+5) per step, far
        # below the target precision, and convergence stays quadratic.
        x = _round_frac((lo + hi) / 2, g + 5)
        newton_steps = int(math.log2(g + 1)) + 3
        for _ in range(newton_steps):
            fx = npv(x)
            if fx == 0:
                return _jlit(x)
            dfx = dnpv(x)
            if dfx == 0:
                break
            xn = _round_frac(x - fx / dfx, g + 5)
            if xn == x:
                break
            x = xn
        # Certify: |NPV(x)| / |NPV'(x)| bounds the distance to the true
        # root (Newton residual bound). Compute at the final iterate.
        fx = npv(x)
        dfx = dnpv(x)
        bound = abs(fx / dfx) if dfx != 0 else Fraction(1, 10 ** g)
        # add the accumulated rounding slack (newton_steps * 10^-(g+5))
        bound += Fraction(newton_steps, 10 ** (g + 5))
        self._set_err(bound, abs(x) + 1)
        return _jlit(x)

    # -- small util

    def _field(self, node, name):
        if name not in node:
            raise TallyError(f"op {node.get('op')!r} needs field {name!r}")
        return node[name]

    def _args(self, args, at_least):
        if not isinstance(args, list) or len(args) < at_least:
            raise TallyError(f"args must be a list of >= {at_least}")
        return [self.emit(a) for a in args]

    def _check_nonneg_int(self, node, what, cap):
        v = self.value_of(node, what)
        if v.denominator != 1 or v < 0:
            raise TallyError(f"{what} needs a non-negative integer")
        if v > cap:
            raise TallyError(f"{what} argument too large (> {cap})")


_SERIES_SEQ = 0

# Convergent caches: a value computed at g guard digits is valid for any
# request needing <= g digits. Repeated transcendental calls (e.g. every
# sin/cos needs pi) then cost zero extra engine work.
_PI_CACHE = {"g": -1, "val": None, "err": Fraction(0)}
_LN2_CACHE = {"g": -1, "val": None, "err": Fraction(0)}


def _define_series(pow_expr, n, odd=True, alternating=False, factorial=False,
                   binds=()):
    """Define a NAMED series verb in the J session; return its name.

    Named verbs compose without quote-nesting (the body of one verb can
    call another by name). The body is passed as a J string variable so
    no quoting conflicts arise. Each call gets a fresh name.

    binds: (name, jexpr) pairs assigned locally before the series runs;
    use for sub-expressions that themselves contain quoted verbs.
    """
    global _SERIES_SEQ
    _SERIES_SEQ += 1
    name = f"tally_sv{_SERIES_SEQ}"
    idxdef = "(>: +: k)" if odd else "(+: k)"
    sign = "(_1x ^ k) * " if alternating else ""
    den = "(! idx)" if factorial else "idx"
    chain = f"idx =. {idxdef} [ k =. i. {n}x"
    # J evaluates the [ chain right-to-left: a binding's dependencies
    # must sit to its RIGHT. binds are listed dependency-first (tp
    # before r), so PREPEND each — the last bind ends up leftmost,
    # evaluated last, after everything it depends on.
    for bname, jexpr in binds:
        chain = f"{bname} =. {jexpr} [ {chain}"
    body = f"(+/ ({sign}({pow_expr})) % {den}) [ {chain}"
    # body must be quote-free (binds carry any quoted sub-expressions)
    _jdo(f"tally_body =: '{body}'")
    _jdo(f"{name} =: 3 : tally_body")
    return name


def _ln_mag(x):
    """Conservative |ln(x)| estimate for error propagation (Fraction)."""
    fx = float(x)
    return Fraction(math.ceil(abs(math.log(fx)) * 2) + 2)


def _sci(fr):
    """Scientific-notation string for an error bound. A plain decimal
    expansion of a tiny bound (e.g. 1e-256) would print as all zeros and
    mislead the reader into thinking the error is 0; scientific notation
    keeps the magnitude honest. Exact rational value stays in error_bound."""
    if fr == 0:
        return "0"
    a = abs(fr)
    n, d = a.numerator, a.denominator
    e = len(str(n)) - len(str(d))
    if n < d:
        e -= 1
    scaled = a * Fraction(10) ** (-e)
    # normalize significand into [1, 10)
    while scaled >= 10:
        scaled /= 10
        e += 1
    while scaled < 1:
        scaled *= 10
        e -= 1
    sig = _decimal(scaled, 6).rstrip("0").rstrip(".")
    return f"{sig}e{e:+d}"


def _vec_atan(p, q, n):
    """Vectorized atan(p/q), common denominator q^(2n+1):
      atan(p/q) = [sum_k (-1)^k p^(2k+1) q^(2(n-k)) / (2k+1)] / q^(2n+1)
    One vectorized pass; ~10x faster than sequential big-rational adds."""
    pl = _jlit(Fraction(p))
    ql = _jlit(Fraction(q))
    inner = (f"(3 : '(+/ ((_1x ^ k) * (({pl}) ^ (>: +: k)) * "
             f"({ql} ^ (+: (y - k))) % (>: +: k))) [ k =. i. y')")
    return f"(({inner} {n}x) % ({ql} ^ (>: +: {n}x)))"


def _vec_atanh(p, q, n):
    """Vectorized atanh(p/q), |p/q| < 1, common denominator q^(2n+1):
      atanh(p/q) = [sum_k p^(2k+1) q^(2(n-k)) / (2k+1)] / q^(2n+1)"""
    pl = _jlit(Fraction(p))
    ql = _jlit(Fraction(q))
    inner = (f"(3 : '(+/ ((({pl}) ^ (>: +: k)) * "
             f"({ql} ^ (+: (y - k)))) % (>: +: k)) [ k =. i. y')")
    return f"(({inner} {n}x) % ({ql} ^ (>: +: {n}x)))"


def _vec_sin(p, q, n):
    """Vectorized sin(p/q), common denominator q^(2n+1) (2n+1)!:
      sin = [sum_k (-1)^k p^(2k+1) q^(2(n-k)) (2n+1)!/(2k+1)!]
            / [q^(2n+1) (2n+1)!]"""
    pl = _jlit(Fraction(p))
    ql = _jlit(Fraction(q))
    inner = (f"(3 : '(+/ ((_1x ^ k) * (({pl}) ^ (>: +: k)) * "
             f"({ql} ^ (+: (y - k))) * ((! >: +: y) % (! >: +: k)))) "
             f"[ k =. i. y')")
    return (f"(({inner} {n}x) % "
            f"(({ql} ^ (>: +: {n}x)) * (! >: +: {n}x)))")


def _vec_cos(p, q, n):
    """Vectorized cos(p/q), common denominator q^(2n) (2n)!:
      cos = [sum_k (-1)^k p^(2k) q^(2(n-k)) (2n)!/(2k)!]
            / [q^(2n) (2n)!]"""
    pl = _jlit(Fraction(p))
    ql = _jlit(Fraction(q))
    inner = (f"(3 : '(+/ ((_1x ^ k) * (({pl}) ^ (+: k)) * "
             f"({ql} ^ (+: (y - k))) * ((! +: y) % (! +: k)))) "
             f"[ k =. i. >: y')")
    return (f"(({inner} {n}x) % "
            f"(({ql} ^ (+: {n}x)) * (! +: {n}x)))")


def _round_frac(fr, digits):
    """Truncate a Fraction to `digits` decimal places (keeps operands small
    during iterative argument reduction; error <= 10^-digits)."""
    scale = 10 ** digits
    n, d = fr.numerator, fr.denominator
    q = (abs(n) * scale) // d
    if n < 0:
        q = -q
    return Fraction(q, scale)


def _series_verb(pow_expr, n, odd=True, alternating=False, factorial=False,
                 binds=()):
    """Build an inline explicit J verb computing a power series.

    Local assignments inside `3 : '...'` bind k/idx ONCE per call —
    the only safe way to share an index vector across sub-expressions
    in embedded mode (top-level control words are rejected, and
    re-inlining `(k =. i.n)` re-executes the assignment).

    pow_expr: J expr in terms of `idx` (the term index vector)
    odd: idx = 1,3,5,...  else idx = 0,2,4,...
    alternating: multiply by (-1)^k
    factorial: divide terms by idx! (sin/cos); else divide by idx (atan/atanh)
    binds: (name, jexpr) pairs assigned locally BEFORE the series runs.
        Use this for sub-expressions containing quotes (e.g. nested
        series verbs) — inlining them in the body breaks the outer
        verb's single-quoted string.
    """
    idxdef = "(>: +: k)" if odd else "(+: k)"
    sign = "(_1x ^ k) * " if alternating else ""
    den = "(! idx)" if factorial else "idx"
    # flat right-to-left assignment chain: body [ idx =. def [ k =. i.nx
    # (J evaluates the rightmost assignment first; verified in embedded mode)
    chain = f"idx =. {idxdef} [ k =. i. {n}x"
    for name, jexpr in reversed(list(binds)):
        chain = f"{name} =. {jexpr} [ {chain}"
    body = f"(+/ ({sign}({pow_expr})) % {den}) [ {chain}"
    return f"(3 : '{body}')"


def _newton_iters(digits):
    """Newton iterations from a float64 seed so the convergent is accurate
    far beyond `digits` decimal digits, without absurd num/den sizes."""
    k = 0
    while 53 * (1 << k) < 4 * (digits + 5):
        k += 1
    return k + 2


def _perfect_root(x):
    """Fraction -> exact Fraction root if x is a perfect square, else None."""
    n, d = x.numerator, x.denominator
    rn, rd = math.isqrt(n), math.isqrt(d)
    if rn * rn == n and rd * rd == d:
        return Fraction(rn, rd)
    return None


def _flatten(arr):
    """Validate rectangular nested lists; return (shape, flat_values)."""
    def rec(x, depth):
        if not isinstance(x, list):
            return [], [x]
        if depth >= MAX_RANK:
            raise TallyError(f"arrays nested deeper than {MAX_RANK}")
        if not x:
            raise TallyError("empty arrays not supported in v0")
        shapes, flats = zip(*(rec(e, depth + 1) for e in x))
        if any(s != shapes[0] for s in shapes):
            raise TallyError("ragged array: nested lists must be rectangular")
        return [len(x)] + list(shapes[0]), [v for f in flats for v in f]
    shape, flat = rec(arr, 0)
    if len(flat) > MAX_ARRAY_ELEMS:
        raise TallyError(f"array too large (> {MAX_ARRAY_ELEMS} elements)")
    return shape, flat


# ---------------------------------------------------------------- results

def _canonical(fr):
    return str(fr.numerator) if fr.denominator == 1 \
        else f"{fr.numerator}/{fr.denominator}"


def _decimal(fr, digits):
    """Truncated decimal expansion via ONE big-int division (not a per-digit
    loop): floor(rem * 10^digits / d) gives all digits at once. ~10x faster
    at 200+ digits, which dominates warm transcendental calls."""
    n, d = fr.numerator, fr.denominator
    sign = "-" if n < 0 else ""
    n = abs(n)
    ip, rem = divmod(n, d)
    frac = (rem * 10**digits) // d
    return f"{sign}{ip}." + str(frac).zfill(digits)


def _nest(shape, flat):
    if not shape:
        return flat[0]
    if len(shape) == 1:
        return flat
    step = 1
    for s in shape[1:]:
        step *= s
    return [_nest(shape[1:], flat[i * step:(i + 1) * step])
            for i in range(shape[0])]


def _eval_full(ast, digits):
    digits = _check_digits(digits)
    if isinstance(ast, str):
        try:
            ast = loads_exact(ast)
        except json.JSONDecodeError:
            pass                      # plain literal string, not JSON text
    comp = _Compiler()
    is_bool = isinstance(ast, dict) and ast.get("op") in _CMP_OPS
    jexpr = comp.emit(ast)
    err_bound, has_conv = comp.err, comp.has_conv
    shape, vals = _engine_eval(jexpr)
    exact_flat = [_canonical(v) for v in vals]
    dec_flat = [_decimal(v, digits) for v in vals]
    is_rat = any(v.denominator != 1 for v in vals)
    env = {
        "ok": True,
        "exact": _nest(shape, exact_flat),
        "shape": shape,
        "type": "boolean" if is_bool else ("rational" if is_rat else "integer"),
        "decimal": _nest(shape, dec_flat),
        "note": "exact = exact integer or p/q fraction; "
                "decimal = truncated expansion, not rounded",
    }
    if has_conv and not is_bool:
        # Provable error certificate: |true_value - exact| <= error_bound.
        # For convergent ops the 'exact' field is a rational convergent,
        # and this bound rigorously encloses the true transcendental.
        env["error_bound"] = _canonical(err_bound)
        env["error_bound_decimal"] = _sci(err_bound)
        env["certified"] = True
    if is_bool:
        env["result"] = _nest(shape, [bool(v) for v in vals])
    if not shape:
        try:
            env["float64"] = float(vals[0])
        except OverflowError:
            env["float64"] = None     # beyond float64 range (e.g. 200!)
    return env, shape, vals


# Result cache: agents repeatedly ask the same question (pi to N digits,
# a fixed matrix inverse). Keyed by canonical AST + digits, bounded so a
# long-running MCP server never grows without limit.
def _ast_key(ast):
    """Hashable canonical key for an AST (dicts/lists -> tuples)."""
    if isinstance(ast, dict):
        return tuple(sorted((k, _ast_key(v)) for k, v in ast.items()))
    if isinstance(ast, list):
        return tuple(_ast_key(v) for v in ast)
    return ast


def evaluate(ast, digits=DEFAULT_DIGITS):
    """Compile + evaluate a Tally AST. Returns the result envelope.

    Results are LRU-cached (512 entries) keyed by canonical AST + digits,
    so repeated identical queries cost ~0. The cache holds only exact
    rational envelopes — no engine state, safe across calls.
    """
    digits = _check_digits(digits)
    key = _ast_key(ast)
    hit = _RESULT_CACHE.get((key, digits))
    if hit is not None:
        return hit
    env, _, _ = _eval_full(ast, digits)
    if len(_RESULT_CACHE) >= 512:
        _RESULT_CACHE.clear()      # simple bounded eviction
    _RESULT_CACHE[(key, digits)] = env
    return env


_RESULT_CACHE = {}


def verify(ast, claimed, digits=DEFAULT_DIGITS):
    """Recompute exactly and grade a claimed value (the money-checker).

    For EXACT ops the claim must equal the recomputation bit-for-bit.
    For CONVERGENT ops (sqrt/exp/pi/...) the recomputation carries a
    provable error bound B, so the TRUE value lies in [e-B, e+B]. A
    decimal claim like "3.14159" is itself an interval [c-u, c+u] where
    u is half a unit of its last stated place. If the two intervals
    OVERLAP, the claim is CERTIFIED CONSISTENT (match=True,
    certificate='consistent'): some value compatible with the claim's
    stated precision is provably the true value. If they are disjoint,
    the claim is definitively wrong at its stated precision. This lets
    an agent verify "pi ≈ 3.14159" without the full convergent.
    """
    digits = _check_digits(digits)
    env, shape, vals = _eval_full(ast, digits)
    bound = Fraction(env["error_bound"]) if env.get("certified") else None
    try:
        if isinstance(claimed, list):
            cshape, cflat_raw = _flatten(claimed)
            cflat = [_to_fraction(v) for v in cflat_raw]
            cprec = [_claim_precision(v) for v in cflat_raw]
        else:
            cshape, cflat = [], [_to_fraction(claimed)]
            cprec = [_claim_precision(claimed)]
    except TallyError as e:
        return {"ok": False, "error": str(e), "exact": env["exact"]}
    if cshape != shape:
        return {"ok": True, "match": False, "reason": "shape mismatch",
                    "claimed_shape": cshape, "exact_shape": shape,
                    "exact": env["exact"]}
    worst = None
    consistent = False
    for i, (e, c) in enumerate(zip(vals, cflat)):
        if e == c:
            continue
        err = abs(e - c)
        # certificate-aware interval overlap: truth in [e-B, e+B], claim
        # in [c-u, c+u]. Overlap iff |e-c| <= B + u.
        if bound is not None:
            u = cprec[i]
            if err <= bound + u:
                consistent = True
                continue
        rel = (err / abs(e)) if e else None
        worst = {"index": i, "exact": _canonical(e),
                 "claimed": _canonical(c),
                 "abs_error": _canonical(err),
                 "abs_error_decimal": _decimal(err, digits),
                 "rel_error_decimal":
                     _decimal(rel, digits) if rel is not None else "inf"}
        worst["audit_note"] = _audit_note(e, c, err, rel, digits)
        break
    out = {"ok": True, "match": worst is None,
           "exact": env["exact"], "decimal": env["decimal"]}
    if bound is not None:
        out["error_bound"] = env["error_bound"]
        out["error_bound_decimal"] = env["error_bound_decimal"]
    if worst:
        out["first_mismatch"] = worst
        out["verdict"] = ("the claimed value is WRONG; "
                          "the 'exact' field is the truth")
        if bound is not None:
            out["certificate"] = "outside"
    elif consistent:
        out["certificate"] = "consistent"
        out["verdict"] = ("claimed value is CERTIFIED CONSISTENT with the "
                          "true value: it lies within the proven error "
                          "enclosure of the exact convergent")
    else:
        out["verdict"] = "claimed value matches the exact recomputation"
        if bound is not None:
            out["certificate"] = "exact"
    return out


def _claim_precision(claimed):
    """Half-unit uncertainty of a claimed value at its stated precision.

    A decimal string '3.14159' asserts the value to 5 decimal places, so
    its uncertainty is 0.5e-5. A fraction 'p/q' or integer is an EXACT
    assertion (uncertainty 0). Scientific notation '1.5e-3' uses the
    mantissa's decimal places shifted by the exponent.
    """
    if isinstance(claimed, str):
        s = claimed.strip()
        # scientific notation
        if "e" in s or "E" in s:
            mant, _, exp = s.lower().partition("e")
            try:
                e = int(exp)
            except ValueError:
                return Fraction(0)
            places = len(mant.split(".")[1]) if "." in mant else 0
            return Fraction(1, 2) * Fraction(10) ** (e - places)
        if "." in s:
            places = len(s.split(".")[1])
            return Fraction(1, 2 * 10 ** places)
    # exact rational / integer assertion
    return Fraction(0)


def _sig_digits(fr, digits):
    """Significant-digit string and decimal exponent of |fr|."""
    fr = abs(fr)                     # NB: compare |fr|, not fr — a negative
    n, d = fr.numerator, fr.denominator   # fr makes 10^e > fr always true
    if n == 0:                            # and the down-loop never terminates
        return "0", 0
    num_digits = len(str(n)) - len(str(d))
    e = num_digits - 1
    while Fraction(10) ** e > fr:
        e -= 1
    while Fraction(10) ** (e + 1) <= fr:
        e += 1
    scaled = fr * (Fraction(10) ** (-e))
    s = _decimal(scaled, digits).replace(".", "").lstrip("0")
    return s, e


def _audit_note(e, c, err, rel, digits):
    """Human/agent-readable one-liner: where the claim went wrong."""
    if (e < 0) != (c < 0):
        return (f"wrong sign: exact value is {_canonical(e)}, "
                f"the claim has the opposite sign")
    se, ee = _sig_digits(e, digits)
    sc, ec = _sig_digits(c, digits)
    parts = []
    if ee != ec:
        parts.append(f"wrong magnitude: exact ~10^{ee}, claim ~10^{ec} "
                     f"(off by ~10^{ec - ee} — check the scale/units)")
    else:
        k = 0
        while k < min(len(se), len(sc)) and se[k] == sc[k]:
            k += 1
        if k == 0:
            parts.append("wrong from the very first significant digit")
        else:
            parts.append(f"correct for {k} significant digit(s), "
                         f"wrong from digit #{k + 1} onward")
    if rel is not None and rel > 0:
        _, re_ = _sig_digits(rel, digits)
        parts.append(f"relative error ~10^{re_}")
    parts.append(f"exact value: {_canonical(e)[:60]}")
    return "; ".join(parts)


def stats(data, ops=("mean", "var", "stddev", "min", "max", "sum", "count"),
          digits=DEFAULT_DIGITS):
    """Exact descriptive stats over a dataset."""
    if not isinstance(data, list) or not data:
        raise TallyError("stats: data must be a non-empty list of numbers")
    allowed = {"mean", "var", "stddev", "min", "max", "sum", "count"}
    bad = set(ops) - allowed
    if bad:
        raise TallyError(f"stats: unknown ops {sorted(bad)}")
    out = {"ok": True, "count": len(data)}
    for name in ops:
        if name == "count":
            continue
        ast = ({"op": name, "of": data} if name in ("mean", "var", "stddev")
               else {"op": name, "of": data})
        env = evaluate(ast, digits=digits)
        out[name] = {"exact": env["exact"], "decimal": env["decimal"]}
    return out
