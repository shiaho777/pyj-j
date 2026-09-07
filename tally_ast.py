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

Ops (v0):
  add sub mul div          {"op":"add","args":[...]}   n-ary, exact
  neg abs sign floor ceil  {"op":"neg","arg":e}
  pow                      {"op":"pow","base":e,"exp":e}  integer exponent
  sqrt exp                 {"op":"sqrt","arg":e,"digits":40}  certified convergent
  factorial comb perm      {"op":"comb","n":e,"k":e}
  gcd lcm                  {"op":"gcd","args":[...]}   integers
  mod                      {"op":"mod","a":e,"b":e}    a mod b
  eq ne lt le gt ge        {"op":"lt","a":e,"b":e}     exact comparison -> bool
  sum prod min max         {"op":"sum","args":[...]} or {"op":"sum","of":[...]}
  mean var stddev          {"op":"mean","of":[...]}
  matmul dot det inv solve transpose
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
MAX_DIGITS = 200
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

    Readback protocol (long-line-safe): the J session wraps/truncates
    display lines around ~259 chars, so a 375-digit integer cannot be
    printed raw. Instead, each raveled element is decomposed with
    `2 x:` into (numerator, denominator) extended integers, each big
    string is cut into 200-char chunks *inside J* via `_200 <\\`, and
    every chunk reads back as its own short line. Python reassembles.
    """
    _jdo(f"tally_v =: {jexpr}")
    shape_line = _jdo('": $ tally_v')
    shape = ([int(t) for t in shape_line[0].split()]
             if shape_line and shape_line[0].strip() else [])
    n = 1
    for s in shape:
        n *= s
    toks = []
    for i in range(n):
        _jdo(f"tally_e =: {i} {{ , tally_v")
        raw = _read_big('": tally_e')
        if raw in ("_", "__", "_."):
            raise TallyError(
                "non-finite result (division by zero or overflow); "
                "exact domain covers finite integers and rationals only")
        _jdo("tally_nd =: 2 x: tally_e")
        num = _read_big('": 0 { tally_nd')
        den = _read_big('": 1 { tally_nd')
        toks.append(num if den == "1" else f"{num}r{den}")
    return shape, [_parse_token(t) for t in toks]


def _read_big(varname):
    """Read a J string of arbitrary length, chunked J-side at 200 chars."""
    _jdo(f"tally_cb =: _200 <\\ {varname}")
    nchunks = int(_jdo("# tally_cb")[0].strip())
    return "".join(_jdo(f"> {c} {{ tally_cb")[0] for c in range(nchunks))


# ---------------------------------------------------------------- compiler

_CMP_OPS = ("eq", "ne", "lt", "le", "gt", "ge")


class _Compiler:
    def __init__(self):
        self.nodes = 0

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
            toks = " ".join(_jlit(_to_fraction(v)) for v in flat)
            return f"({' '.join(map(str, shape))}) $ ({toks})"
        return _jlit(_to_fraction(node))

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
                   "base", "exp", "of", "m", "digits"}
        extra = set(node) - allowed
        if extra:
            raise TallyError(f"unknown fields for op {op!r}: {sorted(extra)}")

        args = node.get("args")
        of = node.get("of")

        if op in ("add", "mul", "sub", "div", "gcd", "lcm"):
            sym = {"add": "+", "mul": "*", "sub": "-",
                   "div": "%", "gcd": "+.", "lcm": "*."}[op]
            if of is not None:
                return f"({sym}/ ({self.emit(of)}))"
            es = self._args(args, 2)
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
            return _jlit(Fraction(0))
        root = _perfect_root(x)
        if root is not None:                     # exact square -> exact root
            return _jlit(root)
        fx = float(x)
        if not (1e-280 < fx < 1e280):
            raise TallyError("sqrt argument outside v0 range (|x| > 1e280)")
        seed = _jlit(Fraction(math.sqrt(fx)))
        xl = _jlit(x)
        k = _newton_iters(digits)
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
        n = digits + 20 + 8 * j
        xl = _jlit(x)
        t = f"(({xl}) % ({_jlit(Fraction(2 ** j))}))" if j else xl
        series = f"(+/ (({t})^(k)) % ! (k =. i. {n + 1}x))"
        return f"((*:^:{j}) {series})" if j else series

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
    n, d = fr.numerator, fr.denominator
    sign = "-" if n < 0 else ""
    n = abs(n)
    ip, rem = divmod(n, d)
    out = []
    for _ in range(digits):
        rem *= 10
        q, rem = divmod(rem, d)
        out.append(str(q))
    return f"{sign}{ip}." + "".join(out)


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
    shape, vals = _engine_eval(comp.emit(ast))
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
    if is_bool:
        env["result"] = _nest(shape, [bool(v) for v in vals])
    if not shape:
        try:
            env["float64"] = float(vals[0])
        except OverflowError:
            env["float64"] = None     # beyond float64 range (e.g. 200!)
    return env, shape, vals


def evaluate(ast, digits=DEFAULT_DIGITS):
    """Compile + evaluate a Tally AST. Returns the result envelope."""
    env, _, _ = _eval_full(ast, digits)
    return env


def verify(ast, claimed, digits=DEFAULT_DIGITS):
    """Recompute exactly and grade a claimed value (the money-checker)."""
    digits = _check_digits(digits)
    env, shape, vals = _eval_full(ast, digits)
    try:
        if isinstance(claimed, list):
            cshape, cflat_raw = _flatten(claimed)
            cflat = [_to_fraction(v) for v in cflat_raw]
        else:
            cshape, cflat = [], [_to_fraction(claimed)]
    except TallyError as e:
        return {"ok": False, "error": str(e), "exact": env["exact"]}
    if cshape != shape:
        return {"ok": True, "match": False, "reason": "shape mismatch",
                    "claimed_shape": cshape, "exact_shape": shape,
                    "exact": env["exact"]}
    worst = None
    for i, (e, c) in enumerate(zip(vals, cflat)):
        if e != c:
            err = abs(e - c)
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
    if worst:
        out["first_mismatch"] = worst
        out["verdict"] = ("the claimed value is WRONG; "
                          "the 'exact' field is the truth")
    else:
        out["verdict"] = "claimed value matches the exact recomputation"
    return out


def _sig_digits(fr, digits):
    """Significant-digit string and decimal exponent of |fr|."""
    n, d = fr.numerator, fr.denominator
    n = abs(n)
    if n == 0:
        return "0", 0
    num_digits = len(str(n)) - len(str(d))
    lo, hi = num_digits - 1, num_digits + 1
    e = lo
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
