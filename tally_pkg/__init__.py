"""
tally -- exact-arithmetic kernel.

Every result is an exact integer or p/q rational. Floats never enter
the engine: 0.1 means 1/10, not a binary approximation.

Quick start:
    import tally
    tally.compute("0.1 + 0.2")          # -> '3/10'
    tally.compute("sqrt(2)", digits=50) # -> certified rational convergent
    tally.pi(100)                       # -> 100-digit pi as p/q
    tally.verify("1/3 + 1/6", "1/2")    # -> {'match': True, ...}

The J engine is bundled in the wheel; no setup, no env vars.
"""
from tally.ast import (  # noqa: F401
    evaluate, verify, stats, loads_exact, TallyError,
    DEFAULT_DIGITS, MAX_DIGITS,
)

__version__ = "1.0.0"
__all__ = ["compute", "pi", "verify", "stats", "evaluate",
           "TallyError", "__version__"]


def _expr_to_ast(expr):
    """Accept a JSON AST string, a JSON AST dict, or a simple infix string."""
    if isinstance(expr, dict):
        return expr
    if isinstance(expr, str):
        s = expr.strip()
        if s.startswith("{") or s.startswith("["):
            return loads_exact(s)
        return _parse_infix(s)
    raise TallyError("expression must be a string or AST dict")


def _parse_infix(s):
    """Minimal infix parser for the CLI: + - * / ^ ( ) and function calls.

    Numbers stay exact (parsed as rational literals). This is a convenience
    layer; the full power is the AST (see TALLY.md).
    """
    import re
    tokens = re.findall(r"\d+/\d+|\d*\.\d+(?:[eE][+-]?\d+)?|\d+|[a-zA-Z_]\w*|[+\-*/^(),]", s)
    pos = [0]

    def peek():
        return tokens[pos[0]] if pos[0] < len(tokens) else None

    def take():
        t = peek()
        pos[0] += 1
        return t

    def atom():
        t = take()
        if t is None:
            raise TallyError("unexpected end of expression")
        if t == "(":
            v = expr()
            if take() != ")":
                raise TallyError("missing )")
            return v
        if t == "-":
            return {"op": "neg", "arg": atom()}
        if re.fullmatch(r"\d+/\d+|\d*\.\d+(?:[eE][+-]?\d+)?|\d+", t):
            return t
        # function call: name(arg, ...)
        if peek() == "(":
            take()
            args = []
            if peek() != ")":
                args.append(expr())
                while peek() == ",":
                    take()
                    args.append(expr())
            if take() != ")":
                raise TallyError(f"missing ) in {t}(...)")
            return _call(t, args)
        raise TallyError(f"unknown token: {t}")

    def _call(name, args):
        unary = {"sqrt", "exp", "ln", "log2", "log10", "sin", "cos", "tan",
                 "atan", "neg", "abs", "sign", "floor", "ceil", "factorial"}
        if name in unary:
            return {"op": name, "arg": args[0]}
        if name == "log":
            return {"op": "log", "arg": args[0], "base": args[1]}
        if name == "pi":
            return {"op": "pi"}
        if name in ("add", "sub", "mul", "div", "gcd", "lcm",
                    "sum", "prod", "min", "max"):
            return {"op": name, "args": args}
        if name in ("pow",):
            return {"op": "pow", "base": args[0], "exp": args[1]}
        if name in ("mod",):
            return {"op": "mod", "a": args[0], "b": args[1]}
        if name in ("comb", "perm"):
            return {"op": name, "n": args[0], "k": args[1]}
        if name == "irr":
            return {"op": "irr", "args": args}
        if name in ("fv", "pv", "pmt"):
            # fv(rate, periods, payment?, pv?, fv?) positional
            keys = ["rate", "periods", "payment", "pv", "fv"]
            return {"op": name, **dict(zip(keys, args))}
        if name == "nper":
            keys = ["rate", "payment", "pv", "fv"]
            return {"op": "nper", **dict(zip(keys, args))}
        if name in ("det", "inv", "transpose"):
            return {"op": name, "m": args[0]}
        if name in ("matmul", "dot", "solve"):
            return {"op": name, "a": args[0], "b": args[1]}
        if name == "round":
            return {"op": "round", "arg": args[0],
                    **({"places": args[1]} if len(args) > 1 else {})}
        raise TallyError(f"unknown function: {name}")

    def term():
        v = atom()
        while peek() in ("*", "/"):
            op = take()
            r = atom()
            v = {"op": "mul" if op == "*" else "div", "args": [v, r]}
        return v

    def expr():
        v = term()
        while peek() in ("+", "-"):
            op = take()
            r = term()
            v = {"op": "add" if op == "+" else "sub", "args": [v, r]}
        return v

    result = expr()
    if pos[0] != len(tokens):
        raise TallyError(f"unexpected token: {peek()}")
    return result


def compute(expr, digits=DEFAULT_DIGITS):
    """Evaluate an expression exactly. Returns the result envelope.

    expr: infix string ("0.1+0.2"), JSON AST string, or AST dict.
    The 'exact' field is the canonical p/q (or integer) result.
    """
    return evaluate(_expr_to_ast(expr), digits=digits)


def pi(digits=DEFAULT_DIGITS):
    """pi to `digits` decimal places, as an exact rational convergent."""
    return evaluate({"op": "pi", "digits": digits}, digits=digits)