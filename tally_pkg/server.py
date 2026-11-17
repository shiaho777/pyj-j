#!/usr/bin/env python3
"""
tally_server.py -- Tally MCP server (stdio transport, NDJSON framing).

MCP stdio spec: one JSON-RPC 2.0 message per line on stdin/stdout.
(Content-Length framed input is also tolerated; responses are always NDJSON.)

Tools:
  tally_compute   evaluate an exact-arithmetic AST
  tally_verify    recompute exactly and grade a claimed value
  tally_stats     exact descriptive statistics over a dataset

Wire into Claude Desktop / any MCP host:
  {"mcpServers": {"tally": {
     "command": "/usr/bin/env", "args": ["python3", "/path/to/tally_server.py"],
     "env": {"PYJ_LIBPATH": "/path/to/jlibrary/bin",
             "DYLD_LIBRARY_PATH": "/path/to/jlibrary/bin"}}}}
Nothing may print to stdout except protocol messages; logs go to stderr.
"""
import json
import os
import sys
from fractions import Fraction

from tally import ast as T  # noqa: E402

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "tally", "version": "1.0.0"}

AST_DOC = (
    "Tally AST. Every value is EXACT: integers stay integers, decimals and "
    "'p/q' strings become exact rationals -- 0.1 means 1/10, never a float. "
    "Forms: (1) number: 42 | true | \"0.1\" | \"22/7\" | \"1e-8\" (JSON "
    "numbers OK); (2) array: nested rectangular lists of numbers; "
    "(3) op node {\"op\": NAME, ...}. "
    "Arithmetic: add/sub/mul/div(args[], n-ary), neg/abs/sign/floor/ceil"
    "(arg), round(arg, places -- half away from zero), pow(base,exp integer "
    "exponent), sqrt/ln(arg, digits?), exp(arg, digits?) -- certified "
    "rational convergents, pi(digits?) via Machin, sin/cos/tan/atan(arg, "
    "digits?) with exact range reduction, factorial(arg), comb/perm(n,k), "
    "gcd/lcm(args[], integers), mod(a,b). "
    "Comparisons: eq/ne/lt/le/gt/ge(a,b) -> boolean, exact. "
    "Aggregates: sum/prod/min/max(args[] or of=array), mean/var/stddev(of). "
    "Linear algebra: matmul/dot(a,b), det/inv/transpose(m), solve(a,b) -- "
    "exact rational results (e.g. inverse of a Hilbert matrix is integer). "
    "Finance (exact rational rates): fv/pv/pmt(rate, periods, payment?, "
    "pv?, fv?), nper(rate, payment, pv, fv?), irr(args=[c0,c1,...], "
    "digits?) -- loan/mortgage/annuity math with no float drift. "
    "Use this tool whenever a number must be RIGHT: money, rates, matrix "
    "results, combinatorics, or checking your own arithmetic."
)

TOOLS = [
    {
        "name": "tally_compute",
        "description": "Evaluate an exact-arithmetic expression. " + AST_DOC,
        "inputSchema": {
            "type": "object",
            "properties": {
                "expression": {"description": AST_DOC},
                "digits": {"type": "integer", "minimum": 1, "maximum": 1000,
                           "default": 30,
                           "description": "decimal expansion digits to return"},
            },
            "required": ["expression"],
        },
    },
    {
        "name": "tally_verify",
        "description": (
            "Recompute an expression EXACTLY and grade a claimed value "
            "(e.g. your own earlier computation). Reports match, and on "
            "mismatch the exact truth plus absolute/relative error. " + AST_DOC),
        "inputSchema": {
            "type": "object",
            "properties": {
                "expression": {"description": AST_DOC},
                "claimed": {"description":
                            "the claimed value: number, string, or nested array"},
                "digits": {"type": "integer", "minimum": 1, "maximum": 1000,
                           "default": 30},
            },
            "required": ["expression", "claimed"],
        },
    },
    {
        "name": "tally_stats",
        "description": (
            "Exact descriptive statistics over a dataset: mean, var "
            "(population, two-pass -- no cancellation), stddev, min, max, "
            "sum, count. Data points are parsed exactly."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "data": {"type": "array",
                         "description": "non-empty list of numbers/strings",
                         "items": {}},
                "ops": {"type": "array", "items": {"type": "string"},
                        "description": "subset of mean,var,stddev,min,max,sum,count"},
                "digits": {"type": "integer", "minimum": 1, "maximum": 1000,
                           "default": 30},
            },
            "required": ["data"],
        },
    },
    {
        "name": "tally_batch",
        "description": (
            "Evaluate MANY exact-arithmetic expressions in ONE call. Each "
            "item is {expression, digits?, label?}; results come back as a "
            "list in the same order, each a full compute envelope (exact, "
            "decimal, error_bound when convergent). Use this instead of "
            "many tally_compute calls when you have several numbers to "
            "check at once — one round trip, shared engine warm-up. " + AST_DOC),
        "inputSchema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "list of {expression, digits?, label?}",
                    "items": {
                        "type": "object",
                        "properties": {
                            "expression": {"description": AST_DOC},
                            "digits": {"type": "integer", "minimum": 1,
                                       "maximum": 1000, "default": 30},
                            "label": {"type": "string",
                                      "description": "optional tag echoed back"},
                        },
                        "required": ["expression"],
                    },
                },
            },
            "required": ["items"],
        },
    },
    {
        "name": "tally_compare",
        "description": (
            "Exactly compare two expressions: returns relation ('lt','eq',"
            "'gt') of a vs b, plus both exact values. For convergent ops "
            "the comparison is CERTIFIED when the error enclosures do not "
            "overlap (the relation is then provably true of the actual "
            "values); if they overlap it reports 'indeterminate_at_precision' "
            "and suggests more digits. Never compares binary floats. " + AST_DOC),
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"description": AST_DOC},
                "b": {"description": AST_DOC},
                "digits": {"type": "integer", "minimum": 1, "maximum": 1000,
                           "default": 30},
            },
            "required": ["a", "b"],
        },
    },
]


def _log(*a):
    print("[tally]", *a, file=sys.stderr, flush=True)


def _result(id_, value):
    return {"jsonrpc": "2.0", "id": id_, "result": value}


def _error(id_, code, message):
    return {"jsonrpc": "2.0", "id": id_,
            "error": {"code": code, "message": message}}


def _tool_result(value):
    return {"content": [{"type": "text",
                         "text": json.dumps(value, ensure_ascii=False, indent=2)}],
            "structuredContent": value}


def _tool_error(message):
    return {"content": [{"type": "text", "text": f"Tally error: {message}"}],
            "isError": True}


# Wall-clock guard for the Python-side iterative algorithms (irr bisection,
# atan range reduction, series term counting). The C engine calls themselves
# are bounded by the input limits (MAX_NODES / MAX_ARRAY_ELEMS / MAX_DIGITS).
CALL_TIMEOUT_S = float(os.environ.get("TALLY_TIMEOUT", "30"))


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


def _call_tool(name, args):
    if not isinstance(args, dict):
        raise T.TallyError("tool arguments must be an object")
    digits = args.get("digits", T.DEFAULT_DIGITS)
    if name == "tally_compute":
        if "expression" not in args:
            raise T.TallyError("missing 'expression'")
        return T.evaluate(args["expression"], digits=digits)
    if name == "tally_verify":
        if "expression" not in args or "claimed" not in args:
            raise T.TallyError("need 'expression' and 'claimed'")
        return T.verify(args["expression"], args["claimed"], digits=digits)
    if name == "tally_stats":
        if "data" not in args:
            raise T.TallyError("missing 'data'")
        return T.stats(args["data"], ops=tuple(args.get("ops") or
                       ("mean", "var", "stddev", "min", "max", "sum", "count")),
                       digits=digits)
    if name == "tally_batch":
        items = args.get("items")
        if not isinstance(items, list) or not items:
            raise T.TallyError("batch needs a non-empty 'items' list")
        if len(items) > 256:
            raise T.TallyError("batch too large (> 256 items)")
        results = []
        for i, item in enumerate(items):
            if not isinstance(item, dict) or "expression" not in item:
                results.append({"index": i, "ok": False,
                                "error": "item needs 'expression'"})
                continue
            d = item.get("digits", T.DEFAULT_DIGITS)
            try:
                env = T.evaluate(item["expression"], digits=d)
                if "label" in item:
                    env["label"] = item["label"]
                env["index"] = i
                results.append(env)
            except T.TallyError as e:
                results.append({"index": i, "ok": False, "error": str(e),
                                **({"label": item["label"]} if "label" in item else {})})
        return {"ok": True, "count": len(results), "results": results}
    if name == "tally_compare":
        if "a" not in args or "b" not in args:
            raise T.TallyError("compare needs 'a' and 'b'")
        ea = T.evaluate(args["a"], digits=digits)
        eb = T.evaluate(args["b"], digits=digits)
        va, vb = _scalar(ea), _scalar(eb)
        ba = Fraction(ea["error_bound"]) if ea.get("certified") else Fraction(0)
        bb = Fraction(eb["error_bound"]) if eb.get("certified") else Fraction(0)
        out = {"ok": True, "a_exact": ea["exact"], "b_exact": eb["exact"]}
        if va == vb and ba == 0 and bb == 0:
            out["relation"] = "eq"
            out["certified"] = True
        elif va - ba > vb + bb:
            out["relation"] = "gt"
            out["certified"] = True
        elif va + ba < vb - bb:
            out["relation"] = "lt"
            out["certified"] = True
        else:
            # enclosures overlap: relation not provable at this precision
            out["relation"] = ("gt" if va > vb else "lt" if va < vb else "eq")
            out["certified"] = False
            out["note"] = ("indeterminate_at_precision: error enclosures "
                           "overlap; increase digits to separate them")
            out["a_error_bound"] = ea.get("error_bound", "0")
            out["b_error_bound"] = eb.get("error_bound", "0")
        return out
    raise KeyError(name)


def _scalar(env):
    """Extract a scalar Fraction from a compute envelope (reject arrays)."""
    ex = env["exact"]
    if isinstance(ex, list):
        raise T.TallyError("compare needs scalar expressions, not arrays")
    return Fraction(ex)


def _handle(msg):
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        return _error(msg.get("id") if isinstance(msg, dict) else None,
                      -32600, "invalid JSON-RPC 2.0 request")
    method = msg.get("method")
    id_ = msg.get("id")
    params = msg.get("params") or {}

    if method == "initialize":
        return _result(id_, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
            "instructions":
                "Tally computes EXACTLY: integers, rationals, matrices, "
                "statistics -- no float error ever. When a number must be "
                "right, or to check your own arithmetic, call tally_compute "
                "or tally_verify."})
    if method == "ping":
        return _result(id_, {})
    if method == "tools/list":
        return _result(id_, {"tools": TOOLS})
    if method == "tools/call":
        name = params.get("name")
        import signal
        old = None
        if hasattr(signal, "SIGALRM"):
            old = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, CALL_TIMEOUT_S)
        try:
            value = _call_tool(name, params.get("arguments"))
        except KeyError:
            return _error(id_, -32602, f"unknown tool {name!r}")
        except _Timeout:
            return _result(id_, _tool_error(
                f"computation exceeded {CALL_TIMEOUT_S:.0f}s; simplify the "
                "expression or lower digits (set TALLY_TIMEOUT to adjust)"))
        except T.TallyError as e:
            return _result(id_, _tool_error(str(e)))
        except Exception as e:                       # never leak a traceback
            _log("internal error:", repr(e))
            return _result(id_, _tool_error(f"internal: {e!r}"))
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.setitimer(signal.ITIMER_REAL, 0)
                if old is not None:
                    signal.signal(signal.SIGALRM, old)
        return _result(id_, _tool_result(value))
    if method == "shutdown":
        return _result(id_, None)
    if id_ is None:                                  # notification: stay silent
        return None
    return _error(id_, -32601, f"method not found: {method!r}")


def _read_message(buf):
    """NDJSON per MCP spec; tolerate Content-Length framing too."""
    line = buf.readline()
    if not line:
        return None
    line = line.decode("utf-8", "replace") if isinstance(line, bytes) else line
    s = line.strip()
    if not s:
        return ""
    if s.lower().startswith("content-length:"):
        n = int(s.split(":", 1)[1])
        while True:                                  # consume remaining headers
            h = buf.readline()
            h = h.decode("utf-8", "replace") if isinstance(h, bytes) else h
            if h.strip() in ("", "\r\n"):
                break
        return buf.read(n).decode("utf-8", "replace")
    return s


def main():
    _log(f"up ({SERVER_INFO['name']} {SERVER_INFO['version']}, "
         f"protocol {PROTOCOL_VERSION})")
    buf = sys.stdin.buffer
    out = sys.stdout
    while True:
        try:
            raw = _read_message(buf)
        except Exception as e:
            _log("read error:", repr(e))
            break
        if raw is None:
            break
        if raw == "":
            continue
        try:
            # parse_float captures decimal literals as exact text BEFORE any
            # binary float can touch them — the core Tally contract applies
            # to the whole wire message, not just stringified expressions.
            msg = T.loads_exact(raw)
        except Exception:
            resp = _error(None, -32700, "parse error")
            out.write(json.dumps(resp) + "\n")
            out.flush()
            continue
        resp = _handle(msg)
        if resp is not None:
            out.write(json.dumps(resp, ensure_ascii=False) + "\n")
            out.flush()
        if isinstance(msg, dict) and msg.get("method") == "exit":
            break
    _log("down")


if __name__ == "__main__":
    main()
