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

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("PYJ_LIBPATH", os.path.join(_HERE, "jlibrary", "bin"))
sys.path.insert(0, _HERE)

import tally_ast as T  # noqa: E402

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "tally", "version": "0.1.0"}

AST_DOC = (
    "Tally AST. Every value is EXACT: integers stay integers, decimals and "
    "'p/q' strings become exact rationals -- 0.1 means 1/10, never a float. "
    "Forms: (1) number: 42 | true | \"0.1\" | \"22/7\" | \"1e-8\" (JSON "
    "numbers OK); (2) array: nested rectangular lists of numbers; "
    "(3) op node {\"op\": NAME, ...}. "
    "Arithmetic: add/sub/mul/div(args[], n-ary), neg/abs/sign/floor/ceil"
    "(arg), pow(base,exp integer exponent), sqrt(arg, digits?), exp(arg,"
    " digits?) -- sqrt/exp return certified rational convergents, "
    "factorial(arg), comb/perm(n,k), gcd/lcm(args[], integers), mod(a,b). "
    "Comparisons: eq/ne/lt/le/gt/ge(a,b) -> boolean, exact. "
    "Aggregates: sum/prod/min/max(args[] or of=array), mean/var/stddev(of). "
    "Linear algebra: matmul/dot(a,b), det/inv/transpose(m), solve(a,b) -- "
    "exact rational results (e.g. inverse of a Hilbert matrix is integer). "
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
                "digits": {"type": "integer", "minimum": 1, "maximum": 200,
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
                "digits": {"type": "integer", "minimum": 1, "maximum": 200,
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
                "digits": {"type": "integer", "minimum": 1, "maximum": 200,
                           "default": 30},
            },
            "required": ["data"],
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
    raise KeyError(name)


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
        try:
            value = _call_tool(name, params.get("arguments"))
        except KeyError:
            return _error(id_, -32602, f"unknown tool {name!r}")
        except T.TallyError as e:
            return _result(id_, _tool_error(str(e)))
        except Exception as e:                       # never leak a traceback
            _log("internal error:", repr(e))
            return _result(id_, _tool_error(f"internal: {e!r}"))
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
            msg = json.loads(raw)
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
