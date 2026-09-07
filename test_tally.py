#!/usr/bin/env python3
"""
test_tally.py -- Tally v0 test suite.

1. Unit tests: every AST op + exactness contracts from the shootout report.
2. MCP stdio integration: real handshake over pipes (NDJSON framing per MCP
   spec), unknown method -> JSON-RPC error, tool error -> isError result.
"""
import json
import os
import subprocess
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tally_ast as T

PASS = 0
FAIL = 0
FAILURES = []


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        FAILURES.append(name)
        print(f"  FAIL {name}  {detail}")


def ev(ast, **kw):
    return T.evaluate(T.loads_exact(json.dumps(ast)), **kw)


def unit_tests():
    print("== unit: exact scalars & shootout cases ==")
    r = ev({"op": "add", "args": ["0.1", "0.2"]})
    check("0.1+0.2 == 3/10", r["exact"] == "3/10", r)
    check("0.1+0.2 decimal", r["decimal"].startswith("0.300000000000000000"), r["decimal"])

    r = ev({"op": "eq", "a": {"op": "sum", "of": ["0.1"] * 10}, "b": 1})
    check("sum(0.1 x10) == 1", r["result"] is True, r)

    r = ev({"op": "eq", "a": {"op": "add", "args": ["0.1", "0.1", "0.1"]}, "b": "0.3"})
    check("0.1+0.1+0.1 == 0.3 (exact)", r["result"] is True, r)

    r = ev({"op": "pow", "base": "100000", "exp": 0})
    check("pow x^0", r["exact"] == "1", r)

    r = ev({"op": "mul", "args": ["100000", "1.0041666666666667"]})
    check("money mul rational", r["type"] == "rational", r)

    r = ev({"op": "pow", "base": "1.00416666666666666666666666666667", "exp": 360})
    check("compound ~ 4.4677443", abs(r["float64"] - 4.4677443) < 1e-4, r["float64"])

    r = ev({"op": "pow", "base": "100000", "exp": {"op": "mul", "args": [2, 3]}})
    check("pow computed exp", r["exact"] == str(100000**6), r)

    r = ev({"op": "det", "m": [[f"1/{i+j+1}" for j in range(4)] for i in range(4)]})
    check("det Hilbert4 == 1/6048000", r["exact"] == "1/6048000", r)

    r = ev({"op": "inv", "m": [[f"1/{i+j+1}" for j in range(4)] for i in range(4)]})
    check("inv Hilbert4 integer matrix",
          r["type"] == "integer" and r["shape"] == [4, 4], r["type"])
    check("inv Hilbert4 [0][0] == 16", r["exact"][0][0] == "16", r["exact"][0])
    check("inv Hilbert4 [3][3] == 2800", r["exact"][3][3] == "2800", r["exact"][3])

    r = ev({"op": "solve",
            "a": [[f"1/{i+j+1}" for j in range(4)] for i in range(4)],
            "b": [1, 1, 1, 1]})
    check("solve Hilbert4 exact", r["exact"] == ["-4", "60", "-180", "140"], r)

    r = ev({"op": "matmul", "a": [["1.5", "2.5"], ["3.5", "4.5"]], "b": [[1, 0], [0, 1]]})
    check("matmul rational", r["exact"] == [["3/2", "5/2"], ["7/2", "9/2"]], r)

    r = ev({"op": "var", "of": [str(10**9 + i) for i in range(10)]})
    check("var 1e9+0..9 == 33/4", r["exact"] == "33/4", r)

    r = ev({"op": "mean", "of": [str(10**9 + i) for i in range(10)]})
    check("mean 1e9+0..9 == 2000000009/2", r["exact"] == "2000000009/2", r)

    r = ev({"op": "stddev", "of": [str(10**9 + i) for i in range(10)]})
    c = Fraction(r["exact"])
    check("stddev^2 ~ 33/4", abs(c * c - Fraction(33, 4)) < Fraction(1, 10**25), r["exact"][:40])

    r = ev({"op": "sqrt", "arg": 2, "digits": 60})
    c = Fraction(r["exact"])
    check("sqrt(2) convergent", abs(c * c - 2) < Fraction(1, 10**50), r["exact"][:40])
    check("sqrt(2) decimal", r["decimal"].startswith("1.414213562373095048801688724209"), r["decimal"])

    r = ev({"op": "sqrt", "arg": "1e-16", "digits": 40})
    check("sqrt(1e-16) == 1/1e8 exact", r["exact"] == "1/100000000", r)

    r = ev({"op": "sqrt", "arg": 4, "digits": 40})
    check("sqrt(4) == 2 exact", r["exact"] == "2", r)

    r = ev({"op": "exp", "arg": 0, "digits": 40})
    check("exp(0) == 1", r["exact"] == "1", r)

    r = ev({"op": "exp", "arg": 1, "digits": 40})
    check("exp(1) ~ e",
          r["decimal"].startswith("2.71828182845904523536028747135"), r["decimal"])

    r = ev({"op": "exp", "arg": -5, "digits": 40})
    check("exp(-5) ~ 0.006737946999085467",
          r["decimal"].startswith("0.006737946999085467"), r["decimal"])

    r = ev({"op": "exp", "arg": 10, "digits": 40})
    check("exp(10) ~ 22026.46579480671651",
          r["decimal"].startswith("22026.46579480671651"), r["decimal"])

    r = ev({"op": "factorial", "arg": 100})
    check("100! exact",
          r["exact"] == str(93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000),
          r["exact"][:40])

    r = ev({"op": "comb", "n": 100, "k": 50})
    check("C(100,50)", r["exact"] == "100891344545564193334812497256", r)

    r = ev({"op": "perm", "n": 10, "k": 3})
    check("P(10,3) == 720", r["exact"] == "720", r)

    r = ev({"op": "gcd", "args": [1071, 462]})
    check("gcd(1071,462)==21", r["exact"] == "21", r)

    r = ev({"op": "lcm", "args": [12, 18]})
    check("lcm(12,18)==36", r["exact"] == "36", r)

    r = ev({"op": "mod", "a": 23, "b": 7})
    check("23 mod 7 == 2", r["exact"] == "2", r)

    r = ev({"op": "mod", "a": -23, "b": 7})
    check("-23 mod 7 == 5 (floor-mod)", r["exact"] == "5", r)

    r = ev({"op": "floor", "arg": "-3.5"})
    check("floor(-3.5) == -4", r["exact"] == "-4", r)
    r = ev({"op": "ceil", "arg": "-3.5"})
    check("ceil(-3.5) == -3", r["exact"] == "-3", r)
    r = ev({"op": "neg", "arg": "0.5"})
    check("neg 0.5 == -1/2", r["exact"] == "-1/2", r)
    r = ev({"op": "sign", "arg": "-0.001"})
    check("sign(-0.001) == -1", r["exact"] == "-1", r)

    r = ev({"op": "lt", "a": "1/3", "b": "0.333333333333333333333333333333333333"})
    check("1/3 < 0.333...33 is False", r["result"] is False, r)
    r = ev({"op": "gt", "a": "1/3", "b": "0.333333333333333333333333333333333333"})
    check("1/3 > 0.333...33", r["result"] is True, r)

    r = ev({"op": "transpose", "m": [[1, 2, 3], [4, 5, 6]]})
    check("transpose", r["exact"] == [["1", "4"], ["2", "5"], ["3", "6"]], r)

    r = ev({"op": "dot", "a": ["0.1", "0.2"], "b": ["0.3", "0.4"]})
    check("dot == 11/100", r["exact"] == "11/100", r)

    print("== unit: edge & error handling ==")
    for name, ast, frag in [
        ("div by zero", {"op": "div", "args": [1, 0]}, "non-finite"),
        ("sqrt negative", {"op": "sqrt", "arg": -1}, "negative"),
        ("unknown op", {"op": "wibble", "args": [1]}, "unknown op"),
        ("ragged array", [[1, 2], [3]], "ragged"),
        ("bad number", "abc", "exact number"),
        ("pow frac exp", {"op": "pow", "base": 2, "exp": "0.5"}, "integer exponent"),
        ("digits too big", {"op": "sqrt", "arg": 2, "digits": 999}, "digits"),
    ]:
        try:
            ev(ast)
            check(f"error: {name}", False, "no exception raised")
        except T.TallyError as e:
            check(f"error: {name}", frag.lower() in str(e).lower(), str(e))

    try:
        T.evaluate(0.30000000000000004)          # programmatic float
        check("error: float leaked", False, "no exception")
    except T.TallyError as e:
        check("error: float leaked", "enemy" in str(e), str(e))

    r = ev({"op": "factorial", "arg": 200})
    check("200! survives print-width (375 digits)",
          len(r["exact"]) == 375, len(r["exact"]))
    check("200! ends with 49 zeros", r["exact"].endswith("0" * 49), r["exact"][-55:])

    print("== unit: stats & verify ==")
    s = T.stats([str(10**9 + i) for i in range(10)])
    check("stats mean", s["mean"]["exact"] == "2000000009/2", s["mean"])
    check("stats var", s["var"]["exact"] == "33/4", s["var"])

    v = T.verify({"op": "pow", "base": "1.00416666666666666666666666666667", "exp": 360},
                 claimed="6.0225")
    check("verify mismatch flagged", v["match"] is False, v)
    check("verify gives truth", "first_mismatch" in v, v)

    v = T.verify({"op": "add", "args": ["0.1", "0.2"]}, claimed="0.3")
    check("verify correct claim", v["match"] is True, v)

    v = T.verify({"op": "add", "args": ["0.1", "0.2"]},
                 claimed="0.30000000000000004")
    check("verify float64 claim rejected", v["match"] is False, v)
    check("verify abs_error exact",
          v["first_mismatch"]["abs_error"] == "1/25000000000000000",
          v["first_mismatch"])


# ------------------------------------------------------------- MCP layer

class MCPClient:
    """Minimal NDJSON MCP stdio client (one JSON-RPC message per line)."""

    def __init__(self):
        env = dict(os.environ)
        env.setdefault("DYLD_LIBRARY_PATH", os.path.join(HERE, "jlibrary", "bin"))
        env.setdefault("PYJ_LIBPATH", os.path.join(HERE, "jlibrary", "bin"))
        self.p = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "tally_server.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        self._id = 0

    def send(self, method, params=None, is_notif=False):
        self._id += 1
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if not is_notif:
            msg["id"] = self._id
        self.p.stdin.write(json.dumps(msg).encode() + b"\n")
        self.p.stdin.flush()

    def call(self, method, params=None, timeout=60):
        rid = self._id + 1
        self.send(method, params)
        import selectors
        sel = selectors.DefaultSelector()
        sel.register(self.p.stdout, selectors.EVENT_READ)
        while True:
            if not sel.select(timeout):
                raise TimeoutError(f"no response to {method}")
            line = self.p.stdout.readline()
            if not line:
                raise BrokenPipeError("server closed stdout")
            m = json.loads(line.decode())
            if m.get("id") == rid:
                return m

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=10)


def mcp_tests():
    print("== mcp: stdio integration ==")
    c = MCPClient()
    try:
        r = c.call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "tally-test", "version": "0"}})
        check("initialize result", "result" in r and "error" not in r, r)
        check("server name", r["result"]["serverInfo"]["name"] == "tally", r)
        c.send("notifications/initialized", is_notif=True)

        r = c.call("tools/list")
        names = [t["name"] for t in r["result"]["tools"]]
        check("tools/list has 3 tools",
              names == ["tally_compute", "tally_verify", "tally_stats"], names)
        schema = r["result"]["tools"][0]["inputSchema"]
        check("compute schema has expression", "expression" in schema["properties"], schema)

        r = c.call("tools/call", {"name": "tally_compute", "arguments": {
            "expression": {"op": "add", "args": ["0.1", "0.2"]}, "digits": 30}})
        payload = json.loads(r["result"]["content"][0]["text"])
        check("compute 0.1+0.2 == 3/10", payload["exact"] == "3/10", payload)
        check("compute decimal", payload["decimal"].startswith("0.3000000000000000"), payload)

        r = c.call("tools/call", {"name": "tally_compute", "arguments": {
            "expression": {"op": "det", "m": [[f"1/{i+j+1}" for j in range(4)] for i in range(4)]}}})
        payload = json.loads(r["result"]["content"][0]["text"])
        check("compute det Hilbert4", payload["exact"] == "1/6048000", payload)

        r = c.call("tools/call", {"name": "tally_verify", "arguments": {
            "expression": {"op": "pow", "base": "1.00416666666666666666666666666667", "exp": 360},
            "claimed": "446774.4314"}})
        payload = json.loads(r["result"]["content"][0]["text"])
        check("verify catches wrong claim", payload["match"] is False, payload)
        check("verify verdict text", "WRONG" in payload["verdict"], payload)

        r = c.call("tools/call", {"name": "tally_stats", "arguments": {
            "data": [str(10**9 + i) for i in range(10)]}})
        payload = json.loads(r["result"]["content"][0]["text"])
        check("stats var == 33/4", payload["var"]["exact"] == "33/4", payload)

        r = c.call("tools/call", {"name": "tally_compute", "arguments": {
            "expression": {"op": "div", "args": [1, 0]}}})
        check("tool error -> isError", r["result"].get("isError") is True, r)
        check("error text mentions non-finite",
              "non-finite" in r["result"]["content"][0]["text"], r)

        r = c.call("no/such/method")
        check("unknown method -> -32601",
              r.get("error", {}).get("code") == -32601, r)

        r = c.call("tools/call", {"name": "tally_nope", "arguments": {}})
        check("unknown tool -> -32602", r.get("error", {}).get("code") == -32602, r)

        r = c.call("tools/call", {"name": "tally_compute", "arguments": {
            "expression": {"op": "factorial", "arg": 200}}})
        payload = json.loads(r["result"]["content"][0]["text"])
        check("200! over MCP (375 digits)", len(payload["exact"]) == 375, len(payload["exact"]))

        r = c.call("tools/call", {"name": "tally_compute", "arguments": {
            "expression": json.dumps({"op": "sqrt", "arg": 2, "digits": 40})}})
        payload = json.loads(r["result"]["content"][0]["text"])
        check("expression-as-JSON-string accepted",
              payload["decimal"].startswith("1.4142135623"), payload["decimal"])
    finally:
        c.close()


def main():
    unit_tests()
    mcp_tests()
    print(f"\n{'=' * 50}\n{PASS} passed, {FAIL} failed")
    if FAILURES:
        print("failures:", *FAILURES, sep="\n  - ")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
