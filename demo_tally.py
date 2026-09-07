#!/usr/bin/env python3
"""
demo_tally.py -- the money-checker story, told over a real MCP handshake.

An agent tries to answer: "Balance on $100,000 at 0.05/12 monthly rate
after 360 months?" Its float64 instinct gives an answer that LOOKS right.
Tally recomputes exactly and shows the drift. This is the 30-second video.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def mcp_call(proc, _id, method, params):
    proc.stdin.write(json.dumps(
        {"jsonrpc": "2.0", "id": _id, "method": method, "params": params}
    ).encode() + b"\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise BrokenPipeError("server closed stdout")
        msg = json.loads(line.decode())
        if msg.get("id") == _id:
            return msg


def banner(text):
    print(f"\n{'=' * 64}\n  {text}\n{'=' * 64}")


def trunc(s, n=110):
    """Exact values can be thousands of digits; demo prints a preview."""
    return s if len(s) <= n else s[:n] + f"...  [{len(s):,} chars total]"


def main():
    env = dict(os.environ)
    env.setdefault("DYLD_LIBRARY_PATH", os.path.join(HERE, "jlibrary", "bin"))
    env.setdefault("PYJ_LIBPATH", os.path.join(HERE, "jlibrary", "bin"))
    proc = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "tally_server.py")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
    _id = 0

    def call(method, params):
        nonlocal _id
        _id += 1
        return mcp_call(proc, _id, method, params)

    try:
        r = call("initialize", {"protocolVersion": "2024-11-05",
                                "capabilities": {},
                                "clientInfo": {"name": "demo", "version": "1"}})
        print(f"[handshake] {r['result']['serverInfo']['name']} "
              f"{r['result']['serverInfo']['version']} "
              f"(protocol {r['result']['protocolVersion']})")
        r = call("tools/list", {})
        print(f"[tools] {[t['name'] for t in r['result']['tools']]}")

        banner("SCENE 1: the agent's float64 instinct")
        naive = 100000 * (1 + 0.05 / 12) ** 360
        print(f"agent computes in its head (float64): {naive!r}")
        print("looks plausible. it even rounds to cents:",
              f"${naive:,.2f}")

        banner("SCENE 2: tally_verify grades the claim")
        expr = {"op": "mul", "args": [
            "100000",
            {"op": "pow", "base": "1.00416666666666666666666666666667",
             "exp": 360}]}
        r = call("tools/call", {"name": "tally_verify", "arguments": {
            "expression": expr, "claimed": repr(naive)}})
        v = json.loads(r["result"]["content"][0]["text"])
        print(f"match: {v['match']}")
        m = v["first_mismatch"]
        print(f"truth (exact rational):\n  {trunc(v['exact'])}")
        print(f"truth (decimal, truncated): {v['decimal'][:42]}")
        print(f"abs error in the agent's answer: {trunc(m['abs_error'])}")
        print(f"rel error: {m['rel_error_decimal']}")

        banner("SCENE 3: the branch that mattered")
        r = call("tools/call", {"name": "tally_compute", "arguments": {
            "expression": {"op": "eq",
                           "a": {"op": "sum", "of": ["0.1"] * 10},
                           "b": 1}}})
        c = json.loads(r["result"]["content"][0]["text"])
        print(f"does 0.1 added ten times equal exactly 1?  {c['result']}")
        print("(in float64: False. a reconciliation job would take "
              "the else-branch.)\n")

        banner("SCENE 4: Hilbert_4 inverse -- integer, provably")
        r = call("tools/call", {"name": "tally_compute", "arguments": {
            "expression": {"op": "inv", "m": [
                [f"1/{i+j+1}" for j in range(4)] for i in range(4)]}}})
        c = json.loads(r["result"]["content"][0]["text"])
        for row in c["exact"]:
            print("  " + " ".join(f"{x:>6}" for x in row))
        print("(numpy gives 15.9999999999998-ish floats. Tally gives "
              "integers -- because the truth IS integer.)")
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
