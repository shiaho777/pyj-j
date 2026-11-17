"""
tally CLI -- exact arithmetic from the terminal.

  tally 0.1 + 0.2            -> 3/10
  tally "sqrt(2)" -d 50      -> certified convergent, 50 digits
  tally "pi()" -d 100        -> 100-digit pi
  tally --json '{"op":"det","m":[["1/2","1/3"],["1/4","1/5"]]}'
  tally --verify "1/3+1/6" "1/2"
  tally -i                   -> interactive REPL

Every result is exact. The decimal expansion is truncated, not rounded.
"""
import argparse
import json
import sys


def _repl(digits):
    """Interactive exact-arithmetic REPL. `ans` holds the last exact value."""
    import tally
    print(f"tally {tally.__version__} — exact arithmetic REPL "
          f"(digits={digits}). Type 'help' for commands, 'quit' to exit.")
    ans = "0"
    while True:
        try:
            line = input("tally> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("quit", "exit", "q"):
            break
        if line == "help":
            print("  <expr>          evaluate exactly (e.g. 0.1+0.2, sqrt(2))")
            print("  ans             the previous exact result")
            print("  digits <n>      set expansion digits (1..1000)")
            print("  cert            show the last error certificate")
            print("  quit            exit")
            continue
        if line.startswith("digits "):
            try:
                digits = max(1, min(1000, int(line.split()[1])))
                print(f"digits = {digits}")
            except (ValueError, IndexError):
                print("usage: digits <n>")
            continue
        try:
            # substitute `ans` with the previous exact value
            expr = line.replace("ans", f"({ans})")
            ast = tally._expr_to_ast(expr)
            r = tally.evaluate(ast, digits=digits)
            exact = r["exact"]
            if isinstance(exact, str):
                ans = exact
                print(f"  {exact}    ({r['type']})")
                if r["type"] != "boolean" and r["decimal"] != exact:
                    print(f"  = {r['decimal']}")
                if r.get("certified"):
                    print(f"  certified: |true - exact| <= {r['error_bound_decimal']}")
            else:
                print(json.dumps(exact, ensure_ascii=False))
        except tally.TallyError as e:
            print(f"  error: {e}")
        except Exception as e:
            print(f"  error: {e}")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="tally",
        description="Exact-arithmetic calculator. 0.1 means 1/10, never a float.")
    p.add_argument("expr", nargs="*", help="infix expression, e.g. 0.1 + 0.2")
    p.add_argument("-d", "--digits", type=int, default=30,
                   help="decimal expansion digits (default 30, max 1000)")
    p.add_argument("-i", "--interactive", action="store_true",
                   help="interactive REPL")
    p.add_argument("--json", metavar="AST", help="evaluate a JSON AST directly")
    p.add_argument("--verify", nargs=2, metavar=("EXPR", "CLAIM"),
                   help="recompute EXPR exactly and grade CLAIM")
    p.add_argument("--exact-only", action="store_true",
                   help="print only the exact p/q result")
    args = p.parse_args(argv)

    import tally

    if args.interactive:
        _repl(args.digits)
        return 0

    try:
        if args.verify:
            expr, claim = args.verify
            r = tally.verify(tally._expr_to_ast(expr), claim, digits=args.digits)
            if args.exact_only:
                print(r.get("exact", ""))
            else:
                print(json.dumps(r, indent=2, ensure_ascii=False))
            return 0 if r.get("match") else 1

        if args.json:
            ast = tally.loads_exact(args.json)
        elif args.expr:
            ast = tally._expr_to_ast(" ".join(args.expr))
        else:
            p.print_help()
            return 2

        r = tally.evaluate(ast, digits=args.digits)
        if args.exact_only:
            print(r["exact"])
        else:
            exact = r["exact"]
            dec = r["decimal"]
            typ = r["type"]
            if isinstance(exact, str):
                print(f"{exact}    ({typ})")
                if typ != "boolean" and dec != exact:
                    print(f"= {dec}    (truncated, not rounded)")
            else:
                print(json.dumps(r, indent=2, ensure_ascii=False))
        return 0
    except tally.TallyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())