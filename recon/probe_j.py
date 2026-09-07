import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyj

def probe(s, label=""):
    rc, out = pyj.do(s)
    lines = [l.rstrip("\n") for l in out]
    print(f"[{label}] rc={rc} :: {s!r}")
    for i, l in enumerate(lines):
        print(f"    L{i}: ({len(l)} chars) {l[:100]}")
    print()

probe('tally_v =: 5x', 'assign-no-output')
probe('": $ tally_v', 'scalar-shape')
probe('": , tally_v', 'scalar-ravel')
probe('(2x) % (0x)', 'div-zero')
A = '% 1 + +/~ i.4x'
probe(f'b =. 4 $ 1x', 'setup-b')
probe(f'b %. {A}', 'solve-rational')
probe('! 200x', 'big-factorial-width')
probe('12x +. 18x', 'gcd')
probe('12x *. 18x', 'lcm')
probe('7x | 23x', 'mod-direction')
probe('23x | 7x', 'mod-direction2')
probe('<. _7r2', 'floor-neg')
probe('>. _7r2', 'ceil-neg')
probe('(3x) ^ (_2x)', 'neg-exp')
probe('(1r3) ^ (2x)', 'rat-exp')
probe('", (2 2 $ 1r2 5r2 7r2 9r2)', 'matrix-ravel')
probe('; (,&(10{a.)@":)"0 , (2 2 $ 1r2 5r2 7r2 9r2)', 'per-line-ravel')
probe('! 150x', 'width-263')
probe('x: %: (2x)', 'sqrt-seed')
probe('((((2x) + *:) % +:)^:8 (x: %: (2x)))', 'sqrt-fork-inline')
probe('* (2x) - (2x)', 'star?')
probe('(0x) - (1r2)', 'neg-rat-print')
probe('* (0x) - (1r2)', 'sign')
probe('| (0x) - (1r2)', 'abs')
probe('(1r3) < (1r2)', 'cmp')
probe('(3r10) = (10 %: 1x)', 'eq-mixed')
