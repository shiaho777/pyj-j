import sys, os
os.environ.setdefault("PYJ_LIBPATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "jlibrary", "bin"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyj

def jq(s):
    rc, out = pyj.do(s)
    assert rc == 0, (s, rc)
    return out[-1].strip() if out else ""

pyj.do("newton2 =: 3 : '-: y + 2 % y'")
s = jq('": ((newton2^:8) (1x))')
n, d = s.split("r"); n, d = int(n), int(d)
print("sqrt2 Pell check: n^2 - 2*d^2 =", n*n - 2*d*d)
print("sqrt2 value:", str(n//d) + "." + str((n * 10**29) // d)[-29:])
print("2.3*100 exact =", jq('": 100 * 23r10'))
print("2^53+1 exact  =", jq('": 1x + 2x^53'))
print("hilbert5 inv (exact ints) =", jq('": ,. %. % 1 + +/~ i.5x'))
