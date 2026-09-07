import sys, os
from fractions import Fraction
os.environ.setdefault("PYJ_LIBPATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "jlibrary", "bin"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyj

def jq(s):
    rc, out = pyj.do(s)
    assert rc == 0, (s, rc, out)
    return out[-1].strip() if out else ""

pyj.do("newton2 =: 3 : '-: y + 2 % y'")
s = jq('": ((newton2^:8) (1x))')
n, d = s.split("r"); n, d = int(n), int(d)
print("sqrt2 Pell check: n^2 - 2*d^2 =", n*n - 2*d*d)

# exact compound interest vs float64 claim
E = jq('digits5 =: 3 : \'": <. y * 10x^5\'')
exact_cents = jq('": <. (100 * 100000x) * (1205r1200)^360')
print("exact cents (floor):", exact_cents)
big = jq('": (100000x) * (1205r1200)^360')
print("exact rational p/q lens:", len(big.split('r')))

# debug first_wrong_digit anomaly from earlier: compare truncations
from fractions import Fraction
exact05 = Fraction(100000) * (Fraction(241,240)**360)
claim = 100000*(1+0.05/12)**360
c = Fraction(claim)
for dd in range(6, 16):
    ie = int(exact05 * 10**dd); ic = int(c * 10**dd)
    print(dd, ie == ic, ie, ic)
print("exact05 ~", float(exact05))
print("claim      ~", repr(claim))
print("claim exact binary ~", float(c), "{:.20f}".format(float(c)))
