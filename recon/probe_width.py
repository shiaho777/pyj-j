import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyj

def j(s):
    rc, out = pyj.do(s)
    assert rc == 0, (s, rc)
    return [l.rstrip("\r\n") for l in out]

# Print pure long lines of various lengths; find where the session wraps.
for n in [195, 196, 197, 198, 199, 200, 201, 255, 256, 257, 258, 259, 260, 261, 300]:
    j(f"tally_w =: {n} # 'x'")
    out = j("tally_w")
    lens = [len(l) for l in out]
    print(f"n={n:3d}  -> {len(out)} line(s), lens={lens[:4]}")

# same but for a value that is a big number (digits)
print()
j("tally_b =: ! 150x")
out = j("tally_b")
print("!150x printed as:", len(out), "line(s), lens:", [len(l) for l in out])
print("first line head:", out[0][:40])
