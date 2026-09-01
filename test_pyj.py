"""pyj bridge correctness tests: numpy <-> J via 3!:1."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyj

fails = 0
def check(label, got, want):
    global fails
    ok = np.allclose(got, want) if isinstance(want, np.ndarray) else got == want
    status = "ok  " if ok else "FAIL"
    if not ok:
        fails += 1
        print(f"{status} {label}: got {got!r} want {want!r}")
    else:
        print(f"{status} {label}")

# --- float64 scalar/vector/matrix ---
pyj.set("v", np.array([1.0, 2.0, 3.0, 4.0]))
rc, _ = pyj.do("t=: +/ v")
check("sum via J", pyj.get("t"), np.float64(10.0))

pyj.set("m", np.array([[1.0, 2.0], [3.0, 4.0]]))
pyj.do("mm=: +/ . *")
pyj.set("m2", np.array([[1.0, 1.0], [1.0, 1.0]]))
pyj.do("r=: m mm m2")
check("matmul via J", pyj.get("r"), np.array([[3.0, 3.0], [7.0, 7.0]]))

# rank/shape fidelity
pyj.set("m3", np.arange(24, dtype=np.float64).reshape(2, 3, 4))
pyj.do("m3s=: $ m3")
check("3d shape", pyj.get("m3s"), np.array([2, 3, 4]))
pyj.do('m3t=: +/"1 m3')        # +/"1 sums last axis -> 2x3
check("3d axis2 sum", pyj.get("m3t"), np.arange(24).reshape(2,3,4).sum(2).astype(float))
pyj.do('m3u=: +/"2 m3')        # +/"2 sums middle axis -> 2x4
check("3d axis1 sum", pyj.get("m3u"), np.arange(24).reshape(2,3,4).sum(1).astype(float))

# --- int64 ---
pyj.set("iv", np.array([10, 20, 30], dtype=np.int64))
pyj.do("is=: +/ iv")
check("int sum", pyj.get("is"), np.int64(60))
pyj.do("ip=: */ iv")
check("int product", pyj.get("ip"), np.int64(6000))

# --- bool ---
pyj.set("bv", np.array([True, False, True]))
pyj.do("bs=: +/ bv")
check("bool sum", pyj.get("bs"), np.int64(2))

# --- complex128 ---
pyj.set("cv", np.array([1 + 2j, 3 - 4j]))
pyj.do("cs=: +/ cv")
check("complex sum", pyj.get("cs"), np.complex128(4 - 2j))

# --- rank mechanism: vmap for free ---
pyj.set("rows", np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
pyj.do("tot=: +/\"1 rows")          # row sums via rank
check("rank-1 sums", pyj.get("tot"), np.array([3.0, 7.0, 11.0]))

# --- precision: J extended int through do() side-channel ---
pyj.do("bpv=: !32x")                # 32! in extended precision (beyond int64)
pyj.do("sc=: \": bpv")
sc = pyj.get("sc").tobytes().decode().rstrip('\x00')   # LIT char array
check("extended 32! digits", sc, "263130836933693530167218012160000000")

# --- error path ---
try:
    pyj.get("no_such_name_xyz")
    check("missing name raises", False, True)
except ValueError:
    check("missing name raises", True, True)

# --- unsupported dtype rejected ---
try:
    pyj.set("bad", np.array(["a", "b"]))
    check("str dtype rejected", False, True)
except TypeError:
    check("str dtype rejected", True, True)

print()
print("ALL OK" if fails == 0 else f"{fails} FAILURES")
sys.exit(1 if fails else 0)
