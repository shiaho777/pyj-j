"""Bridge overhead micro-benchmark: pyj roundtrip vs pure numpy."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyj

def bench(label, fn, iters=2000):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    dt = (time.perf_counter() - t0) / iters
    print(f"{label:40s} {dt*1e6:10.2f} us/op")
    return dt

print("=== small vector (1k float64) ===")
v = np.arange(1000, dtype=np.float64)
pyj.set("v1k", v)
d_np   = bench("numpy: v.sum()", lambda: v.sum())
d_set  = bench("pyj.set (numpy->J 3!:1)", lambda: pyj.set("v1k", v), 2000)
d_jsum = bench("pyj do J sum (set+do+get)", lambda: (pyj.set("v1k", v), pyj.do("s=: +/ v1k"), pyj.get("s")), 500)

print()
print("=== large vector (10M float64, 80MB) ===")
big = np.arange(10_000_000, dtype=np.float64)
pyj.set("big", big)
# one-shot timings (op is big enough to time singly)
t0=time.perf_counter(); pyj.set("big", big); t_set=(time.perf_counter()-t0)*1e3
t0=time.perf_counter(); pyj.do("sb=: +/ big"); t_do=(time.perf_counter()-t0)*1e3
t0=time.perf_counter(); r=pyj.get("sb"); t_get=(time.perf_counter()-t0)*1e3
t0=time.perf_counter(); bsum=big.sum(); t_np=(time.perf_counter()-t0)*1e3
print(f"pyj.set 10M            {t_set:10.2f} ms")
print(f"pyj.do J sum 10M       {t_do:10.2f} ms")
print(f"pyj.get scalar         {t_get:10.2f} ms")
print(f"numpy sum 10M          {t_np:10.2f} ms")
print(f"J sum correct: {float(r)} == {bsum}")
print()
print("=== matmul 1000x1000 ===")
A = np.random.rand(1000,1000); B = np.random.rand(1000,1000)
pyj.set("A", A); pyj.set("B", B)
pyj.do("mp=: +/ . *")
t0=time.perf_counter(); C=pyj.get("C2") if False else None; pyj.set("A",A); pyj.set("B",B); pyj.do("C1=: A mp B"); t_j=(time.perf_counter()-t0)*1e3
t0=time.perf_counter(); r1=pyj.get("C1"); t_jget=(time.perf_counter()-t0)*1e3
t0=time.perf_counter(); C2=A@B; t_np=(time.perf_counter()-t0)*1e3
print(f"pyj set+set+do (J matmul) {t_j:10.2f} ms")
print(f"pyj.get C1                {t_jget:10.2f} ms")
print(f"numpy @                   {t_np:10.2f} ms")
print(f"max diff: {np.abs(r1-C2).max() if r1.shape==C2.shape else 'shape mismatch'}")
