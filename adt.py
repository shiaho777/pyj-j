"""adt.py — Tensor factory API over pyj + ad.ijs.

Wraps the node-id plumbing behind a numpy-like interface:

    from adt import Graph
    g = Graph()
    X = g.tensor(Xa, "X")        # data leaf
    W = g.tensor(Wa, "W")        # parameter leaf
    L = ((X @ W).tanh()).sum()   # build graph with operators
    g.back(L)                    # backprop through the tape
    print(W.grad)                # numpy gradient

Design notes:
- Node ids live in the J tape (ad.ijs). A Tensor wraps (id, graph).
- Leaf values cross via pyj.set + ADSETP (value-passing; J-side eval is
  never used, so variable shadowing cannot corrupt leaf capture).
- Every op emits one sentence per node; the tape rebuilds each forward pass
  (correctness first — replay/fusion is a later step).
"""
import numpy as np
import pyj


def _jdo(sentence):
    rc, out = pyj.do(sentence)
    import os
    if os.environ.get('PYJ_DEBUG'):
        print('J>', sentence, '->', rc)
    if rc != 0:
        raise RuntimeError(f"J error {rc} in: {sentence}\n{''.join(out)}")


class Graph:
    def __init__(self, ad_path=None):
        here = ad_path or __file__.rsplit("/", 1)[0]
        _jdo(f"0!:0 <'{here}/ad.ijs'")
        self._leaves = []          # list[Tensor] in registration order
        self._last_loss = None

    # ---------- leaf management ----------
    def tensor(self, value, name):
        if self._leaves and getattr(self, "_synced", False):
            raise RuntimeError("all leaves must be created before building the graph")
        t = Tensor(-1, self, name=name, value=np.asarray(value))
        self._leaves.append(t)
        return t

    def _sync_leaves(self):
        """Register all leaves (once) and push their values. Leaves must all be
        created before the first op — adding leaves mid-graph is unsupported."""
        for t in self._leaves:
            pyj.set(f"{t.name}_v", t.value)
        pairs = " , ".join(f"(<'{t.name}_v';{t.name}_v)" for t in self._leaves)
        _jdo(f"ADSETP {pairs}")
        for i, t in enumerate(self._leaves):
            t.id = i
            t.order = i
        self._synced = True

    def _unop(self, verb, a):
        if not getattr(self, "_synced", False):
            self._sync_leaves()
        _jdo(f"adnode=: {verb} {a.id}")
        return Tensor(self._nodeid(), self)

    def _binop(self, verb, a, b):
        if not getattr(self, "_synced", False):
            self._sync_leaves()
        _jdo(f"adnode=: {a.id} {verb} {b.id}")
        return Tensor(self._nodeid(), self)

    def _nodeid(self):
        _jdo("adnid=: <:ADN")
        return int(pyj.get("adnid"))

    def _const(self, v):
        # constant leaf: appended to tape via nleaf; not registered in ADLEAVES output order
        if not getattr(self, "_synced", False):
            self._sync_leaves()
        self._const_n = getattr(self, "_const_n", 0) + 1
        name = f"adc{self._const_n}"
        pyj.set(name, np.asarray(v, dtype=np.float64))
        _jdo(f"adcval=: {name}")
        _jdo("adnode=: nleaf adcval")
        return Tensor(self._nodeid(), self)

    def _iconst(self, v):
        # INT constant leaf (index vectors, take/drop specs): same as _const
        # but INT dtype so J rank ops/indexing see integers, not floats
        if not getattr(self, "_synced", False):
            self._sync_leaves()
        self._iconst_n = getattr(self, "_iconst_n", 0) + 1
        name = f"adci{self._iconst_n}"
        pyj.set(name, np.asarray(v, dtype=np.int64))
        _jdo(f"adcival=: {name}")
        _jdo("adnode=: nleaf adcival")
        return Tensor(self._nodeid(), self)

    def _relup(self, a):
        if not getattr(self, "_synced", False):
            self._sync_leaves()
        _jdo("adnode=: nleaf 0")
        zid = self._nodeid()
        _jdo(f"adnode=: {a.id} nemax {zid}")
        return Tensor(self._nodeid(), self)

    # ---------- forward/backward ----------
    def back(self, loss):
        _jdo(f"gg=: ADGET {loss.id}")
        for t in self._leaves:
            t._grad = None   # invalidate cached grads after each backprop


class Tensor:
    def __init__(self, id, graph, name=None, value=None):
        self.id = id
        self.g = graph
        self.name = name
        self.value = value
        self._grad = None
        self.order = None   # leaf order within ADSETP; set by Graph
        self.shape = None if value is None else value.shape

    def _shape(self):
        if self.shape is not None:
            return self.shape
        _jdo("adshp=: $ nval " + str(self.id))
        import pyj as _p
        _jdo("adshpn=: #adshp")
        n = int(_p.get("adshpn"))
        if n == 0:
            self.shape = ()
        else:
            dims = []
            for i in range(n):
                _jdo(f"adshpi=: {i}{{adshp")
                dims.append(int(_p.get("adshpi")))
            self.shape = tuple(dims)
        return self.shape

    def value_now(self):
        """fetch current node value from J (works for intermediate nodes)"""
        _jdo(f"advalx=: nval {self.id}")
        return pyj.get("advalx")

    @property
    def grad(self):
        if self._grad is None:
            if self.order is None:
                raise RuntimeError("grad only defined for leaves")
            _jdo(f"adggx=: >{self.order}{{gg")
            self._grad = pyj.get("adggx")
        return self._grad

    # ---- ops ----
    def __add__(self, other):
        if isinstance(other, Tensor):
            sa, sb = self._shape(), other._shape()
            if len(sa) == 2 and len(sb) == 1 and sa[1] == sb[0]:
                return self.g._binop("nbadd", self, other)
            return self.g._binop("nadd", self, other)
        return self + self.g._const(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return self.g._binop("nsub", self, other)
        return self - self.g._const(other)

    def __rsub__(self, other):
        return self.g._const(other) - self

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return self.g._binop("nmul", self, other)
        return self * self.g._const(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return self.g._binop("ndiv", self, other)
        return self * self.g._const(1.0 / float(other))

    def __rtruediv__(self, other):
        return self.g._const(other) / self

    def __matmul__(self, other):
        return self.g._binop("nmp", self, other)

    def __neg__(self):
        return self.g._unop("nneg", self)

    def tanh(self):      return self.g._unop("ntanh", self)

    # ---- rank-1 (per-row) ops ----
    def _rbinop(self, verb, a, b):
        return self.g._binop(verb, a, b)

    def rsub1(self, other):
        """self -"(1 0) other (other: per-row vector tensor)"""
        return self.g._binop("nrsub1", self, other)

    def rdiv1(self, other):
        """self %"(1 0) other"""
        return self.g._binop("nrdiv1", self, other)

    def rsum1(self):
        return self.g._unop("nrsum1", self)

    def rmax1(self):
        return self.g._unop("nrmax1", self)

    def softmax(self):
        """row-wise, max-stabilized. self: (N,C) tensor. Uses rank-1 ops
        (rsub1/rsum1/rdiv1) whose VJPs sum over the right axis."""
        m = self.rmax1()
        e = (self.rsub1(m)).exp()
        return e.rdiv1(e.rsum1())

    def reshape(self, *dims):
        """reshape self to dims (tuple). VJP: reshape grad back."""
        import numpy as _np
        c = self.g._const(_np.array(dims, dtype=np.int64))
        return self.g._binop("nreshape", self, c)

    def take(self, n):
        """n{."1 — truncate trailing axis to n (VJP: append n zero columns)"""
        c = self.g._iconst([int(n)])
        return self.g._binop("ntake", self, c)

    def drop(self, n):
        """n}."1 — drop n cells from trailing axis (VJP: prepend n zero columns)"""
        c = self.g._iconst([int(n)])
        return self.g._binop("ndrop", self, c)

    def gather(self, idx):
        """(,idx){y — index the leading axis (embedding lookup); VJP: scatter-accumulate"""
        import numpy as _np
        c = self.g._iconst(_np.asarray(idx, dtype=np.int64).ravel())
        return self.g._binop("ngather", self, c)

    def relu(self):      return self.g._relup(self)
    def exp(self):       return self.g._unop("nexp", self)
    def log(self):       return self.g._unop("nlog", self)
    def square(self):    return self.g._unop("nsq", self)
    def sum(self):       return self.g._unop("nsum", self)
    def max(self):       return self.g._unop("nmax", self)
    def transpose(self): return self.g._unop("ntr", self)
    def T(self):         return self.transpose()
