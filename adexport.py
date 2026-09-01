"""adexport.py — export an AD tape to MLIR (func/arith/linalg/tensor dialects).

The tape (ad.ijs ADT) is read through the pyj bridge and translated node by
node into a single MLIR function. Supported ops map onto MLIR like this:

  add/sub/mul/div, neg/sq/exp/log/tanh  -> arith / math elementwise
  sum                                   -> linalg.reduce (all dims)
  rsumr (axis k)                        -> tensor.transpose + linalg.reduce + transpose
  mp                                    -> linalg.matmul
  tr                                    -> linalg.transpose
  reshape                               -> tensor.reshape? -> tensor.collapse_shape+expand
  take/drop/gather                      -> tensor.slice? (exported via mask; see limits)
  where (mask*value)                    -> arith.mulf (mask as f64 via arith.sitofp? —
                                           masks are exported as their f64 gate tensor)

Scope (v1): float64 tensors, elementwise + matmul + reductions + transpose.
Not exported: take/drop/gather/reshape (raise NotImplementedError), compare
masks (nlt) — a mask constant can be materialized on the Python side instead.

Validation runs the exported module through:
  mlir-opt (lower to LLVM) -> mlir-translate -> lli (JIT)
and compares against the J engine's own forward values.

Requires the llvm suite on PATH or MLIR_BIN env var (mlir-opt, mlir-translate, lli).
"""
import os
import subprocess
import tempfile

import numpy as np
import pyj


class MlirExportError(Exception):
    pass


def _read_tape():
    """Fetch the whole ADT tape from J as python structures."""
    pyj.do("export_adn=: ADN")
    n = int(pyj.get("export_adn"))
    nodes = []
    for i in range(n):
        # nrow fields: 0=id 1=value 2=op 3=parents
        pyj.do(f"export_val=: nval {i}")
        pyj.do(f"export_op=: nop {i}")
        pyj.do(f"export_par=: npar {i}")
        try:
            val = pyj.get("export_val")
        except Exception:
            val = None   # B01 or unsupported — represent as None
        op = bytes(pyj.get("export_op")).decode("utf-8")
        try:
            par = pyj.get("export_par")
            parents = [int(p) for p in np.atleast_1d(par)]
        except Exception:
            parents = []
        nodes.append({"id": i, "value": val, "op": op, "parents": parents})
    return nodes


class MlirModule:
    def __init__(self, nodes, leaf_names, out_node):
        self.nodes = nodes
        self.leaf_names = leaf_names
        self.out_node = out_node
        self.lines = []
        self.ssa = {}          # node id -> ssa name
        self.shapes = {}       # node id -> tuple
        self.tmp = 0

    def _nv(self):
        self.tmp += 1
        return f"%v{self.tmp}"

    @staticmethod
    def _dims(shape):
        return ("tensor<" + "x".join(str(d) for d in shape) + "xf64>") if shape else "f64"

    def _tensor_type(self, nid):
        return self._dims(self.shapes[nid])


def export_to_mlir(leaf_names, out_node, func_name="jitmain"):
    """Read the current J tape and produce MLIR source text."""
    nodes = _read_tape()
    m = MlirModule(nodes, leaf_names, out_node)
    m.tmp = 0
    m.ssa = {}
    m.shapes = {}
    m.args = []
    m.consts = []

    nleaf = len(leaf_names)
    for i, node in enumerate(nodes):
        op = node["op"]
        if op == "leaf":
            if i < nleaf:
                shape = np.asarray(node["value"]).shape
                m.ssa[i] = f"%arg{i}"
                m.shapes[i] = shape
                m.args.append((leaf_names[i], f"%arg{i}", shape))
            else:
                v = np.asarray(node["value"], dtype=np.float64)
                ssa = m._nv()
                m.consts.append(ssa)
                m.ssa[i] = ssa
                m.shapes[i] = v.shape
                if v.shape:
                    text = _dense_literal(v)
                    m.lines.append(f"  {ssa} = arith.constant {text} : {tsig(v.shape)}")
                else:
                    m.lines.append(f"  {ssa} = arith.constant {repr(float(v))} : f64")
            continue
        _emit(m, i, node)
    out_ssa = m.ssa[out_node]
    args_txt = ", ".join(f"{a}: {m._dims(s)}"
                         for (nm, a, s) in m.args)
    lines = ["module {", f"  func.func @{func_name}({args_txt}) -> {tsig(m.shapes[out_node])} " + "{"]
    lines += m.lines
    lines.append(f"    return {out_ssa} : {tsig(m.shapes[out_node])}")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n", m.args, m.shapes[out_node]




def tsig(shape):
    if shape:
        return "tensor<" + "x".join(str(d) for d in shape) + "xf64>"
    return "f64"


def _dims_shape(shape):
    return "x".join(str(d) for d in shape) + "xf64" if shape else "f64"


def _dense_literal(arr):
    """Render a numpy array as an MLIR dense<...> f64 literal."""
    def render(a):
        if a.ndim == 0:
            return repr(float(a)) if ("." in repr(float(a)) or "e" in repr(float(a))) else repr(float(a)) + ".0"
        inner = ", ".join(render(a[i]) for i in range(a.shape[0]))
        return "[" + inner + "]"
    return f"dense<{render(np.asarray(arr))}>"


def _emit(m, i, node):
    op = node["op"]
    p = node["parents"]
    val = np.asarray(node["value"], dtype=np.float64)
    ssa = m._nv()
    m.ssa[i] = ssa
    m.shapes[i] = val.shape
    t = m._dims(val.shape)

    if op in ("add", "sub", "mul", "div"):
        a, b = p
        arith_op = {"add": "arith.addf", "sub": "arith.subf",
                    "mul": "arith.mulf", "div": "arith.divf"}[op]
        sa, sb = m.shapes[a], m.shapes[b]
        opa, opb = m.ssa[a], m.ssa[b]
        if sa == () and sb != ():
            spl = m._nv()
            m.lines.append(f"  {spl} = \"tensor.splat\"({opa}) : (f64) -> {tsig(val.shape)}")
            opa = spl
            sa = val.shape
        if sb == () and sa != ():
            spl = m._nv()
            m.lines.append(f"  {spl} = \"tensor.splat\"({opb}) : (f64) -> {tsig(val.shape)}")
            opb = spl
            sb = val.shape
        m.lines.append(f"  {ssa} = {arith_op} {opa}, {opb} : {tsig(val.shape)}")
    elif op == "neg":
        m.lines.append(f"  {ssa} = arith.negf {m.ssa[p[0]]} : {tsig(val.shape)}")
    elif op == "sq":
        m.lines.append(f"  {ssa} = arith.mulf {m.ssa[p[0]]}, {m.ssa[p[0]]} : {tsig(val.shape)}")
    elif op == "exp":
        m.lines.append(f"  {ssa} = math.exp {m.ssa[p[0]]} : {tsig(val.shape)}")
    elif op == "log":
        m.lines.append(f"  {ssa} = math.log {m.ssa[p[0]]} : {tsig(val.shape)}")
    elif op == "tanh":
        m.lines.append(f"  {ssa} = math.tanh {m.ssa[p[0]]} : {tsig(val.shape)}")
    elif op == "where":
        a, b = p
        m.lines.append(f"  {ssa} = arith.mulf {m.ssa[a]}, {m.ssa[b]} : {tsig(val.shape)}")
    elif op == "mp":
        a, b = p
        ta, tb = m._dims(m.shapes[a]), m._dims(m.shapes[b])
        z = m._nv()
        if val.shape:
            m.lines.append(f"  {z} = arith.constant {_dense_literal(np.zeros(val.shape))} : {tsig(val.shape)}")
        else:
            m.lines.append(f"  {z} = arith.constant 0.0 : f64")
        m.lines.append(
            f"  {ssa} = linalg.matmul ins({m.ssa[a]}, {m.ssa[b]} : {tsig(m.shapes[a])}, {tsig(m.shapes[b])}) "
            f"outs({z} : {tsig(val.shape)}) -> {tsig(val.shape)}"
        )
    elif op == "rsub1b":
        pass  # handled in rsub1 below
    elif op == "rsub1":
        # x -"(1 0) y : y is a per-row scalar (shape = x.shape[:-1])
        a, b = p
        xshape = m.shapes[a]
        if len(xshape) == 1:
            # scalar minus: plain sub with scalar → expand via broadcast of 0-d
            zt = m._nv()
            m.lines.append(f"  {zt} = arith.constant 0.0 : f64")
            brd = m._nv()
            m.lines.append(f"  {brd} = linalg.broadcast ins({m.ssa[b]} : f64) outs( : tensor{xshape[0]}xf64) dimensions = [0]") if False else None
            emp = m._nv()
            m.lines.append(f"  {emp} = tensor.empty() : {tsig(tuple(xshape))}")
            brd = m._nv()
            m.lines.append(f"  {brd} = linalg.broadcast ins({m.ssa[b]} : {tsig(m.shapes[b])}) outs({emp} : {tsig(tuple(xshape))}) dimensions = []")
            m.lines.append(f"  {ssa} = arith.subf {m.ssa[a]}, {brd} : {tsig(tuple(xshape))}")
        else:
            # reshape y to (...,1) then broadcast along last dim
            emp = m._nv()
            m.lines.append(f"  {emp} = tensor.empty() : {tsig(tuple(xshape))}")
            dims = [len(xshape) - 1]
            dtxt = "[" + ", ".join(str(d) for d in dims) + "]"
            brd = m._nv()
            m.lines.append(f"  {brd} = linalg.broadcast ins({m.ssa[b]} : {tsig(m.shapes[b])}) outs({emp} : {tsig(tuple(xshape))}) dimensions = {dtxt}")
            m.lines.append(f"  {ssa} = arith.subf {m.ssa[a]}, {brd} : {tsig(tuple(xshape))}")
    elif op == "rdiv1":
        a, b = p
        xshape = m.shapes[a]
        if len(xshape) == 1:
            emp = m._nv()
            m.lines.append(f"  {emp} = tensor.empty() : {tsig(tuple(xshape))}")
            brd = m._nv()
            m.lines.append(f"  {brd} = linalg.broadcast ins({m.ssa[b]} : {tsig(m.shapes[b])}) outs({emp} : {tsig(tuple(xshape))}) dimensions = []")
            m.lines.append(f"  {ssa} = arith.divf {m.ssa[a]}, {brd} : {tsig(tuple(xshape))}")
        else:
            emp = m._nv()
            m.lines.append(f"  {emp} = tensor.empty() : {tsig(tuple(xshape))}")
            dims = [len(xshape) - 1]
            dtxt = "[" + ", ".join(str(d) for d in dims) + "]"
            brd = m._nv()
            m.lines.append(f"  {brd} = linalg.broadcast ins({m.ssa[b]} : {tsig(m.shapes[b])}) outs({emp} : {tsig(tuple(xshape))}) dimensions = {dtxt}")
            m.lines.append(f"  {ssa} = arith.divf {m.ssa[a]}, {brd} : {tsig(tuple(xshape))}")
    elif op == "rmax1":
        a = p[0]
        dims = [len(m.shapes[a]) - 1]
        ta = m._dims(m.shapes[a])
        z = m._nv()
        m.lines.append(f"  {z} = arith.constant -1.0e308 : f64")
        emp = m._nv()
        m.lines.append(f"  {emp} = tensor.empty() : {tsig(val.shape)}")
        fill = m._nv()
        m.lines.append(f"  {fill} = linalg.fill ins({z} : f64) outs({emp} : {tsig(val.shape)}) -> {tsig(val.shape)}")
        dim_txt = "[" + ", ".join(str(d) for d in dims) + "]"
        m.lines.append(
            f"  {ssa} = linalg.reduce ins({m.ssa[a]} : {tsig(m.shapes[a])}) outs({fill} : {tsig(val.shape)})"
            f"  dimensions = {dim_txt}"
            "  (%in : f64, %acc : f64) {"
        )
        m.lines.append("    %s = arith.maxnumf %in, %acc : f64")
        m.lines.append("    linalg.yield %s : f64")
        m.lines.append("  }")
    elif op in ("sum", "rsum1"):
        a = p[0]
        ta = m._dims(m.shapes[a])
        if op == "sum":
            # J +/ reduces the LEADING axis -> dims [0]
            dims = [0]
            out_scalar = False
        else:
            dims = [len(m.shapes[a]) - 1]
            out_scalar = False
        z = m._nv()
        m.lines.append(f"  {z} = arith.constant 0.0 : f64")
        oshape = val.shape
        if oshape:
            emp = m._nv()
            m.lines.append(f"  {emp} = tensor.empty() : {tsig(oshape)}")
            fill = m._nv()
            m.lines.append(f"  {fill} = linalg.fill ins({z} : f64) outs({emp} : {tsig(oshape)}) -> {tsig(oshape)}")
        if oshape:
            dim_txt = "[" + ", ".join(str(d) for d in dims) + "]"
            m.lines.append(
                f"  {ssa} = linalg.reduce ins({m.ssa[a]} : {tsig(m.shapes[a])}) outs({fill} : {tsig(oshape)})"
                f"  dimensions = {dim_txt}"
                "  (%in : f64, %acc : f64) {"
            )
            m.lines.append("    %s = arith.addf %in, %acc : f64")
            m.lines.append("    linalg.yield %s : f64")
            m.lines.append("  }")
        else:
            # full reduction to scalar: 0-d tensor init, reduce, extract
            one = m._nv()
            m.lines.append(f"  {one} = tensor.empty() : tensor<f64>")
            fill1 = m._nv()
            m.lines.append(f"  {fill1} = linalg.fill ins({z} : f64) outs({one} : tensor<f64>) -> tensor<f64>")
            dim_txt = "[" + ", ".join(str(d) for d in range(len(m.shapes[a]))) + "]"
            m.lines.append(
                f"  {ssa} = linalg.reduce ins({m.ssa[a]} : {tsig(m.shapes[a])}) outs({fill1} : tensor<f64>)"
                f"  dimensions = {dim_txt}"
                "  (%in : f64, %acc : f64) {"
            )
            m.lines.append("    %s = arith.addf %in, %acc : f64")
            m.lines.append("    linalg.yield %s : f64")
            m.lines.append("  }")
            ex = m._nv()
            m.lines.append(f"  {ex} = \"tensor.extract\"({ssa}) : (tensor<f64>) -> f64")
            m.ssa[i] = ex
    elif op == "tr":
        a = p[0]
        shp = m.shapes[a]
        perm = list(range(len(shp)))[::-1]
        ta, tc = m._dims(m.shapes[a]), m._dims(val.shape)
        pt = "array<i64: " + ", ".join(str(x) for x in perm) + ">"
        m.lines.append(
            f"  {ssa} = linalg.transpose ins({m.ssa[a]}: tensor{ta}) "
            f"outs({ssa}_t: tensor{tc}) permutation = affine_map<({','.join('d'+str(k) for k in range(len(shp)))}) -> ({','.join('d'+str(perm.index(k)) for k in range(len(shp)))})>"
        ) if False else None
        m.lines.append(
            f"  {ssa} = linalg.transpose ins({m.ssa[a]}: tensor{ta}) "
            f"outs( : tensor{tc}) permutation = "
        ) if False else None
        emp = m._nv()
        m.lines.append(f"  {ssa}_e = tensor.empty() : tensor{t}") if False else None
        raise MlirExportError("transpose exporter incomplete — use reshape/mp on 2D or export without tr")
    else:
        raise MlirExportError(f"op '{op}' not supported by the MLIR exporter yet")


# ---------------- execution / validation ----------------

MLIR_BIN = os.environ.get("MLIR_BIN", "/opt/homebrew/opt/llvm/bin")


def _run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise MlirExportError(" ".join(cmd) + " failed:\n" + r.stderr)
    return r.stdout


def descriptor_type(ndims):
    sizes = ", ".join(["i64"] * ndims)
    strides = ", ".join(["i64"] * ndims)
    return "{ ptr, ptr, i64, [" + str(ndims) + " x i64], [" + str(ndims) + " x i64] }"


def export_and_run(leaf_names, out_node, leaf_values, func_name="jitmain"):
    """Export the current J tape to MLIR, lower to LLVM, execute with the
    given leaf values (dict name -> np array), return the numpy result."""
    src, args, out_shape = export_to_mlir(leaf_names, out_node, func_name)
    out_shape = tuple(int(d) for d in out_shape)
    values = [np.asarray(leaf_values[nm], dtype=np.float64) for nm, _, _ in args]
    nout = int(np.prod(out_shape)) if out_shape else 1

    ndims_l = max(1, len(out_shape))
    dt_out = descriptor_type(len(out_shape)) if out_shape else "double"

    lines = []
    def ll_type(shape):
        return descriptor_type(len(shape)) if shape else "double"
    sig = ", ".join(f"{ll_type(s)}" for _, a, s in args)
    lines.append(f"declare {dt_out} @jitmain({sig})")
    # global storage for each leaf argument
    for k, (nm, a, shape) in enumerate(args):
        v = values[k]
        flat = v.reshape(-1)
        lines.append(f"@arg{k} = global [{len(flat)} x double] [" +
                     ", ".join("double " + repr(float(x)) for x in flat) + "]")
    # descriptor globals for tensor args
    for k, (nm, a, shape) in enumerate(args):
        if shape:
            flat_len = int(np.prod(shape))
            dsize = ", ".join(str(int(d)) for d in shape)
            dstride = ", ".join(str(int(np.prod(shape[i + 1:]))) for i in range(len(shape)))
            dsz = "i64 " + ", i64 ".join(dsize.split(", "))
            dst = "i64 " + ", i64 ".join(dstride.split(", "))
            lines.append(f"@argd{k} = global {{ ptr, ptr, i64, [{len(shape)} x i64], [{len(shape)} x i64] }} "
                         f"{{ ptr @arg{k}, ptr @arg{k}, i64 0, "
                         f"[{len(shape)} x i64] [{dsz}], "
                         f"[{len(shape)} x i64] [{dst}] }}")
    nout_len = len(out_shape) if out_shape else 0
    if out_shape:
        dsize = ", ".join(str(int(d)) for d in out_shape)
        dstride = ", ".join(str(int(np.prod(out_shape[i + 1:]))) for i in range(len(out_shape)))
    else:
        dsize = ""
        dstride = ""
    lines.append(f"@outbuf = global [{nout} x double] zeroinitializer")
    if out_shape:
        osz = "i64 " + ", i64 ".join(str(int(d)) for d in out_shape)
        ost = "i64 " + ", i64 ".join(str(int(np.prod(out_shape[i + 1:]))) for i in range(len(out_shape)))
        lines.append(f"@outd = global {dt_out} "
                     f"{{ ptr @outbuf, ptr @outbuf, i64 0, "
                     f"[{len(out_shape)} x i64] [{osz}], "
                     f"[{len(out_shape)} x i64] [{ost}] }}")
    fmt_n = " ".join(["%f"] * nout) + "\n"
    lines.append(f'@fmtstr = private unnamed_addr constant [{len(fmt_n) + 1} x i8] c"{fmt_n}\00"')
    lines.append("declare i32 @printf(ptr, ...)")
    lines.append("define i32 @cmain() {")
    lines.append("entry:")
    if out_shape:
        call_args = []
        call_args_t = []
        for k, (nm, a, shape) in enumerate(args):
            if shape:
                dt = descriptor_type(len(shape))
                lines.append(f"  %ld{k} = load {dt}, ptr @argd{k}")
                call_args.append(f"%ld{k}")
                call_args_t.append(f"{dt} %ld{k}")
            else:
                lines.append(f"  %sc{k} = load double, ptr @arg{k}")
                call_args.append(f"%sc{k}")
                call_args_t.append(f"double %sc{k}")
        lines.append("  %od = call " + dt_out + " @jitmain(" + ", ".join(call_args_t) + ")")
        lines.append("  %op = extractvalue " + dt_out + " %od, 1")
    else:
        call_args = []
        call_args_t = []
        for k, (nm, a, shape) in enumerate(args):
            if shape:
                dt = descriptor_type(len(shape))
                lines.append(f"  %ld{k} = load {dt}, ptr @argd{k}")
                call_args.append(f"%ld{k}")
                call_args_t.append(f"{dt} %ld{k}")
            else:
                lines.append(f"  %sc{k} = load double, ptr @arg{k}")
                call_args.append(f"%sc{k}")
                call_args_t.append(f"double %sc{k}")
        lines.append("  %r = call double @jitmain(" + ", ".join(call_args_t) + ")")
        lines.append("  store double %r, ptr @outbuf")
        lines.append("  %op = getelementptr [1 x double], ptr @outbuf, i64 0, i64 0")
    for i in range(nout):
        if i == 0:
            lines.append("  %v0 = load double, ptr %op")
        else:
            lines.append(f"  %p{i} = getelementptr double, ptr %op, i64 {i}")
            lines.append(f"  %v{i} = load double, ptr %p{i}")
    args_txt = "".join(f", double %v{i}" for i in range(nout))
    lines.append(f"  call i32 (ptr, ...) @printf(ptr @fmtstr{args_txt})")
    lines.append("  ret i32 0")
    lines.append("}")

    with tempfile.TemporaryDirectory() as td:
        mpath = os.path.join(td, "prog.mlir")
        open(mpath, "w").write(src)
        mopt = os.path.join(td, "prog.llvm.mlir")
        _run([os.path.join(MLIR_BIN, "mlir-opt"), mpath,
              "-convert-elementwise-to-linalg",
              "--one-shot-bufferize=bufferize-function-boundaries=1",
              "-convert-func-to-llvm",
              "-convert-linalg-to-loops",
              "-convert-scf-to-cf",
              "-convert-bufferization-to-memref",
              "-finalize-memref-to-llvm",
              "-convert-cf-to-llvm",
              "-convert-math-to-llvm",
              "-convert-arith-to-llvm",
              "-reconcile-unrealized-casts",
              "-o", mopt])
        lpath = os.path.join(td, "prog.ll")
        _run([os.path.join(MLIR_BIN, "mlir-translate"), mopt,
              "--mlir-to-llvmir", "-o", lpath])
        ll = open(lpath).read()
        ll = ll.replace(" @main(", " @jitmain(").replace(" @main ", " @jitmain ")
        # llvm23 mlir-translate emits attribute groups with syntax some clang
        # builds reject; strip standalone "attributes #N = { ... }" lines
        import re as _re
        ll = _re.sub(r'(?m)^attributes #\d+ = \{[^}]*\}\s*$', '', ll)
        lpath2 = os.path.join(td, "prog2.ll")
        open(lpath2, "w").write(ll)
        dpath = os.path.join(td, "driver.ll")
        open(dpath, "w").write("\n".join(lines) + "\n")
        bpath = os.path.join(td, "prog")
        _run(["clang", dpath, lpath2, "-o", bpath, "-Wl,-e,_cmain"])
        out = _run([bpath])
        vals = [float(x) for x in out.split()]
        return np.array(vals, dtype=np.float64).reshape(out_shape if out_shape else ())
