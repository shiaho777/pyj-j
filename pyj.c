/*
 * pyj.c — CPython extension embedding the J engine (libj.dylib/.so)
 *
 * ============ STABLE ABI CONTRACT (see VISION.md invariants) ============
 * The four module functions below are the kernel-facing contract. Downstream
 * non-Python hosts may re-implement them verbatim against their own host
 * language; the semantics they must preserve:
 *
 *   pyj.do(sentence)      -> (rc: int, [output lines]) ; rc 0 = success
 *   pyj.set(name, obj)    -> store a numpy array as J noun `name`
 *   pyj.get(name)         -> fetch J noun `name` as numpy array
 *   pyj.free()            -> release the J instance
 *
 * Concurrency: ONE J instance per process, single-threaded use. JDo carries
 * interpreter state (recurstate); concurrent calls from multiple threads are
 * undefined. Serialize at the host layer (a lock) if needed.
 * =========================================================================
 *
 * Data path (v2, zero-serialization):
 *   - pyj.set(name, obj)  : numpy array -> J noun via JSetM (io.c setterm:
 *                           one memcpy from our buffer into JE memory)
 *   - pyj.get(name)       : J noun -> numpy array via JGetM (io.c JGetM:
 *                           returns raw shape and data pointers into JE memory;
 *                           one memcpy into the new numpy array)
 *   - pyj.do(sentence)    : execute a J sentence (JDo); output captured
 *
 * The 3!:1/JSetA/JGetA path from v1 is kept in pyj.c history but no longer used:
 * JSetM/JGetM transfer raw ravel bytes with no header encode/decode.
 *
 * Supported dtypes: bool, int64, float64, complex128 (J INT/FL are 64-bit,
 * zero-loss), plus char arrays surfaced as uint8 on the way out.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- JE exported API (see jsrc/jlib.h) ---- */
typedef void* JS;
extern JS JInit(void);
extern JS JInit2(char*);
extern void JSM(JS, void**);
extern int JDo(JS, char*);
extern int JSetM(JS, char*, long long*, long long*, long long*, long long*);
extern int JGetM(JS, char*, long long*, long long*, long long*, long long*);
extern int JFree(JS);

/* output callback state */
static void* g_jt = NULL;
static PyObject* g_output_list = NULL;

static void pyj_output(void* jt, int type, char* s)
{
  (void)jt;
  if (type == 5) return;                 /* MTYOEXIT */
  PyObject* item = PyUnicode_FromString(s ? s : "");
  if (!item) return;
  if (!g_output_list) { Py_DECREF(item); return; }
  PyList_Append(g_output_list, item);    /* best effort */
  Py_DECREF(item);
}

/* J one-hot noun types (jsrc/jtype.h) */
#define JT_B01   1LL
#define JT_LIT   2LL
#define JT_INT   4LL
#define JT_FL    8LL
#define JT_CMPX 16LL

static long long jtype_from_numpy(int npy)
{
  switch (npy) {
  case NPY_BOOL:    return JT_B01;
  case NPY_INT64:   return JT_INT;
  case NPY_DOUBLE:  return JT_FL;
  case NPY_CDOUBLE: return JT_CMPX;
  default:          return 0;
  }
}

static int numpy_type_from_jtype(long long t)
{
  if (t & JT_B01)  return NPY_BOOL;
  if (t & JT_LIT)  return NPY_UINT8;   /* J char -> raw bytes */
  if (t & JT_INT)  return NPY_INT64;
  if (t & JT_FL)   return NPY_DOUBLE;
  if (t & JT_CMPX) return NPY_CDOUBLE;
  return NPY_NOTYPE;
}

static long long atom_bytes(int npy)
{
  switch (npy) {
  case NPY_BOOL:
  case NPY_UINT8:  return 1;
  case NPY_INT64:
  case NPY_DOUBLE: return 8;
  case NPY_CDOUBLE:return 16;
  default:         return 0;
  }
}

/* ---------------- module functions ---------------- */

static PyObject* pyj_do(PyObject* self, PyObject* args)
{
  char* sentence;
  if (!g_jt) { PyErr_SetString(PyExc_RuntimeError, "J not initialized"); return NULL; }
  if (!PyArg_ParseTuple(args, "s", &sentence)) return NULL;
  g_output_list = PyList_New(0);
  if (!g_output_list) return NULL;
  int rc = JDo(g_jt, sentence);
  PyObject* res = Py_BuildValue("(iN)", rc, g_output_list);
  g_output_list = NULL;
  return res;
}

static PyObject* pyj_get(PyObject* self, PyObject* args)
{
  char* name;
  if (!g_jt) { PyErr_SetString(PyExc_RuntimeError, "J not initialized"); return NULL; }
  if (!PyArg_ParseTuple(args, "s", &name)) return NULL;

  long long jtype = 0, jrank = 0;
  long long *jshape = NULL, *jdata = NULL;
  int rc = JGetM(g_jt, name, &jtype, &jrank, &jshape, &jdata);
  if (rc != 0) {
    PyErr_Format(PyExc_ValueError, "J error %d getting name '%s'", rc, name);
    return NULL;
  }
  int npy = numpy_type_from_jtype(jtype);
  if (npy == NPY_NOTYPE || jrank < 0 || jrank > 32) {
    PyErr_Format(PyExc_TypeError,
      "J value of '%s' has unsupported type 0x%llx (boxed/extended/rational/sparse not supported)",
      name, jtype);
    return NULL;
  }
  long long n = 1;
  npy_intp dims[32];
  for (int i = 0; i < jrank; i++) {
    if (jshape[i] < 0) { PyErr_SetString(PyExc_ValueError, "negative shape from J"); return NULL; }
    dims[i] = (npy_intp)jshape[i];
    n *= jshape[i];
  }
  PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(jrank, dims, npy);
  if (!out) return NULL;
  memcpy(PyArray_DATA(out), jdata, n * atom_bytes(npy));
  return (PyObject*)out;
}

static PyObject* pyj_set(PyObject* self, PyObject* args)
{
  char* name;
  PyObject* obj;
  if (!g_jt) { PyErr_SetString(PyExc_RuntimeError, "J not initialized"); return NULL; }
  if (!PyArg_ParseTuple(args, "sO", &name, &obj)) return NULL;

  PyArrayObject* arr = (PyArrayObject*)PyArray_FROM_OTF(obj, NPY_NOTYPE,
                              NPY_ARRAY_IN_ARRAY);
  if (!arr) return NULL;
  int npy = PyArray_TYPE(arr);
  long long jt = jtype_from_numpy(npy);
  if (jt == 0) {
    Py_DECREF(arr);
    PyErr_Format(PyExc_TypeError, "unsupported dtype for J (need bool/int64/float64/complex128)");
    return NULL;
  }
  int rank = PyArray_NDIM(arr);
  long long jrank = rank;
  long long jtype = jt;
  long long shape[32];
  for (int i = 0; i < rank; i++) shape[i] = (long long)PyArray_DIM(arr, i);
  /* JSetM takes double-indirect shape/data (io.c setterm: ((I*)*jshape)[i], (void*)*jdata) */
  long long *pshape = rank ? shape : NULL;
  void* pdata = PyArray_DATA(arr);
  long long *ppdata_dummy = NULL;
  (void)ppdata_dummy;
  int rc = JSetM(g_jt, name, &jtype, &jrank, (long long*)&pshape, (long long*)&pdata);
  Py_DECREF(arr);
  if (rc != 0) { PyErr_Format(PyExc_ValueError, "JSetM failed rc=%d", rc); return NULL; }
  Py_RETURN_NONE;
}

static PyObject* pyj_free(PyObject* self, PyObject* args)
{
  if (g_jt) { JFree(g_jt); g_jt = NULL; }
  Py_RETURN_NONE;
}

static PyMethodDef pyj_methods[] = {
  {"do",  pyj_do,  METH_VARARGS, "do(sentence) -> (rc, [output lines])"},
  {"get", pyj_get, METH_VARARGS, "get(name) -> numpy array"},
  {"set", pyj_set, METH_VARARGS, "set(name, numpy_array_or_scalar)"},
  {"free", pyj_free, METH_NOARGS, "free the J instance"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef pyjmodule = {
  PyModuleDef_HEAD_INIT, "pyj", "Embed the J engine in Python (numpy bridge)",
  -1, pyj_methods
};

PyMODINIT_FUNC PyInit_pyj(void)
{
  import_array();
  if (g_jt) Py_RETURN_NONE;  /* re-init no-op */
  void* callbacks[5] = {pyj_output, 0, 0, 0, (void*)3L};
  const char* libpath = getenv("PYJ_LIBPATH");
  if (libpath && *libpath)
    g_jt = JInit2((char*)libpath);
  else
    g_jt = JInit();
  if (!g_jt) {
    PyErr_SetString(PyExc_ImportError,
      "JInit failed: set PYJ_LIBPATH to the folder containing libj.dylib/.so and profile.ijs");
    return NULL;
  }
  JSM(g_jt, callbacks);
  /* The JE needs one symbol-table assignment before name ops work reliably;
     without it, later JDo/JGetM calls can silently fail. Harmless warmup: */
  {
    long long jtype = JT_INT, jrank = 0;
    long long shape = 0;      /* unused for rank 0 but setterm still touches it when rank>0 */
    long long *pshape = &shape;
    long long val = 0;
    long long *pval = &val;
    JSetM(g_jt, "pyjinit_", &jtype, &jrank, (long long*)&pshape, (long long*)&pval);
    JDo(g_jt, (char*)"pyjinit_=: 0");
  }
  return PyModule_Create(&pyjmodule);
}
