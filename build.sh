#!/bin/sh
# Build the pyj extension (J-in-Python CPython extension).
#
# Prereqs:
#   1. A J engine runtime in ./jlibrary/bin (libj.dylib/.so + profile.ijs).
#      Either build it from jsource:
#        git clone https://github.com/jsoftware/jsource.git ../jsource
#        cd ../jsource/make2 && ./build_libj.sh && ./build_jconsole.sh && ./cpbin.sh
#      (jsource cpbin.sh copies binaries into ../jlibrary/bin — point that at
#       this project's jlibrary/bin or copy manually after)
#      Or copy the bin/ folder of a released J distribution.
#   2. GMP (the J engine dlopens it at startup):  brew install gmp   (macOS)
#   3. python3 with numpy in the interpreter you'll import from.
#
# Environment overrides:
#   PY_INC / NP_INC / PY_LIB  — python headers, numpy headers, python libdir
set -e
cd "$(dirname "$0")"

PY_INC="${PY_INC:-$(python3 - <<'PYEOF'
import sysconfig, glob, os
cands = [sysconfig.get_paths()["include"],
         os.path.join(sysconfig.get_config_var("prefix"), "include", "python3")]
# macOS system python: headers live in the Xcode/CLT framework
import subprocess, sys
try:
    sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
    majmin = f"{sys.version_info[0]}.{sys.version_info[1]}"
    cands.append(f"{sdk}/System/Library/Frameworks/Python3.framework/Versions/3.9/Headers")
except Exception:
    pass
try:
    import glob as _g
    cands += _g.glob("/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/*/Headers")
except Exception:
    pass
for c in cands:
    if c and os.path.isfile(os.path.join(c, "Python.h")):
        print(c); break
else:
    print(sysconfig.get_paths()["include"])
PYEOF
)}"
NP_INC="${NP_INC:-$(python3 -c 'import numpy; print(numpy.get_include())')}"
PY_LIB="${PY_LIB:-$(python3 -c 'import sysconfig; p=sysconfig.get_paths().get("LIBDIR") or sysconfig.get_config_var("LIBPL") or sysconfig.get_config_var("prefix"); print(p)')}"
PY_VER="${PY_VER:-$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LDVERSION") or "".join(map(str,sys.version_info[:2])))')}"

LIBDIR="$PWD/jlibrary/bin"
if [ ! -f "$LIBDIR/libj.dylib" ] && [ ! -f "$LIBDIR/libj.so" ]; then
  echo "ERROR: no libj in $LIBDIR — build jsource or copy a J release there (see header)."
  exit 1
fi

CC="${CC:-cc}"
UNAME="$(uname)"
if [ "$UNAME" = "Darwin" ]; then
  # jsource's build leaves libj.dylib with a bare install_name; make it absolute
  otool -D "$LIBDIR/libj.dylib" | grep -q "^$LIBDIR" || \
    install_name_tool -id "$LIBDIR/libj.dylib" "$LIBDIR/libj.dylib"
  # JE dlopens "libgmp.dylib" from its lib dir at startup
  [ -f "$LIBDIR/libgmp.dylib" ] || cp "$(brew --prefix)/lib/libgmp.dylib" "$LIBDIR/"
fi

$CC -O2 -fPIC -shared -o pyj.so pyj.c \
  -I"$PY_INC" -I"$NP_INC" \
  -L"$LIBDIR" -lj -L"$PY_LIB" -lpython$PY_VER \
  $([ "$UNAME" = "Darwin" ] && echo "-Wl,-headerpad_max_install_names")

echo "built pyj.so"
echo "run tests with:"
echo "  DYLD_LIBRARY_PATH=$LIBDIR PYJ_LIBPATH=$LIBDIR python3 test_pyj.py"
