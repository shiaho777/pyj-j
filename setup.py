"""
setup.py -- build the `tally` wheel: exact-arithmetic kernel + MCP server.

The J engine (libj, ~5 MB) and GMP are bundled INSIDE the wheel, so
`pip install tally` works with zero build steps and zero environment
variables on the end-user machine.

Build:  python3 setup.py bdist_wheel
Install: pip install dist/tally-*.whl
Use:     tally-mcp            (MCP stdio server)
         tally 0.1 + 0.2      (CLI calculator)
         python3 -c "import tally; print(tally.compute('0.1+0.2'))"
"""
import os
import shutil
import subprocess
import sys
import sysconfig

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

HERE = os.path.dirname(os.path.abspath(__file__))
LIBDIR = os.path.join(HERE, "jlibrary", "bin")


def _python_include():
    cands = [sysconfig.get_paths()["include"],
             os.path.join(sysconfig.get_config_var("prefix") or "", "include",
                          "python3")]
    if sys.platform == "darwin":
        try:
            sdk = subprocess.check_output(
                ["xcrun", "--show-sdk-path"], text=True).strip()
            cands.append(f"{sdk}/System/Library/Frameworks/"
                         f"Python3.framework/Versions/3.9/Headers")
        except Exception:
            pass
        import glob
        cands += glob.glob("/Applications/Xcode.app/Contents/Developer/"
                           "Library/Frameworks/Python3.framework/"
                           "Versions/*/Headers")
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "Python.h")):
            return c
    return sysconfig.get_paths()["include"]


def _numpy_include():
    # sdist metadata generation must not require numpy: the include path is
    # only needed when actually compiling pyj.c.
    try:
        import numpy
        return numpy.get_include()
    except ImportError:
        return ""


def _engine_archs():
    """Architecture slices of the built libj.dylib (macOS only)."""
    libj = os.path.join(LIBDIR, "libj.dylib")
    if sys.platform != "darwin" or not os.path.isfile(libj):
        return []
    try:
        out = subprocess.check_output(["lipo", "-archs", libj],
                                      text=True).strip()
        return out.split()
    except Exception:
        return []


class BuildExtWithEngine(build_ext):
    """Compile pyj.c, then bundle + relocate the engine into the package."""

    def build_extension(self, ext):
        # macOS: use dynamic_lookup so the host interpreter supplies Python
        # symbols (no libpython link needed for an extension module).
        # Linux: set RPATH to $ORIGIN so the bundled libj.so resolves next
        # to the extension at runtime.
        if sys.platform == "darwin":
            ext.extra_link_args = [
                "-Wl,-undefined,dynamic_lookup",
                "-Wl,-headerpad_max_install_names",
            ]
            # A universal2 interpreter makes distutils add -arch arm64
            # -arch x86_64, but the engine we link against is single-arch
            # on CI runners -> "ld: library 'j' not found" for the missing
            # slice. Build the extension for exactly the engine's archs.
            archs = _engine_archs()
            if archs:
                for a in archs:
                    ext.extra_compile_args += ["-arch", a]
                    ext.extra_link_args += ["-arch", a]
        elif sys.platform.startswith("linux"):
            ext.extra_link_args = ["-Wl,-rpath,$ORIGIN",
                                   "-Wl,--enable-new-dtags"]
        super().build_extension(ext)

        # locate the built extension inside the package build dir
        ext_path = self.get_ext_fullpath(ext.name)
        pkg_dir = os.path.dirname(ext_path)
        os.makedirs(pkg_dir, exist_ok=True)

        if sys.platform == "darwin":
            self._bundle_macos(ext_path, pkg_dir)
        elif sys.platform.startswith("linux"):
            self._bundle_linux(ext_path, pkg_dir)

    def _bundle_macos(self, ext_path, pkg_dir):
        for fname in ("libj.dylib", "profile.ijs"):
            src = os.path.join(LIBDIR, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(pkg_dir, fname))
        libj = os.path.join(pkg_dir, "libj.dylib")
        libgmp = os.path.join(pkg_dir, "libgmp.dylib")

        def _tool(*args):
            subprocess.run(["install_name_tool", *args],
                           check=False, capture_output=True)

        # GMP: prefer jsource's vendored dylib -- it only links libSystem
        # (self-contained). A brew-installed libgmp.dylib instead carries an
        # absolute LC_LOAD_DYLIB to /opt/homebrew/... which breaks the wheel
        # on any machine without brew at that exact path. The engine dlopens
        # libpath/libgmp.dylib before the bare name, so the bundled copy
        # wins over any system GMP.
        gmp_src = None
        for cand in (os.path.join(HERE, "jsource", "mpir", "apple", "macos",
                                  "libgmp.dylib"),
                     os.path.join(HERE, "..", "jsource", "mpir", "apple",
                                  "macos", "libgmp.dylib"),
                     os.path.join(LIBDIR, "libgmp.dylib")):
            if os.path.isfile(cand):
                gmp_src = cand
                break
        if gmp_src:
            shutil.copy2(gmp_src, libgmp)

        if os.path.isfile(libj):
            _tool("-id", "@loader_path/libj.dylib", libj)
        if os.path.isfile(libgmp):
            # fix both the id AND the brew self-reference inside the vendored
            # dylib (its load commands still point at /opt/homebrew or
            # /usr/local); after this it resolves only within the package.
            _tool("-id", "@loader_path/libgmp.dylib", libgmp)
            for old in ("/opt/homebrew/opt/gmp/lib/libgmp.10.dylib",
                        "/usr/local/opt/gmp/lib/libgmp.10.dylib"):
                _tool("-change", old, "@loader_path/libgmp.dylib", libgmp)
        old = os.path.join(LIBDIR, "libj.dylib")
        _tool("-change", old, "@loader_path/libj.dylib", ext_path)
        _tool("-change", "libj.dylib", "@loader_path/libj.dylib", ext_path)
        for f in (libj, libgmp, ext_path):
            if os.path.isfile(f):
                subprocess.run(["codesign", "--remove-signature", f],
                               check=False, capture_output=True)
                subprocess.run(["codesign", "-s", "-", f],
                               check=False, capture_output=True)

    def _bundle_linux(self, ext_path, pkg_dir):
        # libj.so (or the avx2/avx512 alias) + profile.ijs + libgmp.
        # On Linux the engine dlopens libgmp.so.10 (system) then falls back
        # to libpath/libgmp.so — so we bundle libgmp.so for portability.
        libj_name = None
        for cand in ("libj.so", "libjavx2.so", "libjavx512.so"):
            if os.path.isfile(os.path.join(LIBDIR, cand)):
                libj_name = cand
                break
        if libj_name is None:
            return
        shutil.copy2(os.path.join(LIBDIR, libj_name),
                     os.path.join(pkg_dir, "libj.so"))
        for fname in ("profile.ijs", "libgmp.so", "libgmp.so.10"):
            src = os.path.join(LIBDIR, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(pkg_dir, fname))
        # ensure the extension's NEEDED entry is libj.so and RPATH=$ORIGIN
        subprocess.run(["patchelf", "--set-soname", "libj.so",
                        os.path.join(pkg_dir, "libj.so")],
                       check=False, capture_output=True)
        subprocess.run(["patchelf", "--set-rpath", "$ORIGIN", ext_path],
                       check=False, capture_output=True)
        subprocess.run(["patchelf", "--replace-needed", libj_name,
                        "libj.so", ext_path],
                       check=False, capture_output=True)


pyj_ext = Extension(
    "tally._pyj",
    sources=["pyj.c"],
    include_dirs=[_python_include(), _numpy_include(), LIBDIR],
    library_dirs=[LIBDIR],
    libraries=["j"],
    extra_compile_args=["-O2", "-fPIC",
                        "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
                        '-DPYJ_MODULE_NAME="tally._pyj"',
                        "-DPYJ_INIT_FUNC=PyInit__pyj"],
)

setup(
    name="tally",
    version="1.0.0",
    description="Exact-arithmetic kernel: every result is an exact integer "
                "or p/q rational. Floats never enter the engine.",
    long_description=(open(os.path.join(HERE, "README_TALLY.md")).read()
                      if os.path.isfile(os.path.join(HERE, "README_TALLY.md"))
                      else "Exact-arithmetic kernel: integers and p/q "
                           "rationals, floats never enter the engine."),
    long_description_content_type="text/markdown",
    author="pyj",
    license="GPL-3.0",
    packages=["tally"],
    package_dir={"tally": "tally_pkg"},
    ext_modules=[pyj_ext],
    cmdclass={"build_ext": BuildExtWithEngine},
    install_requires=["numpy"],
    entry_points={
        "console_scripts": [
            "tally=tally.cli:main",
            "tally-mcp=tally.server:main",
            "tally-mcp-install=tally.install_mcp:main",
        ],
    },
    python_requires=">=3.9",
    zip_safe=False,
)