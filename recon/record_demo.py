#!/usr/bin/env python3
"""Record the money-checker demo as an asciinema .cast file."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PYJ = os.path.dirname(HERE)
sys.path.insert(0, PYJ)
os.environ.setdefault("DYLD_LIBRARY_PATH", os.path.join(PYJ, "jlibrary", "bin"))
os.environ.setdefault("PYJ_LIBPATH", os.path.join(PYJ, "jlibrary", "bin"))

import asciinema.asciicast.v2 as v2
import asciinema.term


class Recorder(v2.Recorder):
    def record(self, command):
        # v2.Recorder.record spawns the command in a pty and captures output
        return super().record(command)


def main():
    cast_path = os.path.join(HERE, "tally_demo.cast")
    cmd = [sys.executable, os.path.join(PYJ, "demo_tally.py")]
    env = dict(os.environ)
    with v2.Recorder(cast_path) as rec:
        rec.record(cmd, env=env)
    print("wrote", cast_path)


if __name__ == "__main__":
    main()
