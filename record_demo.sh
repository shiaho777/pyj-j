#!/bin/sh
# record_demo.sh — record the money-checker demo as a GIF/cast for the README.
#
# Preferred pipeline (best quality):
#   brew install asciinema agg
#   ./record_demo.sh
#
# Falls back to a plain cast if agg is missing; falls back to instructions
# if asciinema is missing.
set -e
cd "$(dirname "$0")"

CAST=recon/tally_demo.cast
GIF=recon/tally_demo.gif

if ! command -v asciinema >/dev/null 2>&1; then
  cat >&2 <<'MSG'
asciinema not found. Install the toolchain first:

    brew install asciinema agg

or record manually:

    1. Run:  DYLD_LIBRARY_PATH=jlibrary/bin PYJ_LIBPATH=jlibrary/bin \
             python3 demo_tally.py
    2. Record your terminal (Cmd+Shift+5 on macOS).
    3. Trim to ~30 s, upload to recon/tally_demo.gif.

The scene that lands is SCENE 2 (tally_verify saying match: False).
MSG
  exit 1
fi

echo "recording demo to $CAST (3 s lead-in)..."
sleep 3
asciinema rec --overwrite --command \
  "env DYLD_LIBRARY_PATH=jlibrary/bin PYJ_LIBPATH=jlibrary/bin python3 demo_tally.py" \
  "$CAST"
echo "recorded: $CAST"

if command -v agg >/dev/null 2>&1; then
  agg --speed 1.5 --font-size 16 "$CAST" "$GIF"
  echo "wrote $GIF"
else
  echo "agg not found; convert later with:  agg --speed 1.5 $CAST $GIF"
fi
