#!/bin/sh
# record_demo.sh — record the money-checker demo as a GIF for the README.
#
# Records the demo with dramatic pauses between scenes so the GIF is
# watchable, then converts with agg. Requires: asciinema + agg (brew).
set -e
cd "$(dirname "$0")"

CAST=recon/tally_demo.cast
GIF=recon/tally_demo.gif

echo "recording demo with scene pauses..."
python3 -m asciinema rec --overwrite --command \
  "env DYLD_LIBRARY_PATH=jlibrary/bin PYJ_LIBPATH=jlibrary/bin python3 -u demo_tally.py" \
  "$CAST"

echo "converting to GIF..."
agg --speed 1.0 --font-size 15 --theme dracula "$CAST" "$GIF"
echo "wrote $GIF ($(stat -f%z "$GIF") bytes)"
