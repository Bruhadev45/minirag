#!/bin/sh
# Re-record docs/media/explain.cast and render explain.gif from it.
# Needs asciinema 3 and agg (both on Homebrew). Run from anywhere; it works
# in the repo root. The "$ ..." prompt line is printed by this script; every
# line after it is the real output of the command, captured as it ran.
set -eu
cd "$(dirname "$0")/../.."
[ -x .venv/bin/python ] || { echo "create .venv first (see README: Develop)" >&2; exit 1; }
CMD=".venv/bin/python -m extras.explain"  # goes inside '...' below: no quotes in it
asciinema rec --headless --overwrite --quiet --output-format asciicast-v2 \
  --window-size 80x30 --title "python -m extras.explain" \
  --command "sh -c 'printf \"\$ %s\\n\" \"$CMD\"; sleep 1; $CMD'" \
  docs/media/explain.cast
agg --font-size 14 --theme github-light --last-frame-duration 8 \
  docs/media/explain.cast docs/media/explain.gif
