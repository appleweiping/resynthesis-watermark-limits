#!/usr/bin/env bash
# P0-A gate: the paper must contain no placeholder or non-finite numbers. Fails (exit 1)
# if any of these tokens appear in paper/ .tex sources. Run in the formal build and CI.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PATTERN='\[pending\]|\bnan\b|\binf\b|TODO|provisional|XXX|FIXME'
# search generated + hand-written tex (not the .log/.aux)
hits="$(grep -RInE "$PATTERN" "$ROOT/paper"/*.tex 2>/dev/null || true)"
if [ -n "$hits" ]; then
  echo "FAIL: placeholder/non-finite tokens found in paper/*.tex:"
  echo "$hits"
  exit 1
fi
echo "OK: no [pending]/nan/inf/TODO/provisional tokens in paper/*.tex"
