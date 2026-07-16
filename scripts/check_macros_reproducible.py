#!/usr/bin/env python
"""Byte-level reproducibility check for the auto-generated paper macros (P0-3).

Two guarantees:

  --regen (default): regenerate paper/macros_audio.tex + paper/tab_e1.tex from the
    current result JSONs and assert they are BYTE-IDENTICAL to the committed files.
    This proves the committed manuscript numbers were not hand-edited and match the
    results exactly (no drift). Exits non-zero on any difference.

  --hash: print the SHA-256 of the two macro files. Run this after two INDEPENDENT
    executions of run_audio_all.py (deterministic seeds -> identical results ->
    identical macros); equal hashes across the two runs prove end-to-end determinism.

Usage:
    python scripts/check_macros_reproducible.py            # drift check
    python scripts/check_macros_reproducible.py --hash     # print macro-file hashes
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "paper" / "macros_audio.tex", ROOT / "paper" / "tab_e1.tex"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash", action="store_true",
                    help="print SHA-256 of the macro files (for two-run comparison)")
    args = ap.parse_args()

    if args.hash:
        for f in FILES:
            print(f"{sha(f)}  {f.relative_to(ROOT)}")
        return 0

    before = {f: (f.read_bytes() if f.exists() else None) for f in FILES}
    subprocess.run([sys.executable, "scripts/make_paper_macros.py"], cwd=ROOT,
                   check=True)
    ok = True
    for f in FILES:
        after = f.read_bytes()
        if before[f] is None:
            print(f"[NEW] {f.relative_to(ROOT)} did not exist before regeneration")
            ok = False
        elif after != before[f]:
            print(f"[DRIFT] {f.relative_to(ROOT)} changed on regeneration "
                  f"({len(before[f])} -> {len(after)} bytes) — committed macros do not "
                  "match the result JSONs")
            ok = False
        else:
            print(f"[ok] {f.relative_to(ROOT)} byte-identical after regeneration")
    if not ok:
        print("REPRODUCIBILITY CHECK FAILED", file=sys.stderr)
        return 1
    print("macros are byte-reproducible from the result JSONs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
