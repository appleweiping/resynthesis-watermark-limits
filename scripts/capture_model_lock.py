#!/usr/bin/env python
"""Capture the exact model set into models.lock.json (run ONCE on a host, then commit).

Loads every attacker + baseline (populating the HF / torch.hub caches), then records,
per logical model in experiments.audio.model_lock.MODEL_SOURCES:
  * hf_revisions:  {repo -> resolved commit SHA} (HF models only),
  * checkpoint_hashes: {name -> {sha256, files:[basenames], source}} where sha256 is
    the PORTABLE digest (basename + content only, no absolute paths — P0-H).

The committed models.lock.json is then verified on every run (verify_all(require=True)).

    python scripts/capture_model_lock.py --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.audio.attackers import build_attackers
from experiments.audio.baselines import build_baselines
from experiments.audio.e1_survival import DEFAULT_ATTACKERS, DEFAULT_BASELINES
from experiments.audio.model_lock import (LOCK_PATH, MODEL_SOURCES, hf_commit,
                                          model_files, portable_hash)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # load everything so caches are populated
    build_attackers(DEFAULT_ATTACKERS, args.device, strict=True)
    build_baselines(DEFAULT_BASELINES, args.device, strict=True)

    lock: dict = {"hf_revisions": {}, "checkpoint_hashes": {}}
    for name, (kind, ref) in MODEL_SOURCES.items():
        files = model_files(name)
        if not files:
            print(f"[warn] {name}: no files found on this host — skipped")
            continue
        if kind == "hf":
            sha = hf_commit(ref)
            if sha:
                lock["hf_revisions"][ref] = sha
        lock["checkpoint_hashes"][name] = {
            "sha256": portable_hash(files),
            "files": sorted(Path(f).name for f in files),
            "n_files": len(files),
            "source": f"{kind}:{ref}",
        }

    Path(LOCK_PATH).write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    print(f"wrote {LOCK_PATH}")
    print(f"  hf_revisions: {len(lock['hf_revisions'])}")
    print(f"  checkpoint_hashes: {len(lock['checkpoint_hashes'])}")
    for name, rec in lock["checkpoint_hashes"].items():
        print(f"    {name:22s} {rec['sha256'][:16]} ({rec['n_files']} files)")
    for repo, sha in lock["hf_revisions"].items():
        print(f"    rev {repo} @ {sha}")


if __name__ == "__main__":
    main()
