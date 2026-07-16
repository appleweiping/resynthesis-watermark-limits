#!/usr/bin/env python
"""Capture the exact model set into models.lock.json (run ONCE on the GPU host).

Loads every attacker + baseline (populating the HF / torch.hub caches), then records:
  * hf_revisions:  {repo -> resolved commit SHA} read from the HF cache refs,
  * checkpoint_hashes: {name -> {sha256, files[]}} SHA-256 over each model's on-disk
    checkpoint files (HF snapshots, torch.hub caches, local model dirs, GH-release DAC).

Commit the resulting models.lock.json; subsequent runs verify against it
(experiments.audio.model_lock.verify_all) and pin from_pretrained via ``revision``.

    python scripts/capture_model_lock.py --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.audio.attackers import DEFAULT_ATTACKERS, build_attackers
from experiments.audio.baselines import build_baselines
from experiments.audio.model_lock import LOCK_PATH, sha256_tree

HF_REPOS = ["charactr/vocos-mel-24khz", "charactr/vocos-encodec-24khz",
            "hubertsiuzdak/snac_24khz", "M4869/WavMark"]


def _hf_cache_root() -> Path:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return Path(HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def _repo_dir(root: Path, repo: str) -> Path:
    return root / ("models--" + repo.replace("/", "--"))


def _resolved_commit(root: Path, repo: str) -> str | None:
    ref = _repo_dir(root, repo) / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


def _snapshot_files(root: Path, repo: str) -> list[str]:
    snaps = _repo_dir(root, repo) / "snapshots"
    if not snaps.exists():
        return []
    return [str(p) for p in snaps.rglob("*") if p.is_file()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # load everything so caches are populated
    build_attackers(DEFAULT_ATTACKERS, args.device, strict=True)
    build_baselines(["audioseal", "wavmark", "silentcipher"], args.device, strict=True)

    root = _hf_cache_root()
    lock: dict = {"hf_revisions": {}, "checkpoint_hashes": {}}
    for repo in HF_REPOS:
        sha = _resolved_commit(root, repo)
        if sha:
            lock["hf_revisions"][repo] = sha
        files = _snapshot_files(root, repo)
        if files:
            lock["checkpoint_hashes"][repo] = {"sha256": sha256_tree(files),
                                               "files": files}

    # local / non-HF checkpoints (best-effort; only recorded if present)
    import torch

    extra = {
        "silentcipher_44_1k": list(Path(
            "/root/autodl-tmp/silentcipher-models/44_1_khz/73999_iteration").glob("*")),
        "knnvc_hub": list((Path(torch.hub.get_dir()) / "bshall_knn-vc_main").rglob("*")),
    }
    for name, paths in extra.items():
        files = [str(p) for p in paths if p.is_file()]
        if files:
            lock["checkpoint_hashes"][name] = {"sha256": sha256_tree(files),
                                               "files": files}

    Path(LOCK_PATH).write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"wrote {LOCK_PATH}")
    print(f"  hf_revisions: {len(lock['hf_revisions'])} repos")
    print(f"  checkpoint_hashes: {len(lock['checkpoint_hashes'])} entries")
    for repo, sha in lock["hf_revisions"].items():
        print(f"    {repo} @ {sha}")


if __name__ == "__main__":
    main()
