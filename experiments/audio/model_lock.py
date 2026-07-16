"""Model pinning + integrity verification (P0-4).

Two mechanisms, both driven by ``models.lock.json`` at the repo root:

  1. HF revision pinning. ``revision(repo)`` returns the locked commit SHA for a
     Hugging Face repo; the attacker/vocoder wrappers pass it to ``from_pretrained``
     so a run cannot silently pick up a newer upstream commit.
  2. Checkpoint-hash verification. ``verify_all(strict=True)`` re-hashes every
     recorded checkpoint (HF snapshot files, torch.hub caches, local model dirs,
     GitHub-release weights) and aborts on any mismatch.

``models.lock.json`` is produced once on the run host by
``scripts/capture_model_lock.py`` (it records each repo's resolved commit SHA and the
SHA-256 of each checkpoint file). Committing that file freezes the model set; every
subsequent run verifies against it. If the file is absent, ``revision`` returns
``None`` (load latest) and ``verify_all`` is a no-op with a printed warning — so the
pipeline still runs, but the paper build requires the committed lock.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "models.lock.json"


def _load() -> dict:
    if LOCK_PATH.exists():
        return json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    return {}


def revision(repo: str, default: str | None = None) -> str | None:
    """Locked HF commit SHA for `repo`, or `default` (None) if unpinned."""
    return _load().get("hf_revisions", {}).get(repo, default)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_tree(paths) -> str:
    """Order-independent digest of a set of files (path + content)."""
    h = hashlib.sha256()
    for p in sorted(str(x) for x in paths):
        h.update(p.encode("utf-8"))
        h.update(sha256_file(p).encode("ascii"))
    return h.hexdigest()


def verify_all(strict: bool = True) -> dict:
    """Re-hash every recorded checkpoint and compare to the lock. Returns a
    {name: 'ok'|'MISMATCH'|'missing'} report; raises on mismatch when strict."""
    lock = _load()
    ckpts = lock.get("checkpoint_hashes", {})
    if not ckpts:
        print("[model_lock] no models.lock.json — skipping verification "
              "(run scripts/capture_model_lock.py on the host to pin models)")
        return {}
    report: dict = {}
    for name, rec in ckpts.items():
        files = rec.get("files", [])
        present = [f for f in files if Path(f).exists()]
        if not present:
            report[name] = "missing"
            if strict:
                raise RuntimeError(f"[model_lock] {name}: locked files absent {files}")
            continue
        digest = sha256_tree(present)
        if digest == rec.get("sha256"):
            report[name] = "ok"
        else:
            report[name] = "MISMATCH"
            if strict:
                raise RuntimeError(
                    f"[model_lock] {name}: checkpoint hash mismatch — locked "
                    f"{rec.get('sha256', '')[:12]} != loaded {digest[:12]}")
    print(f"[model_lock] verified {sum(v == 'ok' for v in report.values())}/"
          f"{len(report)} checkpoints against models.lock.json")
    return report
