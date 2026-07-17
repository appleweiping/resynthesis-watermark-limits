"""Model pinning + PORTABLE integrity verification (P0-4 / P0-H).

``models.lock.json`` (repo root, committed) freezes the model set. Two mechanisms:

  1. HF revision pinning. ``revision(repo)`` returns the locked commit SHA; the
     vocoder/codec wrappers pass it to ``from_pretrained`` so a run cannot pick up a
     newer upstream commit.
  2. Portable checkpoint-hash verification. Each logical model hashes to
     sha256 over sorted ``(basename, sha256(file bytes))`` pairs — NO absolute paths
     enter the digest, so the same weights verify on a different host or directory.
     ``verify_all(require=True)`` re-resolves each model's files on THIS host and
     aborts on any mismatch or missing model.

Populate the lock ONCE on a host with ``scripts/capture_model_lock.py``; commit it.
Formal reproduction REQUIRES the committed lock (``verify_all(require=True)`` raises if
it is absent) — it is never auto-generated inside a run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "models.lock.json"

# logical model name -> ("hf", repo) | ("dir", absolute-dir) | ("hub", subdir-name)
# (dir/hub locations are host-specific; only basenames+content enter the hash)
MODEL_SOURCES: dict[str, tuple[str, str]] = {
    "vocos-mel": ("hf", "charactr/vocos-mel-24khz"),
    "vocos-encodec": ("hf", "charactr/vocos-encodec-24khz"),
    "snac-24k": ("hf", "hubertsiuzdak/snac_24khz"),
    "wavmark": ("hf", "M4869/WavMark"),
    "silentcipher-44_1k": ("dir",
                           "/root/autodl-tmp/silentcipher-models/44_1_khz/73999_iteration"),
    "audioseal": ("hf", "facebook/audioseal"),
    "encodec-24k": ("hf", "facebook/encodec_24khz"),
    "knnvc-hub": ("hub", "bshall_knn-vc_main"),
}


def _load() -> dict:
    if LOCK_PATH.exists():
        return json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    return {}


def revision(repo: str, default: str | None = None) -> str | None:
    return _load().get("hf_revisions", {}).get(repo, default)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def portable_hash(files) -> str:
    """Digest over sorted (basename, content-sha256) — path-independent (P0-H)."""
    items = sorted((Path(p).name, sha256_file(p)) for p in files)
    h = hashlib.sha256()
    for name, digest in items:
        h.update(name.encode("utf-8")); h.update(b":"); h.update(digest.encode())
    return h.hexdigest()


def _hf_cache_root():
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return Path(HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def hf_commit(repo: str) -> str | None:
    ref = _hf_cache_root() / ("models--" + repo.replace("/", "--")) / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


def model_files(name: str) -> list[str]:
    """Resolve a logical model's checkpoint files on THIS host (shared by capture and
    verify, so both hash exactly the same set)."""
    kind, ref = MODEL_SOURCES[name]
    if kind == "hf":
        snaps = _hf_cache_root() / ("models--" + ref.replace("/", "--")) / "snapshots"
        if not snaps.exists():
            return []
        return [str(p) for p in snaps.rglob("*")
                if p.is_file() and ".no_exist" not in str(p)]
    if kind == "dir":
        d = Path(ref)
        return [str(p) for p in d.rglob("*") if p.is_file()] if d.exists() else []
    if kind == "hub":
        import torch
        d = Path(torch.hub.get_dir()) / ref
        return [str(p) for p in d.rglob("*") if p.is_file()] if d.exists() else []
    return []


def verify_all(strict: bool = True, require: bool = False) -> dict:
    """Re-resolve and re-hash every locked model on THIS host; compare to the lock.
    ``require=True`` aborts if the lock is absent (formal reproduction); otherwise a
    missing lock is a warned no-op."""
    lock = _load()
    ckpts = lock.get("checkpoint_hashes", {})
    if not ckpts:
        msg = ("[model_lock] models.lock.json missing/empty — run "
               "scripts/capture_model_lock.py on a host and commit it")
        if require:
            raise RuntimeError(msg + " (required for a formal run)")
        print(msg + " (skipping verification)")
        return {}
    report: dict = {}
    for name, rec in ckpts.items():
        files = model_files(name) if name in MODEL_SOURCES else []
        if not files:
            report[name] = "missing"
            if strict:
                raise RuntimeError(f"[model_lock] {name}: no checkpoint files found "
                                   "on this host")
            continue
        digest = portable_hash(files)
        report[name] = "ok" if digest == rec.get("sha256") else "MISMATCH"
        if report[name] == "MISMATCH" and strict:
            raise RuntimeError(
                f"[model_lock] {name}: portable hash mismatch — locked "
                f"{rec.get('sha256', '')[:12]} != loaded {digest[:12]}")
    print(f"[model_lock] verified {sum(v == 'ok' for v in report.values())}/"
          f"{len(report)} models against models.lock.json")
    return report
