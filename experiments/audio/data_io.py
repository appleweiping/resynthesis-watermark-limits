"""Manifest-driven clip loading shared by all audio experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SR = 16_000


def load_manifest(path: str | Path) -> dict:
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    if m.get("sr") != SR:
        raise RuntimeError(f"manifest sr={m.get('sr')} != {SR}")
    return m


def load_clip(entry: dict) -> np.ndarray:
    import soundfile as sf

    off, n = int(entry["offset"]), int(entry["clip_samples"])
    x, sr = sf.read(entry["path"], start=off, stop=off + n, dtype="float32")
    if sr != SR:
        raise RuntimeError(f"{entry['path']}: sr={sr} != {SR}")
    if x.ndim > 1:
        x = x[:, 0]
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    # normalize to a fixed RMS so budgets are comparable across clips
    rms = float(np.sqrt(np.mean(x**2)) + 1e-12)
    return (0.05 / rms) * x


def iter_split(manifest: dict, split: str, limit: int | None = None):
    rows = manifest["splits"][split]
    if limit is not None:
        rows = rows[:limit]
    for i, r in enumerate(rows):
        yield i, r, load_clip(r)


def clip_uid(entry: dict) -> str:
    return f"{entry['speaker']}-{Path(entry['path']).stem}-{entry['offset']}"
