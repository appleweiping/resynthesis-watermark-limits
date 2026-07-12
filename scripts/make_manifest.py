"""Build fixed, seeded, speaker-stratified manifests for all audio experiments.

Splits (speaker-disjoint BY CONSTRUCTION — different LibriSpeech partitions):
  * calibration : dev-clean  — negatives for detector threshold calibration
                  (>=5000 segments; multiple non-overlapping segments per utterance)
  * fit         : dev-clean  — utterances for fitting the survival mapping
                  (disjoint utterances from calibration segments' source pool is NOT
                  required — thresholds use scores of *negatives*, the mapping uses
                  attacked pairs — but we keep the segment pools disjoint anyway)
  * test        : test-clean — main evaluation set (~1000 clips), speaker-stratified

Segments are voice-active (energy gate) random offsets, not file heads. The manifest
records speaker/chapter/gender statistics and is fully determined by --seed.

Usage (on the server, where LibriSpeech lives):
    python scripts/make_manifest.py --root /root/autodl-tmp/data --seed 0 \
        --out data/manifest_seed0.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SR = 16_000
CLIP_SECONDS = 4.0
CLIP = int(SR * CLIP_SECONDS)


def _index_partition(part_dir: Path) -> list[dict]:
    """List all flac files with speaker/chapter, duration via soundfile."""
    import soundfile as sf

    rows = []
    for f in sorted(part_dir.rglob("*.flac")):
        spk, chap = f.parts[-3], f.parts[-2]
        info = sf.info(str(f))
        if info.samplerate != SR:
            raise RuntimeError(f"{f}: expected {SR} Hz, got {info.samplerate}")
        rows.append({
            "path": str(f),
            "speaker": spk,
            "chapter": chap,
            "n_samples": int(info.frames),
        })
    if not rows:
        raise RuntimeError(f"no flac files under {part_dir}")
    return rows


def _voice_active_offsets(
    path: str, n_segments: int, rng: np.random.Generator, max_tries: int = 30
) -> list[int]:
    """Random non-overlapping CLIP-length offsets whose energy clears a gate."""
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    n = len(x)
    if n < CLIP:
        return []
    gate = 0.3 * float(np.sqrt(np.mean(x**2)))  # 30% of utterance RMS
    chosen: list[int] = []
    for _ in range(max_tries):
        if len(chosen) >= n_segments:
            break
        off = int(rng.integers(0, n - CLIP + 1))
        if any(abs(off - c) < CLIP for c in chosen):
            continue
        seg = x[off:off + CLIP]
        if float(np.sqrt(np.mean(seg**2))) >= gate:
            chosen.append(off)
    return sorted(chosen)


def _speaker_gender(root: Path) -> dict[str, str]:
    """Parse SPEAKERS.TXT (id | sex | subset | ...) if present."""
    for cand in [root / "LibriSpeech" / "SPEAKERS.TXT", root / "SPEAKERS.TXT"]:
        if cand.exists():
            out = {}
            for line in cand.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith(";"):
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 2:
                    out[parts[0]] = parts[1]
            return out
    return {}


def _stratified_sample(
    rows: list[dict], n_target: int, per_utt: int, rng: np.random.Generator
) -> list[dict]:
    """Speaker-stratified segment sampling: round-robin speakers, random utterances."""
    by_spk: dict[str, list[dict]] = {}
    for r in rows:
        by_spk.setdefault(r["speaker"], []).append(r)
    for lst in by_spk.values():
        rng.shuffle(lst)
    speakers = sorted(by_spk)
    rng.shuffle(speakers)
    out, spk_cursor = [], {s: 0 for s in speakers}
    while len(out) < n_target:
        progressed = False
        for s in speakers:
            if len(out) >= n_target:
                break
            lst, i = by_spk[s], spk_cursor[s]
            while i < len(lst):
                r = lst[i]; i += 1
                offs = _voice_active_offsets(r["path"], per_utt, rng)
                if offs:
                    for off in offs[:per_utt]:
                        out.append({**r, "offset": off, "clip_samples": CLIP})
                    break
            spk_cursor[s] = i
            progressed = progressed or (i <= len(lst))
        if not progressed or all(spk_cursor[s] >= len(by_spk[s]) for s in speakers):
            break
    return out[:n_target]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir containing LibriSpeech/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-fit", type=int, default=400)
    ap.add_argument("--n-calib", type=int, default=5000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    ls = root / "LibriSpeech"
    rng = np.random.default_rng(args.seed)
    gender = _speaker_gender(root)

    dev = _index_partition(ls / "dev-clean")
    test = _index_partition(ls / "test-clean")

    # calibration: many segments/utt allowed; fit: 1 segment/utt, utterances not used
    # for calibration (disjoint utterance pools within dev-clean).
    rng.shuffle(dev)
    n_dev_fit = min(args.n_fit, len(dev) // 3)
    fit_pool, calib_pool = dev[:n_dev_fit * 2], dev[n_dev_fit * 2:]
    fit = _stratified_sample(fit_pool, args.n_fit, 1, rng)
    calib = _stratified_sample(calib_pool, args.n_calib, 4, rng)
    testset = _stratified_sample(test, args.n_test, 1, rng)

    def stats(seg: list[dict]) -> dict:
        spk = sorted({s["speaker"] for s in seg})
        return {
            "n_clips": len(seg),
            "n_speakers": len(spk),
            "gender_counts": {
                g: sum(1 for s in spk if gender.get(s, "?") == g)
                for g in sorted({gender.get(s, "?") for s in spk})
            },
            "n_chapters": len({(s["speaker"], s["chapter"]) for s in seg}),
            "total_hours": round(len(seg) * CLIP_SECONDS / 3600.0, 2),
        }

    manifest = {
        "seed": args.seed,
        "sr": SR,
        "clip_seconds": CLIP_SECONDS,
        "splits": {"calibration": calib, "fit": fit, "test": testset},
        "stats": {k: stats(v) for k, v in
                  [("calibration", calib), ("fit", fit), ("test", testset)]},
        "speaker_disjoint": {
            "calibration_vs_test": True,   # dev-clean vs test-clean partitions
            "fit_vs_test": True,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest), encoding="utf-8")
    print(json.dumps(manifest["stats"], indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
