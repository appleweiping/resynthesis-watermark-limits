"""Recompute E1 operating points with the enlarged calibration set (>=5000).

The original manifests could extract only 1,928 non-overlapping voice-active 4 s
calibration negatives from dev-clean. This script scores the enlarged clean
calibration pool — dev-clean (manifest) + dev-other + test-other (extra manifests;
all speaker-disjoint from the test-clean evaluation split) — with each baseline's
detector, re-derives the 1%-FPR thresholds, and recomputes every operating point
from the SAVED per-clip test score arrays. AUCs are threshold-free and unchanged;
recalibration diagnostics are preserved from the original runs.

Run on the GPU server:
    python scripts/rethreshold_e1.py --baseline silentcipher --keys 1000
    python scripts/rethreshold_e1.py --baseline audioseal --keys 1000,1017
    python scripts/rethreshold_e1.py --baseline wavmark --keys 1000,1017
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.audio.baselines import build_baselines            # noqa: E402
from experiments.audio.data_io import iter_split, load_manifest    # noqa: E402
from experiments.audio.metrics_audio import (                      # noqa: E402
    operating_point, threshold_at_fpr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--keys", required=True, help="comma-separated key list")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--manifest", default="data/manifest_seed0.json")
    ap.add_argument("--extra", default="data/manifest_calib_extra.json,"
                                       "data/manifest_calib_extra2.json")
    ap.add_argument("--target-fpr", type=float, default=0.01)
    args = ap.parse_args()

    keys = [int(k) for k in args.keys.split(",")]
    b = build_baselines([args.baseline], args.device, strict=True)[0]

    # ---- enlarged clean calibration pool ----------------------------------------
    clips = [x for _, _, x in iter_split(load_manifest(ROOT / args.manifest),
                                         "calibration", None)]
    sources = {"dev-clean": len(clips)}
    for extra in args.extra.split(","):
        m = load_manifest(ROOT / extra)
        add = [x for _, _, x in iter_split(m, "calibration", None)]
        sources[m.get("partition", extra)] = len(add)
        clips.extend(add)
    print(f"[rethr] calibration pool: {len(clips)} clips {sources}", flush=True)
    if len(clips) < 5000:
        raise SystemExit(f"FATAL: pool {len(clips)} < 5000 — spec not met")

    res_path = ROOT / "results" / f"e1_survival_{args.baseline}.json"
    data = json.loads(res_path.read_text(encoding="utf-8"))
    z = np.load(ROOT / "results" / f"e1_scores_{args.baseline}.npz")

    thresholds = {}
    for key in keys:
        scores = np.array([b.score(x, key) for x in clips])
        thresholds[key] = threshold_at_fpr(scores, args.target_fpr)
        print(f"[rethr] {args.baseline} key={key} thr={thresholds[key]:.6g}",
              flush=True)

    for row in data["rows"]:
        tag = f"{args.baseline}|{row['key']}|{row['attacker']}"
        neg, pos = z[tag][0], z[tag][1]
        row["operating_point"] = operating_point(neg, pos, thresholds[row["key"]])
    data["n_calib"] = len(clips)
    data["calibration_sources"] = sources
    res_path.write_text(json.dumps(data), encoding="utf-8")
    print(f"[rethr] rewrote {res_path} (n_calib={len(clips)})")


if __name__ == "__main__":
    main()
