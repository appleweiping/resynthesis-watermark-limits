#!/usr/bin/env python
"""Add speaker-cluster CIs to the E1 result JSONs from SAVED score arrays (P1-3).

E1 already ships per-clip paired scores (results/e1_scores_<baseline>.npz, keyed
``baseline|key|attacker`` -> stack([neg, pos])). This recomputes the AUC and
operating-point CIs clustered by SPEAKER (not clip) — cheaply, on CPU, without a GPU
re-run — and writes them back into results/e1_survival_<baseline>.json in place.

    python scripts/recompute_e1_speaker_ci.py --manifest data/manifest_seed0.json

Clip order in the npz is the E1 test order: iter_split(man, "test")[:n_test]. A full
run_audio_all re-run produces these CIs directly (e1_survival now passes
speaker_clusters); this script is the standalone equivalent for existing results.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from experiments.audio.data_io import load_manifest
from experiments.audio.metrics_audio import _cluster_op_ci, cluster_bootstrap_auc

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    man = load_manifest(args.manifest)
    speakers_all = [r["speaker"] for r in man["splits"]["test"]]

    for jf in sorted(glob.glob(str(ROOT / "results" / "e1_survival_*.json"))):
        bl = Path(jf).stem.replace("e1_survival_", "")
        npz_path = ROOT / "results" / f"e1_scores_{bl}.npz"
        if not npz_path.exists():
            print(f"[skip] {bl}: no score npz ({npz_path.name})")
            continue
        # our own E1 score arrays: plain float64 stacks, no pickled objects
        z = np.load(npz_path)
        d = json.loads(Path(jf).read_text(encoding="utf-8"))
        updated = 0
        for row in d["rows"]:
            key = f"{row['baseline']}|{row['key']}|{row['attacker']}"
            if key not in z:
                continue
            neg, pos = z[key][0], z[key][1]
            spk = np.array(speakers_all[:len(neg)])
            _, sr_lo, sr_hi = cluster_bootstrap_auc(neg, pos, spk, args.n_boot,
                                                    0, oriented=False)
            _, so_lo, so_hi = cluster_bootstrap_auc(neg, pos, spk, args.n_boot,
                                                    0, oriented=True)
            row["auc_raw_ci_speaker"] = [sr_lo, sr_hi]
            row["auc_oriented_ci_speaker"] = [so_lo, so_hi]
            thr = row["operating_point"]["threshold"]
            tci, fci = _cluster_op_ci(neg, pos, thr, spk, args.n_boot, 0)
            row["operating_point"]["tpr_ci_speaker"] = tci
            row["operating_point"]["fpr_ci_speaker"] = fci
            row["n_speakers"] = int(len(np.unique(spk)))
            updated += 1
        Path(jf).write_text(json.dumps(d), encoding="utf-8")
        print(f"[ok] {bl}: added speaker-cluster CIs to {updated} rows "
              f"({row.get('n_speakers', '?')} speakers)")


if __name__ == "__main__":
    main()
