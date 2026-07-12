"""E1 — deployed-watermark survival across analysis-resynthesis channels.

Protocol (the paper's operational experiment):
  * Baselines (AudioSeal / WavMark / SilentCipher) are tuned to the SAME median
    PESQ target before any attack (matched perceptual budget, not matched SNR).
  * Detector thresholds for TPR@1%FPR come ONLY from the clean calibration split
    (speaker-disjoint from test, >=`--n-calib` clips). The test split never
    touches threshold selection.
  * Reported per (baseline x attacker x key): raw AUC, oriented AUC (cluster
    bootstrap CIs over utterances), calibrated operating point (Clopper-Pearson),
    attack-aware recalibrated diagnostic, payload bit accuracy, and quality
    numbers (PESQ/SI-SDR of embedding; PESQ of the attack on clean audio).
  * Raw per-clip score arrays are saved to results/e1_scores.npz for full
    reproducibility.

Formal runs use --strict: any missing attacker/baseline raises.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .attackers import build_attackers
from .baselines import build_baselines, calibrate_strength_to_pesq
from .data_io import clip_uid, iter_split, load_manifest
from .metrics_audio import full_detection_report, pesq_wb, si_sdr, snr_db

SR = 16_000
ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ATTACKERS = [
    "stft_gl", "mel80_gl", "vocos", "vocos_encodec6k",
    "encodec6k", "encodec3k", "dac", "snac", "knnvc_self4", "knnvc_self8",
]
DEFAULT_BASELINES = ["audioseal", "wavmark", "silentcipher"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--attackers", default=",".join(DEFAULT_ATTACKERS))
    ap.add_argument("--baselines", default=",".join(DEFAULT_BASELINES))
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-calib", type=int, default=5000)
    ap.add_argument("--keys", type=int, default=2, help="independent keys per clip")
    ap.add_argument("--pesq-target", type=float, default=4.2)
    ap.add_argument("--strict", action="store_true", default=True)
    ap.add_argument("--no-strict", dest="strict", action="store_false")
    ap.add_argument("--out", default=str(ROOT / "results" / "e1_survival.json"))
    ap.add_argument("--scores-out", default=str(ROOT / "results" / "e1_scores.npz"))
    args = ap.parse_args()

    man = load_manifest(args.manifest)
    attackers = build_attackers(args.attackers.split(","), args.device, args.strict)
    baselines = build_baselines(args.baselines.split(","), args.device, args.strict)
    keys = [1000 + 17 * k for k in range(args.keys)]

    # ---- load audio ------------------------------------------------------------
    test = [(clip_uid(r), x) for _, r, x in iter_split(man, "test", args.n_test)]
    calib = [x for _, _, x in iter_split(man, "calibration", args.n_calib)]
    fit_sub = [x for _, _, x in iter_split(man, "fit", 24)]
    print(f"[E1] test={len(test)} calib={len(calib)} attackers="
          f"{[a.name for a in attackers]} baselines={[b.name for b in baselines]}")

    # ---- matched perceptual budget ----------------------------------------------
    strengths = {}
    for b in baselines:
        s = calibrate_strength_to_pesq(b, fit_sub, target=args.pesq_target)
        strengths[b.name] = s
        print(f"[E1] {b.name}: strength={s:.3f} (median PESQ ~ {args.pesq_target})")

    # ---- attack severity on clean audio (channel reference) ----------------------
    severity = {}
    for att in attackers:
        vals = []
        for _, x in test[:40]:
            y = att.apply(x)
            vals.append(pesq_wb(SR, x, y))
        severity[att.name] = {
            "pesq_clean_median": float(np.median(vals)),
            "pesq_clean_iqr": [float(np.percentile(vals, 25)),
                               float(np.percentile(vals, 75))],
        }
        print(f"[E1] attack-on-clean PESQ {att.name}: {np.median(vals):.2f}")

    # ---- calibration scores (clean + attacked) -----------------------------------
    # scores are per (baseline, key); attacked also per attacker
    calib_scores: dict = {}
    for b in baselines:
        for key in keys:
            cs = np.array([b.score(x, key) for x in calib])
            calib_scores[(b.name, key, "clean")] = cs
    for att in attackers:
        attacked_calib = [att.apply(x) for x in calib[:1500]]  # diagnostic subset
        for b in baselines:
            for key in keys:
                calib_scores[(b.name, key, att.name)] = np.array(
                    [b.score(y, key) for y in attacked_calib])
        print(f"[E1] calibration attacked scored: {att.name}")

    # ---- test loop ---------------------------------------------------------------
    # cache attacked clean clips per attacker (shared negatives)
    results, score_store = [], {}
    for b in baselines:
        for key in keys:
            # embedding quality
            q_pesq, q_sisdr, q_snr = [], [], []
            marked = []
            for uid, x in test:
                y = b.embed(x, key)
                marked.append((uid, x, y))
            for uid, x, y in marked[:80]:
                q_pesq.append(pesq_wb(SR, x, y))
                q_sisdr.append(si_sdr(x, y))
                q_snr.append(snr_db(x, y))
            quality = {
                "pesq_median": float(np.median(q_pesq)),
                "si_sdr_median": float(np.median(q_sisdr)),
                "snr_db_median": float(np.median(q_snr)),
                "strength": strengths[b.name],
            }
            # clean (no attack) reference row
            neg = np.array([b.score(x, key) for _, x, _ in marked])
            pos = np.array([b.score(y, key) for _, _, y in marked])
            clusters = np.array([uid for uid, _, _ in marked])
            rep = full_detection_report(
                calib_scores[(b.name, key, "clean")], neg, pos, clusters)
            row = {"baseline": b.name, "key": key, "attacker": "none",
                   "quality": quality, **rep}
            results.append(row)
            score_store[f"{b.name}|{key}|none"] = np.stack([neg, pos])
            print(f"[E1] {b.name} key={key} clean: rawAUC={rep['auc_raw']:.3f} "
                  f"TPR@1%={rep['operating_point']['tpr']:.2f}")

            for att in attackers:
                neg_a, pos_a = [], []
                for uid, x, y in marked:
                    neg_a.append(b.score(att.apply(x), key))
                    pos_a.append(b.score(att.apply(y), key))
                neg_a, pos_a = np.array(neg_a), np.array(pos_a)
                rep = full_detection_report(
                    calib_scores[(b.name, key, "clean")], neg_a, pos_a, clusters,
                    calib_neg_attacked=calib_scores[(b.name, key, att.name)])
                results.append({"baseline": b.name, "key": key,
                                "attacker": att.name, **rep})
                score_store[f"{b.name}|{key}|{att.name}"] = np.stack([neg_a, pos_a])
                print(f"[E1] {b.name} key={key} {att.name}: "
                      f"rawAUC={rep['auc_raw']:.3f} orAUC={rep['auc_oriented']:.3f} "
                      f"TPR@1%={rep['operating_point']['tpr']:.2f}")

    out = {
        "n_test": len(test), "n_calib": len(calib), "keys": keys,
        "pesq_target": args.pesq_target,
        "attack_severity": severity,
        "rows": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    np.savez_compressed(args.scores_out,
                        **{k: v for k, v in score_store.items()})
    print(f"[E1] wrote {args.out} and {args.scores_out}")


if __name__ == "__main__":
    main()
