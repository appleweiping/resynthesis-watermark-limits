"""Merge the two SilentCipher half-shards of E1 into one statistically valid file.

Both halves used the SAME manifest, key, and calibration split, hence identical
thresholds (verified below). Pooling therefore happens at the SCORE level:
per-clip score arrays from the .npz shards are concatenated per attacker, and
every metric (raw/oriented AUC with utterance-clustered bootstrap, calibrated
operating point with Clopper-Pearson) is recomputed on the pooled arrays.
The consumed half-shard JSONs are moved to results/shards/.

Usage: python scripts/merge_e1_shards.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.audio.metrics_audio import (  # noqa: E402
    cluster_bootstrap_auc, operating_point)

RES = ROOT / "results"


def main() -> None:
    a = json.loads((RES / "e1_survival_silentcipher_a.json").read_text())
    b = json.loads((RES / "e1_survival_silentcipher_b.json").read_text())
    za = np.load(RES / "e1_scores_silentcipher_a.npz")
    zb = np.load(RES / "e1_scores_silentcipher_b.npz")

    rows_a = {(r["attacker"], r["key"]): r for r in a["rows"]}
    rows_b = {(r["attacker"], r["key"]): r for r in b["rows"]}
    assert rows_a.keys() == rows_b.keys(), "shard row sets differ"

    merged_rows, key_of = [], None
    for (att, key), ra in rows_a.items():
        rb = rows_b[(att, key)]
        thr_a = ra["operating_point"]["threshold"]
        thr_b = rb["operating_point"]["threshold"]
        if not np.isclose(thr_a, thr_b, rtol=1e-9):
            raise SystemExit(f"FATAL: thresholds differ for {att} "
                             f"({thr_a} vs {thr_b}) — calibration mismatch")
        tag = f"silentcipher|{key}|{att}"
        neg = np.concatenate([za[tag][0], zb[tag][0]])
        pos = np.concatenate([za[tag][1], zb[tag][1]])
        clusters = np.arange(len(neg))          # one cluster per clip (uids disjoint)
        a_raw, a_lo, a_hi = cluster_bootstrap_auc(neg, pos, clusters, 2000, 0, False)
        a_or, o_lo, o_hi = cluster_bootstrap_auc(neg, pos, clusters, 2000, 0, True)
        row = {
            "baseline": "silentcipher", "key": key, "attacker": att,
            "auc_raw": a_raw, "auc_raw_ci": [a_lo, a_hi],
            "auc_oriented": a_or, "auc_oriented_ci": [o_lo, o_hi],
            "sign_inverted": bool(a_raw < 0.5 and a_or > 0.55),
            "operating_point": operating_point(neg, pos, thr_a),
            "target_fpr": ra["target_fpr"],
        }
        if "quality" in ra:                      # clean row carries quality info
            row["quality"] = ra["quality"]
        if "recalibrated_diagnostic" in ra:
            d_thr = ra["recalibrated_diagnostic"]["threshold"]
            flip = -1.0 if row["sign_inverted"] else 1.0
            op2 = operating_point(flip * neg, flip * pos, d_thr)
            op2["diag_fpr"] = ra["recalibrated_diagnostic"].get("diag_fpr")
            row["recalibrated_diagnostic"] = op2
        merged_rows.append(row)

    out = {
        "n_test": a["n_test"] + b["n_test"],
        "n_calib": a["n_calib"], "keys": a["keys"],
        "pesq_target": a["pesq_target"],
        "attack_severity": a["attack_severity"],
        "rows": merged_rows,
        "merged_from": ["silentcipher_a", "silentcipher_b"],
    }
    (RES / "e1_survival_silentcipher.json").write_text(json.dumps(out))
    shards = RES / "shards"; shards.mkdir(exist_ok=True)
    for stem in ["e1_survival_silentcipher_a.json", "e1_survival_silentcipher_b.json"]:
        shutil.move(str(RES / stem), str(shards / stem))
    print(f"merged {len(merged_rows)} rows -> e1_survival_silentcipher.json "
          f"(n_test={out['n_test']})")


if __name__ == "__main__":
    main()
