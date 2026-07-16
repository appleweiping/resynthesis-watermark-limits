"""Synthetic smoke test for E2's cluster-aware inference (P0-2).

Runs entirely on fabricated points (no models), so it can execute in CI on CPU.
Checks that:
  * a planted within-attacker association is recovered (obs rho > 0),
  * the common whole-direction permutation null gives a small p under real signal
    and a non-significant p under a null (broken) association,
  * all three cluster CIs are finite and bracket a positive value under signal.
"""

from __future__ import annotations

import numpy as np

from experiments.audio.e2_stats import (_cluster_cis, _common_perm_test,
                                        _within_pooled_spearman)

ATTS = ["mel80_gl", "vocos", "encodec6k", "dac", "snac"]


def _make_points(n_dirs=40, n_utts=16, signal=True, seed=0):
    rng = np.random.default_rng(seed)
    speakers = [f"spk{ i % 8 }" for i in range(n_utts)]
    pts = []
    for d in range(n_dirs):
        base = rng.uniform(0.05, 1.0)                    # direction-level sensitivity
        for ai, a in enumerate(ATTS):
            s = base * (0.5 + ai * 0.2)                   # attacker level offset
            # planted response: monotone in sensitivity within attacker (+noise)
            mu = (0.5 + 0.3 * s) if signal else 0.65
            sens_u = np.clip(rng.normal(s, 0.02, n_utts), 1e-3, None)
            pos = rng.normal(mu + 0.5, 0.4, n_utts)
            neg = rng.normal(0.5, 0.4, n_utts)
            auc = float(np.mean(pos[:, None] > neg[None, :]))
            auc = max(auc, 1 - auc)
            pts.append({
                "dir": d, "attacker": a, "n_utts": n_utts,
                "pred_sensitivity": float(np.mean(sens_u)),
                "auc_mel": auc,
                "utts": {"speaker": speakers,
                         "sens": [float(v) for v in sens_u],
                         "pos_m": [float(v) for v in pos],
                         "neg_m": [float(v) for v in neg]},
            })
    return pts


def test_signal_recovered_and_significant():
    pts = _make_points(signal=True, seed=1)
    x = np.array([p["pred_sensitivity"] for p in pts])
    y = np.array([p["auc_mel"] for p in pts])
    st = np.array([p["attacker"] for p in pts])
    obs = _within_pooled_spearman(x, y, st)
    assert obs > 0.15, obs
    perm = _common_perm_test(pts, "pred_sensitivity", "auc_mel", n_perm=300, seed=0)
    assert abs(perm["observed_within_spearman"] - obs) < 1e-9
    assert perm["p_value"] < 0.05, perm
    cc = _cluster_cis(pts, n_boot=200, seed=0)
    for k in ("ci_direction_cluster", "ci_speaker_cluster", "ci_two_way_dir_x_utt"):
        lo, hi = cc[k]
        assert lo is not None and hi is not None and lo <= hi, (k, cc[k])


def test_null_is_not_significant():
    pts = _make_points(signal=False, seed=2)
    perm = _common_perm_test(pts, "pred_sensitivity", "auc_mel", n_perm=300, seed=0)
    assert perm["p_value"] > 0.05, perm


if __name__ == "__main__":
    test_signal_recovered_and_significant()
    test_null_is_not_significant()
    print("E2 inference smoke test PASSED")
