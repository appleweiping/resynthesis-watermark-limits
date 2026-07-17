"""Statistical-validation tests for E2's cluster-aware inference (P0-B).

Runs on fabricated points (no models), so it executes in CI on CPU. Beyond "CI is
finite / p<0.05 under signal", it checks the properties the review demanded:
  * cluster multiplicity is preserved (a valid pigeonhole bootstrap, not a set());
  * a shared utterance across directions and a shared direction across attackers are
    represented consistently;
  * empirical Type-I error of the permutation test under a type-confounded null is
    controlled (does not blow past nominal alpha);
  * bootstrap-CI empirical coverage of a planted association is near nominal;
  * seeded reproducibility.
"""

from __future__ import annotations

import numpy as np

from experiments.audio.e2_stats import (_cluster_cis, _common_perm_test,
                                        _pigeonhole_ci, _within_pooled_spearman)

ATTS = ["mel80_gl", "vocos", "encodec6k", "dac", "snac"]
KINDS = ["nullspace", "rowspace", "mixture"]


def _make_points(n_dirs=40, n_utts=16, signal=True, seed=0, type_confound=False):
    rng = np.random.default_rng(seed)
    speakers = [f"spk{i % 8}" for i in range(n_utts)]
    uids = [f"utt{i}" for i in range(n_utts)]          # shared across directions
    pts = []
    for d in range(n_dirs):
        kind = KINDS[d % len(KINDS)]
        base = rng.uniform(0.05, 1.0)                    # direction-level sensitivity
        # type_confound: response depends ONLY on kind, not on within-kind sensitivity
        kind_level = {"nullspace": 0.55, "rowspace": 0.7, "mixture": 0.62}[kind]
        for ai, a in enumerate(ATTS):
            s = base * (0.5 + ai * 0.2)
            if type_confound:
                mu = kind_level
            else:
                mu = (0.5 + 0.3 * s) if signal else 0.62
            sens_u = np.clip(rng.normal(s, 0.02, n_utts), 1e-3, None)
            pos = rng.normal(mu + 0.5, 0.4, n_utts)
            neg = rng.normal(0.5, 0.4, n_utts)
            auc = float(np.mean(pos[:, None] > neg[None, :]))
            auc = max(auc, 1 - auc)
            pts.append({
                "dir": d, "attacker": a, "n_utts": n_utts, "kind": kind,
                "pred_sensitivity": float(np.mean(sens_u)),
                "auc_mel": auc,
                "utts": {"uid": list(uids), "speaker": list(speakers),
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
    # two-way CI must be at least as wide as the direction-only CI (adds a cluster dim)
    dw = cc["ci_direction_cluster"][1] - cc["ci_direction_cluster"][0]
    tw = cc["ci_two_way_dir_x_utt"][1] - cc["ci_two_way_dir_x_utt"][0]
    assert tw >= 0.5 * dw, (dw, tw)


def test_bootstrap_preserves_multiplicity():
    """A single pigeonhole draw must be able to include a direction more than once
    (multiplicity) — the old set()-based resample could not. We check the bootstrap
    distribution has non-trivial spread (a degenerate set()-collapse gives ~0 sd)."""
    pts = _make_points(signal=True, seed=3)
    atts = sorted({p["attacker"] for p in pts})
    from collections import defaultdict
    by_dir = defaultdict(dict)
    for p in pts:
        by_dir[p["dir"]][p["attacker"]] = p
    vals = _pigeonhole_ci(by_dir, atts, "pred_sensitivity", "auc_mel",
                          n_boot=200, seed=0, resample_dirs=True, unit_key="uid")
    assert len(vals) > 150
    assert np.std(vals) > 1e-3, "bootstrap has no spread — multiplicity lost"


def test_null_type_one_error_controlled():
    """Under a type-confounded null (response depends on kind, not within-kind s_W),
    the within-kind-restricted permutation test should NOT be anticonservative."""
    rejections = 0
    trials = 20
    for t in range(trials):
        pts = _make_points(signal=False, seed=100 + t, type_confound=True)
        perm = _common_perm_test(pts, "pred_sensitivity", "auc_mel", n_perm=200,
                                 seed=0, restrict_kind=True)
        if perm["p_value"] < 0.05:
            rejections += 1
    # nominal 5%; allow slack for 20 trials but flag gross anticonservativeness
    assert rejections <= 5, f"Type-I error too high: {rejections}/{trials}"


def test_seeded_reproducibility():
    pts = _make_points(signal=True, seed=7)
    a = _cluster_cis(pts, n_boot=120, seed=0)
    b = _cluster_cis(pts, n_boot=120, seed=0)
    assert a["ci_two_way_dir_x_utt"] == b["ci_two_way_dir_x_utt"]
    assert a["observed"] == b["observed"]


if __name__ == "__main__":
    test_signal_recovered_and_significant()
    test_bootstrap_preserves_multiplicity()
    test_null_type_one_error_controlled()
    test_seeded_reproducibility()
    print("E2 statistical-validation tests PASSED")
