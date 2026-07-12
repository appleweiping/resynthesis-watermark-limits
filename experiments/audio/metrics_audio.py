"""Detection metrics with honest orientation, calibration, and uncertainty.

Rules enforced here (paper-wide):
  * RAW AUC (fixed original score direction) and ORIENTED AUC max(a, 1-a) are BOTH
    reported. Theory statements attach to oriented separability only; a raw AUC < 0.5
    is a score sign/order inversion = OPERATIONAL failure of the deployed rule, never
    "information-theoretic erasure".
  * Operating-point metrics (TPR at target FPR) use a threshold fixed on an
    INDEPENDENT calibration set of negatives (>=5000 clips, speaker-disjoint from
    test); the test set never touches threshold selection. Counts get Clopper-Pearson
    intervals.
  * Attack-aware recalibration (threshold re-fit on attacked calibration negatives)
    is reported as a DIAGNOSTIC of remaining separability, clearly labeled.
  * Bootstrap resamples CLUSTERS (utterances; keys jointly), never independent
    pos/neg shuffling.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ---------- AUC ------------------------------------------------------------------
def auc_raw(neg: np.ndarray, pos: np.ndarray) -> float:
    """Mann-Whitney AUC of pos vs neg in the FIXED original score direction."""
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    n0, n1 = len(neg), len(pos)
    if n0 == 0 or n1 == 0:
        return float("nan")
    order = stats.rankdata(np.concatenate([neg, pos]))
    r1 = float(np.sum(order[n0:]))
    return (r1 - n1 * (n1 + 1) / 2.0) / (n0 * n1)


def auc_oriented(neg: np.ndarray, pos: np.ndarray) -> float:
    """max(AUC, 1-AUC): separability irrespective of score sign."""
    a = auc_raw(neg, pos)
    return float(max(a, 1.0 - a))


def cluster_bootstrap_auc(
    neg: np.ndarray, pos: np.ndarray, clusters: np.ndarray,
    n_boot: int = 2000, seed: int = 0, oriented: bool = False,
) -> tuple[float, float, float]:
    """AUC + 95% CI, resampling CLUSTERS (paired pos/neg per utterance/key).

    `clusters` labels each index of the PAIRED arrays (len(neg)==len(pos):
    per-cluster paired scores). Resampling draws clusters with replacement and keeps
    each cluster's pos and neg together — no independent pos/neg shuffling.
    """
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    clusters = np.asarray(clusters)
    assert len(neg) == len(pos) == len(clusters), "paired per-cluster scores required"
    fn = auc_oriented if oriented else auc_raw
    point = fn(neg, pos)
    uniq = np.unique(clusters)
    idx_by_c = {c: np.flatnonzero(clusters == c) for c in uniq}
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_c[c] for c in draw])
        vals[b] = fn(neg[idx], pos[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


# ---------- calibrated operating point --------------------------------------------
def threshold_at_fpr(calib_neg: np.ndarray, fpr: float = 0.01) -> float:
    """Score threshold achieving `fpr` on the CALIBRATION negatives (upper tail)."""
    calib_neg = np.asarray(calib_neg, float)
    if len(calib_neg) < 1000:
        raise ValueError(
            f"calibration set too small ({len(calib_neg)}) for FPR={fpr}: "
            "need >=1000 (paper uses >=5000) — refusing to fake an operating point")
    return float(np.quantile(calib_neg, 1.0 - fpr, method="higher"))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial CI for a proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def operating_point(
    test_neg: np.ndarray, test_pos: np.ndarray, thr: float
) -> dict:
    """TPR and ACHIEVED FPR on the test split at a calibration-fixed threshold."""
    test_neg, test_pos = np.asarray(test_neg, float), np.asarray(test_pos, float)
    k_fp = int(np.sum(test_neg > thr))
    k_tp = int(np.sum(test_pos > thr))
    fpr_lo, fpr_hi = clopper_pearson(k_fp, len(test_neg))
    tpr_lo, tpr_hi = clopper_pearson(k_tp, len(test_pos))
    return {
        "threshold": float(thr),
        "tpr": k_tp / max(1, len(test_pos)), "tpr_ci": [tpr_lo, tpr_hi],
        "fpr": k_fp / max(1, len(test_neg)), "fpr_ci": [fpr_lo, fpr_hi],
        "n_pos": len(test_pos), "n_neg": len(test_neg),
    }


def full_detection_report(
    calib_neg: np.ndarray,
    test_neg: np.ndarray,
    test_pos: np.ndarray,
    clusters: np.ndarray,
    calib_neg_attacked: np.ndarray | None = None,
    target_fpr: float = 0.01,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """The one reporting function every experiment must use.

    calib_neg          : detector scores on CLEAN calibration negatives
    test_neg/test_pos  : paired per-cluster scores on the test split
    calib_neg_attacked : scores on ATTACKED calibration negatives (for the
                         attack-aware recalibration diagnostic), optional
    """
    a_raw, a_raw_lo, a_raw_hi = cluster_bootstrap_auc(
        test_neg, test_pos, clusters, n_boot, seed, oriented=False)
    a_or, a_or_lo, a_or_hi = cluster_bootstrap_auc(
        test_neg, test_pos, clusters, n_boot, seed, oriented=True)
    thr = threshold_at_fpr(calib_neg, target_fpr)
    op = operating_point(test_neg, test_pos, thr)
    out = {
        "auc_raw": a_raw, "auc_raw_ci": [a_raw_lo, a_raw_hi],
        "auc_oriented": a_or, "auc_oriented_ci": [a_or_lo, a_or_hi],
        "sign_inverted": bool(a_raw < 0.5 and a_or > 0.55),
        "operating_point": op,
        "target_fpr": target_fpr,
    }
    if calib_neg_attacked is not None and len(calib_neg_attacked):
        ca = np.asarray(calib_neg_attacked, float)
        if len(ca) >= 500:
            # DIAGNOSTIC ONLY (labeled as such): a smaller attacked-calibration
            # subset is tolerated here at a laxer 2% FPR floor; the PRIMARY
            # operating point above always uses the full clean calibration set.
            diag_fpr = max(target_fpr, 10.0 / len(ca))
            if out["sign_inverted"]:
                thr2 = float(np.quantile(-ca, 1.0 - diag_fpr, method="higher"))
                op2 = operating_point(-test_neg, -test_pos, thr2)
            else:
                thr2 = float(np.quantile(ca, 1.0 - diag_fpr, method="higher"))
                op2 = operating_point(test_neg, test_pos, thr2)
            op2["diag_fpr"] = diag_fpr
            out["recalibrated_diagnostic"] = op2
    return out


# ---------- audio quality -----------------------------------------------------------
def pesq_wb(sr: int, ref: np.ndarray, deg: np.ndarray) -> float:
    from pesq import pesq as pesq_fn

    return float(pesq_fn(sr, np.asarray(ref, np.float64), np.asarray(deg, np.float64), "wb"))


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """Scale-invariant SDR (dB)."""
    ref = np.asarray(ref, np.float64); est = np.asarray(est, np.float64)
    alpha = float(np.dot(est, ref) / (np.dot(ref, ref) + 1e-20))
    err = est - alpha * ref
    return float(10.0 * np.log10((alpha**2 * np.dot(ref, ref) + 1e-20) /
                                 (np.dot(err, err) + 1e-20)))


def snr_db(ref: np.ndarray, deg: np.ndarray) -> float:
    ref = np.asarray(ref, np.float64); deg = np.asarray(deg, np.float64)
    return float(10.0 * np.log10((np.dot(ref, ref) + 1e-20) /
                                 (np.dot(deg - ref, deg - ref) + 1e-20)))
