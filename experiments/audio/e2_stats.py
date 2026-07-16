"""Pure-Python (no torch) inference for E2 — cluster-aware statistics (P0-2).

Split out of ``e2_predictor`` so the statistics can be unit-tested and run in CI on
CPU without installing torch/torchaudio. Only numpy / scipy / scikit-learn and the
torch-free AUC helpers from ``metrics_audio`` are used here.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .metrics_audio import auc_oriented


def _within_rank(vals: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """Rank-normalize within each stratum (attacker) -> comparable across strata."""
    from scipy import stats as sps

    out = np.empty_like(vals, dtype=float)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        out[idx] = sps.rankdata(vals[idx]) / (len(idx) + 1.0)
    return out


def _within_pooled_spearman(x: np.ndarray, y: np.ndarray,
                            strata: np.ndarray) -> float:
    from scipy import stats as sps
    return float(sps.spearmanr(_within_rank(x, strata),
                               _within_rank(y, strata)).statistic)


def _fit_eval(fit_pts, test_pts, pred_key: str, response: str = "auc_oriented",
              n_boot: int = 1000, seed: int = 0) -> dict:
    from scipy import stats as sps
    from scipy.optimize import curve_fit
    from scipy.stats import norm as snorm

    def norm_pred(pts, lo, hi):
        v = np.array([p[pred_key] for p in pts], float)
        return np.clip((v - lo) / max(hi - lo, 1e-12), 0.0, 1.0)

    xf_raw = np.array([p[pred_key] for p in fit_pts], float)
    lo, hi = float(np.min(xf_raw)), float(np.max(xf_raw))
    xf, yf = norm_pred(fit_pts, lo, hi), np.array([p[response] for p in fit_pts])
    xt, yt = norm_pred(test_pts, lo, hi), np.array([p[response] for p in test_pts])

    model = lambda f, a, b: snorm.cdf(a * np.sqrt(np.clip(f, 0, 1)) + b)
    try:
        (a_hat, b_hat), _ = curve_fit(model, xf, np.clip(yf, 1e-3, 1 - 1e-3),
                                      p0=[3.0, -1.5], maxfev=20000)
        yhat = model(xt, a_hat, b_hat)
    except Exception:
        a_hat = b_hat = float("nan")
        yhat = np.full_like(yt, np.mean(yf))
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip").fit(xf, yf)
    yhat_iso = iso.predict(xt)

    def metrics(yh):
        ss_res = float(np.sum((yt - yh) ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2)) + 1e-20
        return {
            "spearman": float(sps.spearmanr(xt, yt).statistic),
            "r2": 1.0 - ss_res / ss_tot,
            "rmse": float(np.sqrt(np.mean((yt - yh) ** 2))),
        }

    out = {"pred": pred_key, "phi_fit": metrics(yhat),
           "isotonic": metrics(yhat_iso),
           "phi_params": [float(a_hat), float(b_hat)],
           "n_fit": len(xf), "n_test": len(xt)}

    dirs = np.array([p["dir"] for p in test_pts])
    uniq = np.unique(dirs)
    rng = np.random.default_rng(seed)
    sp = []
    for _ in range(n_boot):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(dirs == t) for t in take])
        if len(np.unique(xt[idx])) > 2:
            sp.append(sps.spearmanr(xt[idx], yt[idx]).statistic)
    out["spearman_ci"] = [float(np.percentile(sp, 2.5)),
                          float(np.percentile(sp, 97.5))] if sp else [None, None]
    return out


def _common_perm_test(test_pts, pred_key: str, response: str,
                      n_perm: int = 2000, seed: int = 0) -> dict:
    """WITHIN-attacker rank-pooled Spearman with a COMMON whole-direction permutation
    null (P0-2). The same permutation of the direction axis is applied across every
    attacker stratum, so a direction moves as one unit — respecting the fact that a
    direction is a single cluster shared by all five attackers. An independent
    within-attacker shuffle (the previous null) breaks that coupling and is
    anticonservative."""
    from scipy import stats as sps

    atts = sorted({p["attacker"] for p in test_pts})
    dirs = sorted({p["dir"] for p in test_pts})
    ai = {a: i for i, a in enumerate(atts)}
    di = {d: i for i, d in enumerate(dirs)}
    X = np.full((len(atts), len(dirs)), np.nan)
    Y = np.full((len(atts), len(dirs)), np.nan)
    for p in test_pts:
        X[ai[p["attacker"]], di[p["dir"]]] = p[pred_key]
        Y[ai[p["attacker"]], di[p["dir"]]] = p[response]
    if np.isnan(X).any() or np.isnan(Y).any():
        keep = [j for j in range(len(dirs))
                if not (np.isnan(X[:, j]).any() or np.isnan(Y[:, j]).any())]
        X, Y = X[:, keep], Y[:, keep]

    def pooled(xm, ym):
        xr = np.vstack([sps.rankdata(xm[r]) / (xm.shape[1] + 1.0)
                        for r in range(xm.shape[0])])
        yr = np.vstack([sps.rankdata(ym[r]) / (ym.shape[1] + 1.0)
                        for r in range(ym.shape[0])])
        return float(sps.spearmanr(xr.ravel(), yr.ravel()).statistic)

    obs = pooled(X, Y)
    rng = np.random.default_rng(seed)
    null = np.array([pooled(X[:, rng.permutation(X.shape[1])], Y)
                     for _ in range(n_perm)])
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
    return {"observed_within_spearman": obs, "p_value": p,
            "null_abs_95": float(np.percentile(np.abs(null), 95)),
            "n_perm": n_perm,
            "null_type": "common whole-direction permutation across attackers"}


def _cluster_cis(test_pts, pred_key: str = "pred_sensitivity",
                 response: str = "auc_mel", n_boot: int = 2000,
                 seed: int = 0) -> dict:
    """Bootstrap CIs of the within-attacker rank-pooled Spearman under three
    clusterings (P0-2): by direction, by speaker, and two-way (direction x
    utterance, nested — the widest and the one the paper reports)."""
    atts = sorted({p["attacker"] for p in test_pts})
    by_dir: dict = defaultdict(dict)
    for p in test_pts:
        by_dir[p["dir"]][p["attacker"]] = p
    dirs = sorted(by_dir)
    rng = np.random.default_rng(seed)

    def pooled_from(points):
        x = np.array([p[pred_key] for p in points], float)
        y = np.array([p[response] for p in points], float)
        strata = np.array([p["attacker"] for p in points])
        if len(np.unique(strata)) < 2:
            return None
        return _within_pooled_spearman(x, y, strata)

    obs = pooled_from(test_pts)

    dir_vals = []
    for _ in range(n_boot):
        take = rng.choice(dirs, size=len(dirs), replace=True)
        pts = [by_dir[d][a] for d in take for a in atts if a in by_dir[d]]
        v = pooled_from(pts)
        if v is not None:
            dir_vals.append(v)

    all_spk = sorted({s for d in dirs for s in by_dir[d][atts[0]]["utts"]["speaker"]})
    spk_vals = []
    for _ in range(n_boot):
        keep = set(rng.choice(all_spk, size=len(all_spk), replace=True))
        pts = []
        for d in dirs:
            base = by_dir[d][atts[0]]["utts"]["speaker"]
            sel = [i for i, s in enumerate(base) if s in keep]
            if len(sel) < 2:
                continue
            for a in atts:
                u = by_dir[d][a]["utts"]
                neg = np.array(u["neg_m"])[sel]; pos = np.array(u["pos_m"])[sel]
                pts.append({"attacker": a,
                            pred_key: float(np.mean(np.array(u["sens"])[sel]))
                            if pred_key == "pred_sensitivity"
                            else by_dir[d][a][pred_key],
                            response: auc_oriented(neg, pos)})
        v = pooled_from(pts)
        if v is not None:
            spk_vals.append(v)

    two_vals = []
    for _ in range(n_boot):
        take = rng.choice(dirs, size=len(dirs), replace=True)
        pts = []
        for d in take:
            m = by_dir[d][atts[0]]["n_utts"]
            ridx = rng.integers(0, m, size=m)
            for a in atts:
                u = by_dir[d][a]["utts"]
                neg = np.array(u["neg_m"])[ridx]; pos = np.array(u["pos_m"])[ridx]
                pred_v = (float(np.mean(np.array(u["sens"])[ridx]))
                          if pred_key == "pred_sensitivity" else by_dir[d][a][pred_key])
                pts.append({"attacker": a, pred_key: pred_v,
                            response: auc_oriented(neg, pos)})
        v = pooled_from(pts)
        if v is not None:
            two_vals.append(v)

    def ci(vals):
        return ([float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
                if vals else [None, None])

    return {"observed": obs, "n_speakers": len(all_spk),
            "ci_direction_cluster": ci(dir_vals),
            "ci_speaker_cluster": ci(spk_vals),
            "ci_two_way_dir_x_utt": ci(two_vals)}


def _necessity_test(test_pts, responses=("paired_wave", "paired_mel", "auc_mel",
                                         "auc_wave"),
                    rel_threshold: float = 0.25) -> dict:
    """One-sided implication the theory actually makes: s_W ~ 0 => chance in EVERY
    probe domain. 'Near-null' is ABSOLUTE per attacker (below `rel_threshold` x that
    attacker's median s_W); a waveform codec may have NO near-null direction."""
    out = {"rel_threshold": rel_threshold, "per_attacker_n_low": {}}
    strata = np.array([p["attacker"] for p in test_pts])
    s = np.array([p["pred_sensitivity"] for p in test_pts], float)
    low = np.zeros(len(test_pts), bool)
    for a in np.unique(strata):
        idx = np.flatnonzero(strata == a)
        thr = rel_threshold * float(np.median(s[idx]))
        sel = idx[s[idx] <= thr]
        low[sel] = True
        out["per_attacker_n_low"][a] = int(len(sel))
    for resp in responses:
        y = np.abs(np.array([p[resp] for p in test_pts], float) - 0.5)
        out[resp] = {
            "n_low": int(low.sum()),
            "median_dev_low_sens": float(np.median(y[low])) if low.any() else None,
            "median_dev_rest": float(np.median(y[~low])),
            "max_dev_low_sens": float(np.max(y[low])) if low.any() else None,
        }
    return out
