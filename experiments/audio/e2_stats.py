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


def _restricted_permutation(rng, kind_of):
    """A permutation of the direction axis that only moves directions WITHIN their
    kind block (P0-B.2), so the null does not derive significance from the fixed
    nullspace/rowspace/mixture composition. Falls back to a free permutation when
    no kind labels are supplied."""
    n = len(kind_of)
    perm = np.arange(n)
    if kind_of[0] is None:
        return rng.permutation(n)
    for k in set(kind_of):
        idx = np.flatnonzero(np.array(kind_of) == k)
        perm[idx] = idx[rng.permutation(len(idx))]
    return perm


def _common_perm_test(test_pts, pred_key: str, response: str,
                      n_perm: int = 2000, seed: int = 0,
                      restrict_kind: bool = True) -> dict:
    """WITHIN-attacker rank-pooled Spearman with a COMMON whole-direction permutation
    null (P0-2/P0-B.2). One permutation of the direction axis is applied across every
    attacker stratum (a direction moves as one unit — the cluster shared by all
    attackers); an independent within-attacker shuffle breaks that coupling and is
    anticonservative. With ``restrict_kind`` the permutation is restricted to within
    kind blocks, so the nullspace/rowspace/mixture composition cannot itself create
    significance."""
    from scipy import stats as sps

    atts = sorted({p["attacker"] for p in test_pts})
    dirs = sorted({p["dir"] for p in test_pts})
    ai = {a: i for i, a in enumerate(atts)}
    di = {d: i for i, d in enumerate(dirs)}
    X = np.full((len(atts), len(dirs)), np.nan)
    Y = np.full((len(atts), len(dirs)), np.nan)
    kind_of = [None] * len(dirs)
    for p in test_pts:
        X[ai[p["attacker"]], di[p["dir"]]] = p[pred_key]
        Y[ai[p["attacker"]], di[p["dir"]]] = p[response]
        if restrict_kind and "kind" in p:
            kind_of[di[p["dir"]]] = p["kind"]
    if np.isnan(X).any() or np.isnan(Y).any():
        keep = [j for j in range(len(dirs))
                if not (np.isnan(X[:, j]).any() or np.isnan(Y[:, j]).any())]
        X, Y = X[:, keep], Y[:, keep]
        kind_of = [kind_of[j] for j in keep]

    def pooled(xm, ym):
        xr = np.vstack([sps.rankdata(xm[r]) / (xm.shape[1] + 1.0)
                        for r in range(xm.shape[0])])
        yr = np.vstack([sps.rankdata(ym[r]) / (ym.shape[1] + 1.0)
                        for r in range(ym.shape[0])])
        return float(sps.spearmanr(xr.ravel(), yr.ravel()).statistic)

    obs = pooled(X, Y)
    rng = np.random.default_rng(seed)
    null = np.array([pooled(X[:, _restricted_permutation(rng, kind_of)], Y)
                     for _ in range(n_perm)])
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
    return {"observed_within_spearman": obs, "p_value": p,
            "null_abs_95": float(np.percentile(np.abs(null), 95)),
            "n_perm": n_perm,
            "null_type": ("common whole-direction permutation across attackers"
                          + (", restricted within kind blocks" if restrict_kind
                             and kind_of[0] is not None else ""))}


def _cmp_matrix(u: dict) -> np.ndarray:
    """C[i,j] = 1 if pos_i > neg_j, 0.5 if equal — so a WEIGHTED raw AUC is the
    quadratic form w C w / (sum w)^2 (utterance i's pos and neg share weight w_i)."""
    pos = np.asarray(u["pos_m"], float)
    neg = np.asarray(u["neg_m"], float)
    return ((pos[:, None] > neg[None, :]).astype(float)
            + 0.5 * (pos[:, None] == neg[None, :]))


def _weighted_oriented_auc(C: np.ndarray, w: np.ndarray) -> float | None:
    sw = w.sum()
    if sw <= 0:
        return None
    raw = float(w @ C @ w) / (sw * sw)
    return max(raw, 1.0 - raw)


def _pigeonhole_ci(by_dir, atts, pred_key, response, n_boot, seed,
                   resample_dirs: bool, unit_key: str | None):
    """Valid multiway cluster bootstrap (Owen 2007 pigeonhole; Cameron–Gelbach–Miller
    multiway) of the within-attacker rank-pooled Spearman (P0-B).

    A DIRECTION is one cluster (a whole row across all attackers); an UTTERANCE (or
    SPEAKER) is a second, crossed cluster shared across every direction it appears in.
    Direction multiplicity `w_d` is applied to all five attacker points of that
    direction (same weight across attackers); the global unit weights `v_u` weight
    utterance u identically wherever it occurs (same weight across directions). This
    replaces the earlier implementation that collapsed the speaker resample to a
    ``set`` and so silently dropped cluster multiplicity.

      resample_dirs=True, unit_key=None      -> direction-cluster CI
      resample_dirs=False, unit_key="speaker"-> speaker-cluster CI
      resample_dirs=True, unit_key="uid"     -> two-way (direction x utterance) CI
    """
    dirs = sorted(by_dir)
    prep = {}
    for d in dirs:
        for a in atts:
            u = by_dir[d][a]["utts"]
            prep[(d, a)] = (_cmp_matrix(u), np.asarray(u["sens"], float),
                            list(u[unit_key]) if unit_key else None)
    if unit_key:
        all_units = sorted({uu for d in dirs
                            for uu in by_dir[d][atts[0]]["utts"][unit_key]})
        uidx = {uu: i for i, uu in enumerate(all_units)}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        dcount = (np.bincount(rng.integers(0, len(dirs), len(dirs)),
                              minlength=len(dirs)) if resample_dirs
                  else np.ones(len(dirs), int))
        if unit_key:
            ucount = np.bincount(rng.integers(0, len(all_units), len(all_units)),
                                 minlength=len(all_units)).astype(float)
        xs, ys, strata = [], [], []
        for di, d in enumerate(dirs):
            wd = int(dcount[di])
            if wd == 0:
                continue
            for a in atts:
                C, sens, units = prep[(d, a)]
                if unit_key:
                    w = ucount[[uidx[uu] for uu in units]]
                    sw = w.sum()
                    if sw <= 0:
                        continue
                    pv = (float(w @ sens / sw) if pred_key == "pred_sensitivity"
                          else by_dir[d][a][pred_key])
                    rv = _weighted_oriented_auc(C, w)
                else:
                    pv = by_dir[d][a][pred_key]
                    rv = by_dir[d][a][response]
                if rv is None:
                    continue
                xs.extend([pv] * wd); ys.extend([rv] * wd); strata.extend([a] * wd)
        if len(set(strata)) < 2:
            continue
        v = _within_pooled_spearman(np.array(xs), np.array(ys), np.array(strata))
        if not np.isnan(v):
            vals.append(v)
    return vals


def _cluster_cis(test_pts, pred_key: str = "pred_sensitivity",
                 response: str = "auc_mel", n_boot: int = 2000,
                 seed: int = 0) -> dict:
    """Bootstrap CIs of the within-attacker rank-pooled Spearman under three valid
    clusterings (P0-B): by direction, by speaker, and a valid two-way
    (direction x utterance) pigeonhole bootstrap."""
    atts = sorted({p["attacker"] for p in test_pts})
    by_dir: dict = defaultdict(dict)
    for p in test_pts:
        by_dir[p["dir"]][p["attacker"]] = p

    x = np.array([p[pred_key] for p in test_pts], float)
    y = np.array([p[response] for p in test_pts], float)
    strata = np.array([p["attacker"] for p in test_pts])
    obs = _within_pooled_spearman(x, y, strata)

    dir_vals = _pigeonhole_ci(by_dir, atts, pred_key, response, n_boot, seed,
                              resample_dirs=True, unit_key=None)
    spk_vals = _pigeonhole_ci(by_dir, atts, pred_key, response, n_boot, seed + 1,
                              resample_dirs=False, unit_key="speaker")
    two_vals = _pigeonhole_ci(by_dir, atts, pred_key, response, n_boot, seed + 2,
                              resample_dirs=True, unit_key="uid")
    all_spk = sorted({s for d in by_dir for s in by_dir[d][atts[0]]["utts"]["speaker"]})

    def diag(vals):
        if not vals:
            return {"ci": [None, None], "mean": None, "sd": None, "n": 0}
        a = np.array(vals)
        return {"ci": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
                "mean": float(a.mean()), "sd": float(a.std()), "n": int(a.size)}

    d_dir, d_spk, d_two = diag(dir_vals), diag(spk_vals), diag(two_vals)
    return {"observed": obs, "n_speakers": len(all_spk), "n_directions": len(by_dir),
            "ci_direction_cluster": d_dir["ci"],
            "ci_speaker_cluster": d_spk["ci"],
            "ci_two_way_dir_x_utt": d_two["ci"],
            "bootstrap_diagnostics": {"direction": d_dir, "speaker": d_spk,
                                      "two_way": d_two},
            "method": "Owen pigeonhole two-way cluster bootstrap; direction and "
                      "utterance/speaker are crossed global clusters"}


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
