"""E2 — does channel-relative preserved energy PREDICT survival? (held-out test)

This replaces the old 6-point same-data fit. Protocol:

  * >=100 random watermark DIRECTIONS (mixtures beta~U[0,1], in-phase random,
    pure nullspace, pure rowspace), each instantiated per-utterance with a keyed
    construction and a randomized budget (SNR ~ U[18,30] dB).
  * Directions are evaluated on the FIT split (dev-clean) and the TEST split
    (test-clean) — speaker-disjoint by construction.
  * For each (direction, attacker): the channel-relative predictor is that
    attacker's OWN sensitivity s_W = ||A_W(x+d)-A_W(x)||/||d|| (mean over
    utterances); the response is the oriented AUC of the genie separability probe
    over the same utterances. (The probe correlates with the KNOWN perturbation —
    a diagnostic of channel geometry, not a deployable detector; deployed-detector
    behaviour is E1's job.)
  * The survival mapping (isotonic, and 2-parameter Phi(a*sqrt(f)+b)) is fit ONLY
    on fit-split points and evaluated ONLY on test-split points: Spearman, R2,
    RMSE with bootstrap CIs over directions.
  * Competitor predictors under the identical protocol: waveform SNR, STFT
    magnitude-change fraction, spectral centroid of the perturbation.
  * Permutation test: predictor values shuffled within attacker strata.
  * Nullspace/rowspace construction quality (analysis-change ratios vs a random
    control) is measured and reported — never assumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .analysis import MelAnalysis
from .attackers import build_attackers
from .data_io import clip_uid, iter_split, load_manifest
from .marks import (DIRECTION_BUILDERS, mel_probe_direction, mixture_direction,
                    probe_score, probe_score_mel, scale_to_snr, verify_direction)
from .metrics_audio import auc_oriented, auc_raw

SR = 16_000
ROOT = Path(__file__).resolve().parents[2]
PREDICT_ATTACKERS = ["mel80_gl", "vocos", "encodec6k", "dac", "snac"]


def _direction_spec(i: int, rng: np.random.Generator) -> tuple[str, float]:
    u = rng.random()
    if u < 0.60:
        return "mixture", float(rng.random())
    if u < 0.80:
        return "inphase_rand", 0.5
    if u < 0.90:
        return "nullspace", 0.0
    return "rowspace", 1.0


def _spectral_centroid(delta: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(delta))
    freqs = np.fft.rfftfreq(len(delta), 1.0 / SR)
    return float(np.sum(freqs * spec) / (np.sum(spec) + 1e-12))


def _stft_mag_fraction(an: MelAnalysis, x: torch.Tensor, d: torch.Tensor) -> float:
    X0, X1 = an.stft(x), an.stft(x + d)
    dmag = float(torch.sum((X1.abs() - X0.abs()) ** 2))
    dcplx = float(torch.sum((X1 - X0).abs() ** 2)) + 1e-20
    return dmag / dcplx


def run_split(split: str, man: dict, attackers, an: MelAnalysis, args,
              rng: np.random.Generator) -> list[dict]:
    clips = [(clip_uid(r), x) for _, r, x in iter_split(man, split, args.max_utts)]
    device = args.device
    xs = {uid: torch.as_tensor(x, device=device) for uid, x in clips}
    uids = list(xs)

    # cache attacked CLEAN clips (waveform + unified-mel) per (attacker, uid)
    clean_attacked: dict = {}
    clean_attacked_mel: dict = {}
    for att in attackers:
        for uid in uids:
            ya = torch.as_tensor(att.apply(xs[uid].cpu().numpy()), device=device)
            clean_attacked[(att.name, uid)] = ya
            clean_attacked_mel[(att.name, uid)] = an.mel(ya)
        print(f"[E2:{split}] cached clean-attacked: {att.name}", flush=True)

    points, verif = [], []
    for di in range(args.n_dirs):
        kind, beta = _direction_spec(di, rng)
        # each direction sees m utterances (rotating window over the split)
        sel = [uids[(di * 7 + j) % len(uids)] for j in range(args.utts_per_dir)]
        sel = list(dict.fromkeys(sel))
        per_att = {a.name: {"neg_w": [], "pos_w": [], "neg_m": [], "pos_m": []}
                   for a in attackers}
        sens = {a.name: [] for a in attackers}
        mel_fracs, snrs, mag_fracs, centroids = [], [], [], []
        for uid in sel:
            x = xs[uid]
            key = 100_000 * (di + 1) + hash(uid) % 50_000
            builder = DIRECTION_BUILDERS[kind]
            d_unit = (mixture_direction(an, x, key, beta) if kind == "mixture"
                      else builder(an, x, key))
            snr = float(rng.uniform(18.0, 30.0))
            delta = scale_to_snr(x, d_unit, snr)
            if len(verif) < 60 and kind in ("nullspace", "rowspace"):
                verif.append({"kind": kind,
                              **verify_direction(an, x, delta)})
            mel_fracs.append(an.mel_fraction(x, delta))
            snrs.append(snr)
            mag_fracs.append(_stft_mag_fraction(an, x, delta))
            centroids.append(_spectral_centroid(delta.cpu().numpy()))
            dmel_unit = mel_probe_direction(an, x, delta)
            y_marked = (x + delta).cpu().numpy()
            for att in attackers:
                sens[att.name].append(att.sensitivity(x, delta))
                att_marked = torch.as_tensor(att.apply(y_marked), device=device)
                att_clean = clean_attacked[(att.name, uid)]
                pa = per_att[att.name]
                pa["pos_w"].append(probe_score(att_marked, delta))
                pa["neg_w"].append(probe_score(att_clean, delta))
                pa["pos_m"].append(probe_score_mel(an, an.mel(att_marked), dmel_unit))
                pa["neg_m"].append(probe_score_mel(
                    an, clean_attacked_mel[(att.name, uid)], dmel_unit))
        for att in attackers:
            pa = per_att[att.name]
            neg_w, pos_w = np.array(pa["neg_w"]), np.array(pa["pos_w"])
            neg_m, pos_m = np.array(pa["neg_m"]), np.array(pa["pos_m"])
            # PAIRED transmission statistics: attacks are deterministic and the
            # probe is evaluated on the same utterance marked vs clean, so the
            # per-utterance score difference isolates the transmitted pattern
            # (unpaired AUC drowns it in host variability across utterances).
            paired_m = float(np.mean(pos_m > neg_m) + 0.5 * np.mean(pos_m == neg_m))
            paired_w = float(np.mean(pos_w > neg_w) + 0.5 * np.mean(pos_w == neg_w))
            points.append({
                "split": split, "dir": di, "kind": kind, "beta": beta,
                "attacker": att.name, "n_utts": len(sel),
                "auc_raw_wave": auc_raw(neg_w, pos_w),
                "auc_wave": auc_oriented(neg_w, pos_w),
                "auc_raw_mel": auc_raw(neg_m, pos_m),
                "auc_mel": auc_oriented(neg_m, pos_m),
                "paired_mel": max(paired_m, 1.0 - paired_m),
                "paired_wave": max(paired_w, 1.0 - paired_w),
                "pred_sensitivity": float(np.mean(sens[att.name])),
                "pred_mel_fraction": float(np.mean(mel_fracs)),
                "pred_snr_db": float(np.mean(snrs)),
                "pred_stft_mag_fraction": float(np.mean(mag_fracs)),
                "pred_spectral_centroid": float(np.mean(centroids)),
            })
        if (di + 1) % 10 == 0:
            print(f"[E2:{split}] directions {di + 1}/{args.n_dirs}", flush=True)
    return points, verif


# ---------- fitting & evaluation ---------------------------------------------------
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

    # 2-parameter Phi(a*sqrt(f)+b) fit on FIT split only
    model = lambda f, a, b: snorm.cdf(a * np.sqrt(np.clip(f, 0, 1)) + b)
    try:
        (a_hat, b_hat), _ = curve_fit(model, xf, np.clip(yf, 1e-3, 1 - 1e-3),
                                      p0=[3.0, -1.5], maxfev=20000)
        yhat = model(xt, a_hat, b_hat)
    except Exception:
        a_hat = b_hat = float("nan")
        yhat = np.full_like(yt, np.mean(yf))
    # isotonic on FIT, predict TEST
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

    # bootstrap CI over test DIRECTIONS for spearman
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


def _within_rank(vals: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """Rank-normalize within each stratum (attacker) -> comparable across strata."""
    from scipy import stats as sps

    out = np.empty_like(vals, dtype=float)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        out[idx] = sps.rankdata(vals[idx]) / (len(idx) + 1.0)
    return out


def _permutation_test(test_pts, pred_key: str, response: str = "auc_oriented",
                      n_perm: int = 1000, seed: int = 0) -> dict:
    """WITHIN-ATTACKER pooled Spearman with a within-attacker permutation null.

    Both predictor and response are rank-normalized within attacker before
    pooling, so between-attacker level differences cannot create or mask an
    association (the earlier pooled statistic did exactly that)."""
    from scipy import stats as sps

    x = np.array([p[pred_key] for p in test_pts], float)
    y = np.array([p[response] for p in test_pts], float)
    strata = np.array([p["attacker"] for p in test_pts])
    xr, yr = _within_rank(x, strata), _within_rank(y, strata)
    obs = float(sps.spearmanr(xr, yr).statistic)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        xp = x.copy()
        for s in np.unique(strata):
            idx = np.flatnonzero(strata == s)
            xp[idx] = xp[rng.permutation(idx)]
        null.append(sps.spearmanr(_within_rank(xp, strata), yr).statistic)
    null = np.array(null)
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
    return {"observed_within_spearman": obs, "p_value": p,
            "null_abs_95": float(np.percentile(np.abs(null), 95))}


def _necessity_test(test_pts, responses=("paired_wave", "paired_mel"),
                    rel_threshold: float = 0.25) -> dict:
    """One-sided implication the theory actually makes: s_W ~ 0 => chance in
    EVERY probe domain. 'Near-null' must be ABSOLUTE, not a per-attacker decile:
    a direction counts as near-null for attacker W only if its sensitivity is
    below `rel_threshold` x the median s_W of that attacker (waveform codecs may
    have NO near-null direction — itself a finding: their analysis kernel is
    tiny, so nothing is reliably erased)."""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-dirs", type=int, default=120)
    ap.add_argument("--utts-per-dir", type=int, default=40)
    ap.add_argument("--max-utts", type=int, default=400)
    ap.add_argument("--strict", action="store_true", default=True)
    ap.add_argument("--no-strict", dest="strict", action="store_false")
    ap.add_argument("--out", default=str(ROOT / "results" / "e2_predictor.json"))
    args = ap.parse_args()

    man = load_manifest(args.manifest)
    an = MelAnalysis(device=args.device)
    attackers = build_attackers(PREDICT_ATTACKERS, args.device, args.strict)

    rng_fit = np.random.default_rng(101)
    rng_test = np.random.default_rng(202)
    fit_pts, verif_fit = run_split("fit", man, attackers, an, args, rng_fit)
    test_pts, verif_test = run_split("test", man, attackers, an, args, rng_test)

    preds = ["pred_sensitivity", "pred_mel_fraction", "pred_snr_db",
             "pred_stft_mag_fraction", "pred_spectral_centroid"]
    out = {
        "n_dirs": args.n_dirs, "utts_per_dir": args.utts_per_dir,
        "attackers": PREDICT_ATTACKERS,
        "construction_verification": verif_fit + verif_test,
        "points_fit": fit_pts, "points_test": test_pts,
        "necessity_test": _necessity_test(test_pts),
    }
    for resp in ("paired_mel", "paired_wave", "auc_mel", "auc_wave"):
        out[f"evaluation_{resp}"] = {p: _fit_eval(fit_pts, test_pts, p, resp)
                                     for p in preds}
        out[f"permutation_{resp}"] = {p: _permutation_test(test_pts, p, resp)
                                      for p in preds}
        # per-attacker within-evaluation for the primary predictor
        by_att = {}
        for a in PREDICT_ATTACKERS:
            f = [p for p in fit_pts if p["attacker"] == a]
            t = [p for p in test_pts if p["attacker"] == a]
            by_att[a] = _fit_eval(f, t, "pred_sensitivity", resp, n_boot=500)
        out[f"by_attacker_{resp}"] = by_att
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    for resp in ("paired_mel", "paired_wave"):
        pm = out[f"permutation_{resp}"]["pred_sensitivity"]
        by = out[f"by_attacker_{resp}"]
        rhos = {a: round(by[a]["phi_fit"]["spearman"], 2) for a in by}
        print(f"[E2] {resp}: within-pooled rho={pm['observed_within_spearman']:+.3f} "
              f"perm_p={pm['p_value']:.4g} per-attacker={rhos}")
    print(f"[E2] necessity: {json.dumps(out['necessity_test'])}")
    print(f"[E2] wrote {args.out}")


if __name__ == "__main__":
    main()
