"""E2 — does channel-relative preserved energy PREDICT survival? (held-out test)

Protocol (revised for the pre-submission fix batch — P0-1/2/3):

  * >=100 random watermark DIRECTIONS (mixtures beta~U[0,1], in-phase random,
    pure nullspace, pure rowspace), each instantiated per-utterance with a keyed
    construction whose seed is a STABLE content hash (SHA-256[:8]) of the clip id
    — not Python's process-salted ``hash`` — so a re-run reproduces byte-for-byte.
  * PERCEPTUAL BUDGET (P0-1): every constructed mark is scaled to a common
    PESQ-WB target (4.2) via bisection (``scale_to_pesq``), NOT a random per-utterance
    SNR. The achieved PESQ, SI-SDR and SNR are recorded per instance and enter the
    analysis as competitor predictors, so null/row/mixture directions are compared
    at the same perceptual budget and any residual perceptual confound is visible.
  * Directions are evaluated on the FIT split (dev-clean) and the TEST split
    (test-clean) — speaker-disjoint by construction.
  * For each (direction, attacker): the channel-relative predictor is that
    attacker's OWN sensitivity s_W = ||A_W(x+d)-A_W(x)||/||d|| (mean over
    utterances); the response is the oriented AUC of the genie separability probe
    over the same utterances. (The probe correlates with the KNOWN perturbation —
    a diagnostic of channel geometry, not a deployable detector; deployed-detector
    behaviour is E1's job.)
  * CLUSTER-AWARE INFERENCE (P0-2). The same direction appears across all five
    attacker strata and the same utterance enters multiple directions, so the
    points are doubly clustered. We therefore:
      - test significance with a permutation that permutes the WHOLE direction as a
        unit, applying ONE common permutation across all attacker strata (not an
        independent within-attacker shuffle, which is anticonservative here);
      - bootstrap the within-attacker rank-pooled Spearman by DIRECTION cluster, by
        SPEAKER cluster, and two-way (direction x utterance/speaker, nested) — the
        widest interval is the one reported.
    Per-utterance predictor / positive-score / negative-score / speaker-id are
    saved so every statistic can be recomputed from the JSON.
  * Competitor predictors under the identical protocol: waveform SNR (achieved),
    achieved PESQ, achieved SI-SDR, STFT magnitude-change fraction, perturbation
    spectral centroid. Permutation test on each.
  * Nullspace/rowspace construction quality (analysis-change ratios vs a random
    control) is measured and reported — never assumed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .analysis import MelAnalysis
from .attackers import build_attackers
from .data_io import clip_uid, iter_split, load_manifest
from .e2_stats import (_cluster_cis, _common_perm_test, _fit_eval,
                       _necessity_test)
from .marks import (DIRECTION_BUILDERS, mel_probe_direction, mixture_direction,
                    probe_score, probe_score_mel, scale_to_pesq, verify_direction)
from .metrics_audio import auc_oriented, auc_raw, si_sdr
from .repro import set_determinism, stable_key

SR = 16_000
ROOT = Path(__file__).resolve().parents[2]
PREDICT_ATTACKERS = ["mel80_gl", "vocos", "encodec6k", "dac", "snac"]
PESQ_TARGET = 4.2

# predictors compared head-to-head (the perceptual-budget trio is now included so
# a spurious correlation driven by residual loudness/quality would be exposed).
PREDS = ["pred_sensitivity", "pred_mel_fraction", "pred_snr_db", "pred_pesq",
         "pred_si_sdr", "pred_stft_mag_fraction", "pred_spectral_centroid"]


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
              rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    rows = {clip_uid(r): r for _, r, _ in iter_split(man, split, args.max_utts)}
    clips = [(clip_uid(r), x) for _, r, x in iter_split(man, split, args.max_utts)]
    device = args.device
    xs = {uid: torch.as_tensor(x, device=device) for uid, x in clips}
    spk = {uid: rows[uid]["speaker"] for uid in xs}
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
        per_att = {a.name: {"neg_w": [], "pos_w": [], "neg_m": [], "pos_m": [],
                            "sens": []} for a in attackers}
        speakers = [spk[uid] for uid in sel]
        # mark-level (attacker-independent) per-utterance quantities
        mel_fracs, snrs, pesqs, sisdrs, mag_fracs, centroids = [], [], [], [], [], []
        for uid in sel:
            x = xs[uid]
            key = 100_000 * (di + 1) + stable_key(uid)      # STABLE (P0-3)
            builder = DIRECTION_BUILDERS[kind]
            d_unit = (mixture_direction(an, x, key, beta) if kind == "mixture"
                      else builder(an, x, key))
            # PESQ-matched budget (P0-1): same perceptual target for every direction
            delta, ach_pesq, ach_snr = scale_to_pesq(x, d_unit, SR, target=PESQ_TARGET)
            x_np = x.cpu().numpy()
            ach_sisdr = si_sdr(x_np, (x + delta).cpu().numpy())
            if len(verif) < 60 and kind in ("nullspace", "rowspace"):
                verif.append({"kind": kind, **verify_direction(an, x, delta)})
            mel_fracs.append(an.mel_fraction(x, delta))
            snrs.append(ach_snr)
            pesqs.append(ach_pesq)
            sisdrs.append(ach_sisdr)
            mag_fracs.append(_stft_mag_fraction(an, x, delta))
            centroids.append(_spectral_centroid(delta.cpu().numpy()))
            dmel_unit = mel_probe_direction(an, x, delta)
            y_marked = (x + delta).cpu().numpy()
            for att in attackers:
                pa = per_att[att.name]
                pa["sens"].append(float(att.sensitivity(x, delta)))
                att_marked = torch.as_tensor(att.apply(y_marked), device=device)
                att_clean = clean_attacked[(att.name, uid)]
                pa["pos_w"].append(probe_score(att_marked, delta))
                pa["neg_w"].append(probe_score(att_clean, delta))
                pa["pos_m"].append(probe_score_mel(an, an.mel(att_marked), dmel_unit))
                pa["neg_m"].append(probe_score_mel(
                    an, clean_attacked_mel[(att.name, uid)], dmel_unit))
        for att in attackers:
            pa = per_att[att.name]
            neg_w, pos_w = np.array(pa["neg_w"]), np.array(pa["pos_w"])
            neg_m, pos_m = np.array(pa["neg_m"]), np.array(pa["pos_m"])
            # TWO complementary responses, BOTH reported in the paper:
            #  * unpaired oriented AUC (auc_mel/auc_wave): detectability of the mark
            #    against host variability across utterances -- operationally faced;
            #  * paired transmission (paired_*): per-utterance marked-vs-clean, which
            #    isolates the deterministically-transmitted pattern (host cancels).
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
                "pred_sensitivity": float(np.mean(pa["sens"])),
                "pred_mel_fraction": float(np.mean(mel_fracs)),
                "pred_snr_db": float(np.mean(snrs)),
                "pred_pesq": float(np.mean(pesqs)),
                "pred_si_sdr": float(np.mean(sisdrs)),
                "pred_stft_mag_fraction": float(np.mean(mag_fracs)),
                "pred_spectral_centroid": float(np.mean(centroids)),
                # per-utterance arrays for cluster-aware bootstrap (P0-2). Indices
                # are aligned to `sel`, so the SAME index means the SAME utterance
                # across all attacker points of this direction.
                "utts": {
                    "speaker": speakers,
                    "sens": [float(v) for v in pa["sens"]],
                    "pos_m": [float(v) for v in pa["pos_m"]],
                    "neg_m": [float(v) for v in pa["neg_m"]],
                    "pos_w": [float(v) for v in pa["pos_w"]],
                    "neg_w": [float(v) for v in pa["neg_w"]],
                    "pesq": [float(v) for v in pesqs],
                    "si_sdr": [float(v) for v in sisdrs],
                    "snr_db": [float(v) for v in snrs],
                },
            })
        if (di + 1) % 10 == 0:
            print(f"[E2:{split}] directions {di + 1}/{args.n_dirs}", flush=True)
    return points, verif



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-dirs", type=int, default=108)
    ap.add_argument("--utts-per-dir", type=int, default=32)
    ap.add_argument("--max-utts", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0,
                    help="determinism seed; also offsets the direction RNGs so "
                         "independent replicates explore different directions")
    ap.add_argument("--strict", action="store_true", default=True)
    ap.add_argument("--no-strict", dest="strict", action="store_false")
    ap.add_argument("--out", default=str(ROOT / "results" / "e2_predictor.json"))
    args = ap.parse_args()

    set_determinism(args.seed)                                   # P0-3
    from .model_lock import verify_all
    verify_all(strict=args.strict)                               # P0-4 (no-op if unpinned)
    man = load_manifest(args.manifest)
    an = MelAnalysis(device=args.device)
    attackers = build_attackers(PREDICT_ATTACKERS, args.device, args.strict)

    rng_fit = np.random.default_rng(101 + 1000 * args.seed)
    rng_test = np.random.default_rng(202 + 1000 * args.seed)
    fit_pts, verif_fit = run_split("fit", man, attackers, an, args, rng_fit)
    test_pts, verif_test = run_split("test", man, attackers, an, args, rng_test)

    out = {
        "n_dirs": args.n_dirs, "utts_per_dir": args.utts_per_dir,
        "seed": args.seed, "pesq_target": PESQ_TARGET,
        "attackers": PREDICT_ATTACKERS,
        "budget_protocol": "scale_to_pesq(target=%.2f); achieved PESQ/SI-SDR/SNR "
                           "recorded per instance and used as competitor predictors"
                           % PESQ_TARGET,
        "construction_verification": verif_fit + verif_test,
        "points_fit": fit_pts, "points_test": test_pts,
        "necessity_test": _necessity_test(test_pts),
    }
    # achieved-budget summary (so the paper can state the realized PESQ/SI-SDR spread)
    ppesq = np.array([p["pred_pesq"] for p in test_pts])
    psi = np.array([p["pred_si_sdr"] for p in test_pts])
    psnr = np.array([p["pred_snr_db"] for p in test_pts])
    out["achieved_budget"] = {
        "pesq": [float(np.percentile(ppesq, q)) for q in (5, 50, 95)],
        "si_sdr": [float(np.percentile(psi, q)) for q in (5, 50, 95)],
        "snr_db": [float(np.percentile(psnr, q)) for q in (5, 50, 95)],
    }
    for resp in ("paired_mel", "paired_wave", "auc_mel", "auc_wave"):
        out[f"evaluation_{resp}"] = {p: _fit_eval(fit_pts, test_pts, p, resp)
                                     for p in PREDS}
        out[f"permutation_{resp}"] = {p: _common_perm_test(test_pts, p, resp)
                                      for p in PREDS}
        by_att = {}
        for a in PREDICT_ATTACKERS:
            f = [p for p in fit_pts if p["attacker"] == a]
            t = [p for p in test_pts if p["attacker"] == a]
            by_att[a] = _fit_eval(f, t, "pred_sensitivity", resp, n_boot=500)
        out[f"by_attacker_{resp}"] = by_att
    # cluster-aware CIs for the primary predictor/response pair (headline number)
    out["cluster_cis_auc_mel"] = _cluster_cis(test_pts, "pred_sensitivity", "auc_mel")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    for resp in ("auc_mel", "paired_mel"):
        pm = out[f"permutation_{resp}"]["pred_sensitivity"]
        by = out[f"by_attacker_{resp}"]
        rhos = {a: round(by[a]["phi_fit"]["spearman"], 2) for a in by}
        print(f"[E2] {resp}: within-pooled rho={pm['observed_within_spearman']:+.3f} "
              f"perm_p={pm['p_value']:.4g} per-attacker={rhos}")
    cc = out["cluster_cis_auc_mel"]
    print(f"[E2] auc_mel cluster CIs: dir={cc['ci_direction_cluster']} "
          f"spk={cc['ci_speaker_cluster']} two-way={cc['ci_two_way_dir_x_utt']}")
    print(f"[E2] achieved budget: {json.dumps(out['achieved_budget'])}")
    print(f"[E2] necessity: {json.dumps(out['necessity_test'])}")
    print(f"[E2] wrote {args.out}")


if __name__ == "__main__":
    main()
