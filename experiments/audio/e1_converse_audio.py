r"""E1 — converse validation on real speech (GPU).

Part A (collapse table): named watermarks — a surface (nullspace) spread-spectrum mark, an
invariant magnitude mark, and Meta's deployed AudioSeal — through blind resynthesis
attackers (Griffin-Lim; EnCodec at 6/3/1.5 kbps).  Reports detection AUC and TPR@1%FPR
before vs after laundering, AudioSeal payload bit-accuracy, and each mark's invariant-energy
fraction f (STFT magnitude energy / total — the GL-preserved subspace).

Part B (converse curve): a mixture family sweeping f from ~0 (surface) to ~1 (invariant)
through the exact-analysis Griffin-Lim channel.  Post-laundering AUC follows the single
monotone curve Φ(√f · d0) predicted by Theorem 1 — detectability is destroyed in proportion
to the watermark energy the channel discards.

Outputs: results/e1_audio.json and paper figures fig_e1_auc_before_after.pdf,
fig_e1_auc_vs_fraction.pdf.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attackers import build_attackers, MelGriffinLim, StftGriffinLim
from metrics_audio import (
    bit_accuracy,
    detector_survival,
    empirical_auc,
    mel_invariant_fraction,
    snr_db,
    tpr_at_fpr,
)
from watermarks_audio import (
    AudioSealWatermark,
    MagnitudeSpreadSpectrum,
    MixedSpreadSpectrum,
    SurfaceSpreadSpectrum,
)

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "paper" / "figures"
RESDIR = ROOT / "results"
FIGDIR.mkdir(parents=True, exist_ok=True)
RESDIR.mkdir(parents=True, exist_ok=True)

SR = 16000
CLIP_SEC = 3.0


def load_utterances(n: int, root: str) -> list:
    import torchaudio

    ds = torchaudio.datasets.LIBRISPEECH(root=root, url="test-clean", download=False)
    out, target = [], int(CLIP_SEC * SR)
    for i in range(len(ds)):
        wav, sr, *_ = ds[i]
        wav = wav.mean(0).numpy().astype(np.float32)
        if wav.size < target:
            continue
        seg = wav[:target]
        seg = seg / (np.max(np.abs(seg)) + 1e-8) * 0.95
        out.append(seg.astype(np.float32))
        if len(out) >= n:
            break
    return out


def _mean_gap(pos, neg):
    return float(np.mean(pos) - np.mean(neg))


def part_a(utts, watermarks, attackers) -> list:
    records = []
    for wm in watermarks:
        keys = [1000 + i for i in range(len(utts))]
        xwm = [wm.embed(x, k) for x, k in zip(utts, keys)]
        ief = float(np.mean([mel_invariant_fraction(x, w) for x, w in zip(utts, xwm)]))
        snrs = float(np.mean([snr_db(x, w) for x, w in zip(utts, xwm)]))
        s_clean_b = [wm.score(x, k) for x, k in zip(utts, keys)]
        s_wm_b = [wm.score(w, k) for w, k in zip(xwm, keys)]
        auc_b = empirical_auc(s_wm_b, s_clean_b)
        gap_b = _mean_gap(s_wm_b, s_clean_b)
        base = {"watermark": wm.name, "invariant_fraction": ief, "snr_db": snrs,
                "auc_before": auc_b, "tpr1_before": tpr_at_fpr(s_wm_b, s_clean_b)}
        if isinstance(wm, AudioSealWatermark):
            base["bitacc_before"] = float(np.mean(
                [bit_accuracy(*wm.bits(w, k)) for w, k in zip(xwm, keys)]))
        for att in attackers:
            xa = [att.apply(x) for x in utts]
            xwa = [att.apply(w) for w in xwm]
            s_clean_a = [wm.score(x, k) for x, k in zip(xa, keys)]
            s_wm_a = [wm.score(w, k) for w, k in zip(xwa, keys)]
            rec = dict(base)
            rec["attacker"] = att.name
            rec["auc_after"] = empirical_auc(s_wm_a, s_clean_a)
            rec["tpr1_after"] = tpr_at_fpr(s_wm_a, s_clean_a)
            rec["detector_survival"] = detector_survival(gap_b, _mean_gap(s_wm_a, s_clean_a))
            if isinstance(wm, AudioSealWatermark):
                rec["bitacc_after"] = float(np.mean(
                    [bit_accuracy(*wm.bits(w, k)) for w, k in zip(xwa, keys)]))
            records.append(rec)
            print(f"[E1-A] {wm.name:13s} x {att.name:11s} "
                  f"AUC {rec['auc_before']:.3f}->{rec['auc_after']:.3f} f={ief:.3f}")
    return records


def _combined_auc(mag_w, surf_w, mag_c, surf_c):
    """Standardize each detector by the clean population and combine, then AUC."""
    mag_w, surf_w = np.asarray(mag_w), np.asarray(surf_w)
    mag_c, surf_c = np.asarray(mag_c), np.asarray(surf_c)
    mz = (mag_c.mean(), mag_c.std() + 1e-9)
    sz = (surf_c.mean(), surf_c.std() + 1e-9)
    zw = (mag_w - mz[0]) / mz[1] + (surf_w - sz[0]) / sz[1]
    zc = (mag_c - mz[0]) / mz[1] + (surf_c - sz[0]) / sz[1]
    return empirical_auc(zw, zc), _mean_gap(zw, zc)


def part_b(utts, betas, gl) -> list:
    keys = [2000 + i for i in range(len(utts))]
    out = []
    for beta in betas:
        wm = MixedSpreadSpectrum(beta)
        xwm = [wm.embed(x, k) for x, k in zip(utts, keys)]
        f = float(np.mean([mel_invariant_fraction(x, w) for x, w in zip(utts, xwm)]))
        mc_b, sc_b = zip(*[wm.score_components(x, k) for x, k in zip(utts, keys)])
        mw_b, sw_b = zip(*[wm.score_components(w, k) for w, k in zip(xwm, keys)])
        auc_b, _ = _combined_auc(mw_b, sw_b, mc_b, sc_b)
        xa = [gl.apply(x) for x in utts]
        xwa = [gl.apply(w) for w in xwm]
        mc_a, sc_a = zip(*[wm.score_components(x, k) for x, k in zip(xa, keys)])
        mw_a, sw_a = zip(*[wm.score_components(w, k) for w, k in zip(xwa, keys)])
        auc_a, _ = _combined_auc(mw_a, sw_a, mc_a, sc_a)
        out.append({"beta": beta, "invariant_fraction": f,
                    "auc_before": auc_b, "auc_after": auc_a})
        print(f"[E1-B] beta={beta:.2f} f={f:.3f} AUC {auc_b:.3f}->{auc_a:.3f}")
    return out


def make_figures(summary) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    matplotlib.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                                "savefig.bbox": "tight", "savefig.dpi": 300})
    recs = summary["part_a"]
    wms = sorted({r["watermark"] for r in recs})
    atts = sorted({r["attacker"] for r in recs})

    fig, axes = plt.subplots(1, len(wms), figsize=(2.6 * len(wms), 2.5), sharey=True)
    if len(wms) == 1:
        axes = [axes]
    for ax, w in zip(axes, wms):
        rs = sorted([r for r in recs if r["watermark"] == w],
                    key=lambda r: atts.index(r["attacker"]))
        xi = np.arange(len(rs))
        ax.bar(xi - 0.2, [r["auc_before"] for r in rs], 0.4, color="C0", label="before")
        ax.bar(xi + 0.2, [r["auc_after"] for r in rs], 0.4, color="C3", label="after")
        ax.axhline(0.5, color="0.4", lw=0.8, ls=":")
        ax.set_xticks(xi)
        ax.set_xticklabels([r["attacker"] for r in rs], rotation=40, ha="right", fontsize=6)
        ax.set_title(w, fontsize=8)
        ax.set_ylim(0.4, 1.03)
    axes[0].set_ylabel("detection AUC")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_e1_auc_before_after.pdf")
    plt.close(fig)

    pb = summary["part_b"]
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    f = np.array([r["invariant_fraction"] for r in pb])
    a_after = np.array([r["auc_after"] for r in pb])
    a_before = np.array([r["auc_before"] for r in pb])
    d0 = np.sqrt(2) * norm.ppf(min(0.999, float(np.median(a_before))))
    fg = np.linspace(0, 1, 100)
    ax.plot(fg, norm.cdf(np.sqrt(fg) * d0), "-", color="0.5", lw=1.1,
            label=r"$\Phi(\sqrt{f}\,d_0)$", zorder=2)
    ax.scatter(f, a_after, c="C0", s=34, zorder=3, label="mixture (mel-GL)")
    # overlay the named marks under the lossy mel channel
    gl_named = [r for r in recs if r["attacker"].startswith("mel")]
    ax.scatter([r["invariant_fraction"] for r in gl_named],
               [r["auc_after"] for r in gl_named], c="C3", marker="^", s=40,
               zorder=4, label="named marks (GL)")
    ax.axhline(0.5, color="0.4", lw=0.8, ls=":")
    ax.set_xlabel(r"invariant-energy fraction $f=\||S(x{+}\delta)|-|S(x)|\|^2/\|\Delta S\|^2$")
    ax.set_ylabel("detection AUC after laundering")
    ax.set_ylim(0.45, 1.03)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_e1_auc_vs_fraction.pdf")
    plt.close(fig)


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-root", default="/root/autodl-tmp/data")
    ap.add_argument("--n-utt", type=int, default=80)
    ap.add_argument("--quick", action="store_true", help="GL-only, fewer betas")
    args, _ = ap.parse_known_args()

    utts = load_utterances(args.n_utt, args.data_root)
    print(f"[E1] loaded {len(utts)} utterances")

    watermarks = [SurfaceSpreadSpectrum(), MagnitudeSpreadSpectrum()]
    try:
        watermarks.append(AudioSealWatermark(device=args.device))
    except Exception as exc:
        print(f"[E1] AudioSeal unavailable: {exc}")

    if args.quick:
        attackers = [StftGriffinLim(device=args.device), MelGriffinLim(device=args.device)]
    else:
        attackers = build_attackers(device=args.device)
    mel_gl = next(a for a in attackers if a.name.startswith("mel"))
    betas = [0.0, 0.25, 0.5, 0.75, 1.0] if args.quick else [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    pa = part_a(utts, watermarks, attackers)
    pb = part_b(utts, betas, mel_gl)
    summary = {"part_a": pa, "part_b": pb, "n_utt": len(utts), "clip_sec": CLIP_SEC}
    (RESDIR / "e1_audio.json").write_text(json.dumps(summary, indent=2))
    make_figures(summary)
    print("[E1] wrote results/e1_audio.json and figures")
    return summary


if __name__ == "__main__":
    main()
