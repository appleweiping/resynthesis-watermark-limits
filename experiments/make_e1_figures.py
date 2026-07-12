"""Regenerate the E1 converse-curve figure from results/e1_audio.json (CPU, no torch).

Plots post-laundering detection AUC vs. the invariant-energy fraction f for the
surface->invariant mixture and the named marks (under the lossy mel-vocoder), and overlays
a two-parameter converse fit AUC = Phi(a*sqrt(f) + b): the sqrt(f) functional form of
Theorem 1 with the detector gain a and threshold b fit to the data.  Honest: the fit has
two free parameters; the *claim* is monotonicity in f, which the fit captures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                                "savefig.bbox": "tight", "savefig.dpi": 300})

    d = json.loads((ROOT / "results" / "e1_audio.json").read_text())
    pb = d["part_b"]

    # Fixed-detector mixture sweep only: f predicts survival for ONE matched detector.
    # (Named marks use different detectors and are NOT comparable on this curve.)
    f_mix = np.array([r["invariant_fraction"] for r in pb])
    a_mix = np.clip(np.array([r["auc_after"] for r in pb]), 1e-3, 1 - 1e-3)
    model = lambda f, a, b: norm.cdf(a * np.sqrt(f) + b)
    (a_hat, b_hat), _ = curve_fit(model, f_mix, a_mix, p0=[3.0, -2.0], maxfev=10000)
    fg = np.linspace(f_mix.min() * 0.95, 1.0, 100)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(fg, model(fg, a_hat, b_hat), "-", color="0.5", lw=1.1,
            label=r"$\Phi(a\sqrt{f}+b)$ fit", zorder=2)
    ax.scatter(f_mix, a_mix, c="C0", s=36, zorder=3, label="mixture, fixed detector")
    ax.axhline(0.5, color="0.4", lw=0.8, ls=":")
    ax.set_xlabel(r"invariant-energy fraction $f$")
    ax.set_ylabel("detection AUC after mel-inversion")
    ax.set_ylim(0.45, 1.02)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_e1_auc_vs_fraction.pdf")
    plt.close(fig)
    print(f"fit a={a_hat:.2f} b={b_hat:.2f} on {len(f_mix)} mixture pts; wrote fig_e1_auc_vs_fraction.pdf")

    # ---- fig_e2: payload bit-accuracy, clean vs after mel-inversion, vs SNR ----
    # Survival = clean (dashed) ~ after (solid). Invariant's modest clean rate shown honestly.
    e2 = json.loads((ROOT / "results" / "e2_audio.json").read_text())
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for wm, (c, lab) in {"magnitude": ("C0", "invariant"), "surface": ("C3", "surface")}.items():
        cur = sorted(e2["curves"][wm], key=lambda r: r["snr_db"])
        snr = [r["snr_db"] for r in cur]
        ax.plot(snr, [r["bitacc_clean"] for r in cur], "--", color=c, lw=1.0, alpha=0.7)
        ax.plot(snr, [r["bitacc_gl"] for r in cur], "-o", color=c, ms=4,
                label=f"{lab} (after)")
    ax.axhline(0.5, color="0.4", lw=0.8, ls=":")
    ax.plot([], [], "--", color="0.5", label="clean (dashed)")
    ax.set_xlabel("watermark SNR (dB) $\\;\\leftarrow$ more budget")
    ax.set_ylabel("payload bit-accuracy")
    ax.invert_xaxis()
    ax.set_ylim(0.45, 1.03)
    ax.legend(frameon=False, fontsize=7, loc="center left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_e2_rate_survival.pdf")
    plt.close(fig)
    print("wrote fig_e2_rate_survival.pdf")


if __name__ == "__main__":
    main()
