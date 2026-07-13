"""Paper figures for E2 (held-out predictor validation), from results JSON only.

fig_e2_predictor.pdf : (a) per-attacker scatter of channel-relative sensitivity vs
                       paired mel-probe transmission on the TEST split, with
                       isotonic fits learned on the FIT split; (b) the necessity
                       check: |paired-0.5| for absolute-near-null-sensitivity
                       directions vs the rest, in both probe domains.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ATT_STYLE = {
    "mel80_gl": ("C0", "o", "mel-inv"),
    "vocos": ("C1", "s", "Vocos"),
    "encodec6k": ("C2", "^", "EnC6"),
    "dac": ("C3", "v", "DAC"),
    "snac": ("C4", "D", "SNAC"),
}


def main() -> None:
    d = json.loads((ROOT / "results" / "e2_predictor.json").read_text())
    fit, test = d["points_fit"], d["points_test"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7),
                                   gridspec_kw={"width_ratios": [1.6, 1.0]})
    from sklearn.isotonic import IsotonicRegression

    for att, (c, m, lab) in ATT_STYLE.items():
        ft = [p for p in fit if p["attacker"] == att]
        tt = [p for p in test if p["attacker"] == att]
        xf = np.array([p["pred_sensitivity"] for p in ft])
        yf = np.array([p["auc_mel"] for p in ft])
        xt = np.array([p["pred_sensitivity"] for p in tt])
        yt = np.array([p["auc_mel"] for p in tt])
        ax1.scatter(xt, yt, s=9, alpha=0.55, color=c, marker=m, label=lab,
                    linewidths=0)
        iso = IsotonicRegression(out_of_bounds="clip").fit(xf, yf)
        xs = np.linspace(xf.min(), xf.max(), 120)
        ax1.plot(xs, iso.predict(xs), "-", color=c, lw=1.3, alpha=0.9)
    ax1.axhline(0.5, color="0.4", lw=0.8, ls=":")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"channel-relative sensitivity $s_{\mathcal{W}}$ (log)")
    ax1.set_ylabel("mel-probe detectability (oriented AUC)")
    ax1.set_ylim(0.47, 0.85)
    ax1.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower right",
               handletextpad=0.1, columnspacing=0.6)
    ax1.set_title("(a) test split; isotonic fits from fit split", fontsize=8)

    # (b) necessity: near-null (absolute threshold) vs rest, both probes
    nec = d["necessity_test"]
    thr = nec["rel_threshold"]
    strata = np.array([p["attacker"] for p in test])
    s = np.array([p["pred_sensitivity"] for p in test])
    low = np.zeros(len(test), bool)
    for a in np.unique(strata):
        idx = np.flatnonzero(strata == a)
        low[idx[s[idx] <= thr * np.median(s[idx])]] = True
    data, labels, colors = [], [], []
    for resp, lab in [("auc_mel", "mel"), ("auc_wave", "wav")]:
        y = np.abs(np.array([p[resp] for p in test]) - 0.5)
        if low.any():
            data.append(y[low]); labels.append(f"null\n{lab}"); colors.append("#c6dbef")
        data.append(y[~low]); labels.append(f"rest\n{lab}"); colors.append("#fdd0a2")
    bp = ax2.boxplot(data, tick_labels=labels, widths=0.6, showfliers=False,
                     patch_artist=True)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.9)
    ax2.set_ylabel(r"$|\mathrm{AUC}-0.5|$")
    ax2.tick_params(axis="x", labelsize=7)
    ax2.set_title("(b) near-null $s_\\mathcal{W}$: chance", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG / "fig_e2_predictor.pdf")
    plt.close(fig)
    print("wrote", FIG / "fig_e2_predictor.pdf")


if __name__ == "__main__":
    main()
