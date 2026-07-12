r"""E0 — surrogate demonstration of the two theorems (CPU-only).

Produces the paper's core theory figures directly from the linear-Gaussian surrogate:

  fig_converse_collapse : post-laundering detection AUC vs the fraction of watermark
                          energy placed in the analysis nullspace (Theorem 1 in action).
  fig_rate_survival     : surviving detection exponent and payload rate R* vs the
                          imperceptibility budget, post-hoc vs invariant-aligned
                          (Theorem 1 floor meets Theorem 2 achievable rate).
  fig_theory_vs_sim     : simulated detector AUC vs the closed-form prediction across
                          many random channels (the "theory predicts the numbers" check).

All numbers are also written to results/e0_surrogate.json.
"""

from __future__ import annotations

import numpy as np

from rwl.capacity import invariant_subchannel_capacity, surviving_detection_exponent
from rwl.channel import ResynthesisChannel
from rwl.masking import MaskingBudget
from rwl.watermark import (
    auc_from_deflection,
    detection_deflection,
    invariant_aligned_watermark,
    nullspace_watermark,
    simulate_auc,
)

from _common import save_fig, save_json, use_paper_style

SEED = 20260712


def converse_collapse(rng: np.random.Generator) -> dict:
    """AUC after laundering as the watermark drifts from invariant into the nullspace."""
    ch = ResynthesisChannel.from_random(n=32, k=10, rng=rng)
    mask = MaskingBudget.isotropic(ch.n, D=3.0)
    row_dir = ch.row_basis()[:, 0]
    null_dir = ch.null_basis()[:, 0]

    alphas = np.linspace(0.0, 1.0, 11)
    auc_after_emp, auc_after_thy, auc_before = [], [], []
    for a in alphas:
        d = np.sqrt(1 - a) * row_dir + np.sqrt(a) * null_dir
        d = mask.scale_to_budget(d)
        auc_after_emp.append(simulate_auc(ch, d, 60_000, after=True, rng=rng).auc_empirical)
        auc_after_thy.append(auc_from_deflection(detection_deflection(ch, d, after=True)))
        auc_before.append(auc_from_deflection(detection_deflection(ch, d, after=False)))
    return {
        "alpha": alphas.tolist(),
        "auc_after_emp": auc_after_emp,
        "auc_after_theory": auc_after_thy,
        "auc_before_theory": auc_before,
    }


def rate_survival(rng: np.random.Generator) -> dict:
    """Surviving exponent and R* vs budget, for post-hoc vs invariant-aligned watermarks."""
    ch = ResynthesisChannel.from_random(n=32, k=10, rng=rng)
    w = np.exp(0.6 * rng.standard_normal(ch.n))  # non-isotropic masking
    Ds = np.geomspace(0.1, 10.0, 12)

    exp_inv, exp_post, rstar_inv, rstar_post = [], [], [], []
    for D in Ds:
        mask = MaskingBudget(np.diag(w), D=float(D))
        exp_inv.append(surviving_detection_exponent(ch, mask).exponent)
        # Post-hoc = cheapest nullspace direction: surviving exponent is identically 0.
        post_delta = nullspace_watermark(ch, mask)
        exp_post.append(0.125 * float(post_delta @ (ch.P @ post_delta)))
        rstar_inv.append(invariant_subchannel_capacity(ch, mask).R_star)
        rstar_post.append(0.0)  # nullspace payload after laundering
    return {
        "D": Ds.tolist(),
        "exponent_invariant": exp_inv,
        "exponent_posthoc": exp_post,
        "Rstar_invariant_nats": rstar_inv,
        "Rstar_posthoc_nats": rstar_post,
    }


def theory_vs_sim(rng: np.random.Generator) -> dict:
    """Empirical detector AUC vs closed-form prediction across random channels/watermarks."""
    emp, thy = [], []
    for _ in range(60):
        n = int(rng.integers(10, 40))
        k = int(rng.integers(3, n - 1))
        ch = ResynthesisChannel.from_random(n=n, k=k, rng=rng)
        mask = MaskingBudget.isotropic(n, D=float(rng.uniform(0.5, 4.0)))
        delta = (invariant_aligned_watermark(ch, mask)
                 if rng.random() < 0.5 else nullspace_watermark(ch, mask))
        after = bool(rng.integers(0, 2))
        sim = simulate_auc(ch, delta, 40_000, after=after, rng=rng)
        emp.append(sim.auc_empirical)
        thy.append(auc_from_deflection(detection_deflection(ch, delta, after)))
    emp, thy = np.array(emp), np.array(thy)
    return {
        "auc_empirical": emp.tolist(),
        "auc_theory": thy.tolist(),
        "max_abs_err": float(np.max(np.abs(emp - thy))),
        "rmse": float(np.sqrt(np.mean((emp - thy) ** 2))),
    }


def _plot_converse(data: dict):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    a = np.array(data["alpha"])
    ax.plot(a, data["auc_before_theory"], "--", color="0.5", label="before laundering")
    ax.plot(a, data["auc_after_theory"], "-", color="C0", label="after (theory)")
    ax.plot(a, data["auc_after_emp"], "o", color="C0", ms=4, label="after (sim)")
    ax.axhline(0.5, color="C3", lw=0.9, ls=":")
    ax.set_xlabel(r"nullspace energy fraction $\alpha$")
    ax.set_ylabel("detection AUC")
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc="lower left", frameon=False)
    return fig


def _plot_rate_survival(data: dict):
    import matplotlib.pyplot as plt

    D = np.array(data["D"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.5))
    ax1.semilogx(D, data["exponent_invariant"], "-o", ms=3, color="C0",
                 label="invariant-aligned")
    ax1.semilogx(D, data["exponent_posthoc"], "-s", ms=3, color="C3",
                 label="post-hoc (nullspace)")
    ax1.set_xlabel("imperceptibility budget $D$")
    ax1.set_ylabel("surviving Chernoff exponent")
    ax1.legend(loc="upper left", frameon=False)

    ax2.semilogx(D, data["Rstar_invariant_nats"], "-o", ms=3, color="C0",
                 label=r"invariant $R^*$")
    ax2.semilogx(D, data["Rstar_posthoc_nats"], "-s", ms=3, color="C3",
                 label="post-hoc (=0)")
    ax2.set_xlabel("imperceptibility budget $D$")
    ax2.set_ylabel("surviving rate $R^*$ (nats/use)")
    ax2.legend(loc="upper left", frameon=False)
    return fig


def _plot_theory_vs_sim(data: dict):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    ax.plot([0.5, 1.0], [0.5, 1.0], "-", color="0.6", lw=0.9)
    ax.plot(data["auc_theory"], data["auc_empirical"], "o", ms=3, color="C0", alpha=0.7)
    ax.set_xlabel("AUC (closed form)")
    ax.set_ylabel("AUC (simulation)")
    ax.set_title(f"max err {data['max_abs_err']:.3f}, RMSE {data['rmse']:.3f}")
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.48, 1.02)
    return fig


def main() -> dict:
    use_paper_style()
    rng = np.random.default_rng(SEED)

    conv = converse_collapse(rng)
    rate = rate_survival(rng)
    tvs = theory_vs_sim(rng)

    save_fig(_plot_converse(conv), "fig_converse_collapse.pdf")
    save_fig(_plot_rate_survival(rate), "fig_rate_survival.pdf")
    save_fig(_plot_theory_vs_sim(tvs), "fig_theory_vs_sim.pdf")

    summary = {"converse_collapse": conv, "rate_survival": rate, "theory_vs_sim": tvs}
    save_json(summary, "e0_surrogate.json")

    print("[E0] converse: AUC(alpha=1) after =",
          round(conv["auc_after_emp"][-1], 4), "(chance)")
    print("[E0] converse: AUC(alpha=0) after =",
          round(conv["auc_after_emp"][0], 4), "(survives)")
    print("[E0] rate: R*_invariant(maxD) =",
          round(rate["Rstar_invariant_nats"][-1], 4), "nats; post-hoc = 0")
    print("[E0] theory-vs-sim: max abs err =", round(tvs["max_abs_err"], 4),
          "RMSE =", round(tvs["rmse"], 4))
    return summary


if __name__ == "__main__":
    main()
