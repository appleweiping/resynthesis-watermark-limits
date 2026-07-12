r"""Watermark constructions and their optimal detectors (linear-Gaussian surrogate).

Two watermarks embed a key-dependent shift :math:`x \mapsto x+\delta` under the same
masking budget:

* **Post-hoc / surface** (:func:`nullspace_watermark`): the perceptually cheapest
  direction — which lives in :math:`\ker(A)`.  Models AudioSeal/WavMark-style embedding
  that hides in inaudible, resynthesis-discarded components.  Survives :math:`W` with
  detection exponent **zero**.
* **Invariant-aligned** (:func:`invariant_aligned_watermark`): the budget-optimal
  *surviving* direction (top generalized eigenvector of :math:`(P, M)`).  Rides through
  :math:`W` at the rate :math:`R^\*`.

Detection is the optimal likelihood-ratio test for a known mean shift in unit-covariance
noise: statistic :math:`t = y^\top \mu` with :math:`\mu=\delta` (before laundering) or
:math:`\mu=P\delta` (after).  Its ROC depends only on the **deflection**
:math:`d = \|\mu\|`, with :math:`\mathrm{AUC} = \Phi(d/\sqrt2)` and minimum Bayes error
:math:`\Phi(-d/2)`.  After :math:`W`, a nullspace watermark has :math:`d=\|P\delta\|=0`
so its AUC collapses to :math:`0.5` — chance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .capacity import surviving_detection_exponent
from .channel import ResynthesisChannel
from .masking import MaskingBudget

__all__ = [
    "nullspace_watermark",
    "invariant_aligned_watermark",
    "detection_deflection",
    "auc_from_deflection",
    "bayes_error_from_deflection",
    "DetectionSim",
    "simulate_auc",
]


def nullspace_watermark(channel: ResynthesisChannel, masking: MaskingBudget) -> np.ndarray:
    r"""Cheapest inaudible direction inside :math:`\ker(A)`, scaled to the budget.

    This is the surface watermark: maximally imperceptible, and provably laundered.
    """
    null_basis = channel.null_basis()
    if null_basis.shape[1] == 0:
        raise ValueError("channel has trivial nullspace; no surface watermark exists")
    v = masking.cheapest_direction(null_basis)
    return masking.scale_to_budget(v)


def invariant_aligned_watermark(
    channel: ResynthesisChannel, masking: MaskingBudget
) -> np.ndarray:
    r"""Budget-optimal *surviving* watermark (top generalized eigenvector of ``(P, M)``)."""
    return surviving_detection_exponent(channel, masking).delta


def detection_deflection(
    channel: ResynthesisChannel, delta: np.ndarray, after: bool
) -> float:
    r"""Deflection :math:`\|\mu\|` of the LRT: :math:`\|P\delta\|` after, :math:`\|\delta\|` before."""
    delta = np.asarray(delta, dtype=float)
    mu = channel.project_invariant(delta) if after else delta
    return float(np.linalg.norm(mu))


def auc_from_deflection(d: float) -> float:
    r"""Closed-form ROC AUC :math:`\Phi(d/\sqrt2)` for a unit-covariance mean shift."""
    return float(norm.cdf(d / np.sqrt(2.0)))


def bayes_error_from_deflection(d: float) -> float:
    r"""Minimum Bayes error :math:`\Phi(-d/2)` for equal-prior detection."""
    return float(norm.cdf(-d / 2.0))


@dataclass(frozen=True)
class DetectionSim:
    auc_empirical: float
    auc_theory: float
    deflection: float
    n_trials: int


def _empirical_auc(stat_h1: np.ndarray, stat_h0: np.ndarray) -> float:
    """Mann–Whitney (rank) estimate of P(stat_h1 > stat_h0)."""
    all_vals = np.concatenate([stat_h1, stat_h0])
    ranks = all_vals.argsort().argsort().astype(float) + 1.0
    r1 = ranks[: stat_h1.size].sum()
    n1, n0 = stat_h1.size, stat_h0.size
    u1 = r1 - n1 * (n1 + 1) / 2.0
    return float(u1 / (n1 * n0))


def simulate_auc(
    channel: ResynthesisChannel,
    delta: np.ndarray,
    n_trials: int = 20_000,
    after: bool = True,
    rng: np.random.Generator | None = None,
) -> DetectionSim:
    r"""Monte-Carlo the optimal detector's AUC before/after laundering.

    H0 draws clean :math:`x`; H1 draws :math:`x+\delta`.  When ``after`` is true both are
    pushed through :math:`W`.  The matched-filter statistic uses the surviving template
    :math:`\mu` so the empirical AUC should match :func:`auc_from_deflection`.
    """
    rng = np.random.default_rng() if rng is None else rng
    delta = np.asarray(delta, dtype=float)
    n = channel.n

    x0 = rng.standard_normal((n_trials, n))
    x1 = rng.standard_normal((n_trials, n)) + delta
    if after:
        y0 = channel.apply(x0, rng=rng)
        y1 = channel.apply(x1, rng=rng)
        mu = channel.project_invariant(delta)
    else:
        y0, y1 = x0, x1
        mu = delta

    t0 = y0 @ mu
    t1 = y1 @ mu
    auc_emp = _empirical_auc(t1, t0)
    d = detection_deflection(channel, delta, after)
    return DetectionSim(auc_emp, auc_from_deflection(d), d, n_trials)
