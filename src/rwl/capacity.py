r"""Achievability: what survives the resynthesis channel, and at what rate.

Two survivable quantities, both under the masking budget :math:`\delta^\top M\delta\le D`:

1. **Surviving detection exponent** (zero-rate / one-bit provenance).  The post-laundering
   Chernoff information is :math:`\tfrac18\|P\delta\|^2 = \tfrac18\,\delta^\top P\delta`
   (:math:`P` is an orthogonal projector, so :math:`P^\top P = P`).  Maximizing it under
   the budget is a generalized Rayleigh quotient

   .. math::
       \max_{\delta^\top M\delta \le D} \tfrac18\,\delta^\top P\delta
       = \tfrac{D}{8}\,\lambda_{\max}(P, M),

   attained at the top generalized eigenvector of :math:`(P, M)`.  Restricting
   :math:`\delta\in\ker(A)` forces the quotient to :math:`0` — the post-hoc watermark
   provably dies; the optimizer necessarily has an invariant component that survives.

2. **Invariant sub-channel capacity** :math:`R^\*` (multi-bit payload).  Writing the
   surviving shift as :math:`u = P\delta` in an orthonormal basis of
   :math:`\mathrm{row}(A)`, a *blind* detector faces the clean host as interference,
   :math:`y = u + \eta`, :math:`\eta\sim\mathcal N(0, I)`.  With input covariance
   :math:`Q\succeq0` and masking cost :math:`\mathrm{tr}(M_{\mathrm{row}}Q)\le D`,

   .. math::
       R^\* = \max_{\mathrm{tr}(M_{\mathrm{row}}Q)\le D}\ \tfrac12\log\det(I + Q),

   a water-filling over the eigenmodes of :math:`M_{\mathrm{row}} =
   V_{\mathrm{row}}^\top M\,V_{\mathrm{row}}`.  A nullspace watermark has
   :math:`R^\* = 0` after the channel; the invariant sub-channel gives :math:`R^\*>0`.
   Theorems 1 and 2 meet: the survivable set is exactly the non-nullspace of :math:`A`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg

from .channel import ResynthesisChannel
from .masking import MaskingBudget

__all__ = [
    "SurvivingExponent",
    "CapacityResult",
    "surviving_detection_exponent",
    "subspace_detection_exponent",
    "invariant_subchannel_capacity",
    "subspace_capacity",
    "water_filling",
]


@dataclass(frozen=True)
class SurvivingExponent:
    exponent: float          # surviving Chernoff information (nats)
    delta: np.ndarray        # optimal budgeted perturbation
    invariant_fraction: float  # ||P delta||^2 / ||delta||^2 (0 = fully surface)


@dataclass(frozen=True)
class CapacityResult:
    R_star: float            # nats per use
    power: np.ndarray        # water-filling power per invariant eigenmode
    masking_eigs: np.ndarray  # eigenvalues of the reduced masking metric


def surviving_detection_exponent(
    channel: ResynthesisChannel, masking: MaskingBudget
) -> SurvivingExponent:
    r"""Best post-laundering detection exponent achievable within the masking budget."""
    P, M = channel.P, masking.M
    # Generalized eigenproblem P v = lambda M v; take the largest lambda.
    evals, evecs = linalg.eigh(0.5 * (P + P.T), 0.5 * (M + M.T))
    lam = float(evals[-1])
    v = evecs[:, -1]
    delta = masking.scale_to_budget(v)
    exponent = 0.125 * float(delta @ (P @ delta))
    inv_frac = float((delta @ (P @ delta)) / (delta @ delta))
    return SurvivingExponent(exponent, delta, inv_frac)


def subspace_detection_exponent(
    channel: ResynthesisChannel, masking: MaskingBudget, basis: np.ndarray
) -> SurvivingExponent:
    r"""Surviving exponent when the watermark is restricted to ``span(basis)``.

    Feed ``channel.null_basis()`` to obtain the post-hoc / surface watermark's surviving
    exponent (identically zero) or ``channel.row_basis()`` for the invariant-aligned case.
    """
    basis = np.asarray(basis, dtype=float)
    P, M = channel.P, masking.M
    Pr = basis.T @ P @ basis
    Mr = basis.T @ M @ basis
    evals, evecs = linalg.eigh(0.5 * (Pr + Pr.T), 0.5 * (Mr + Mr.T))
    lam = float(evals[-1])
    u = evecs[:, -1]
    v = basis @ u
    # Guard the degenerate all-zero-quotient case (e.g. nullspace basis).
    if lam <= 1e-12:
        delta = masking.scale_to_budget(v)
        return SurvivingExponent(0.0, delta, 0.0)
    delta = masking.scale_to_budget(v)
    exponent = 0.125 * float(delta @ (P @ delta))
    denom = float(delta @ delta)
    inv_frac = float((delta @ (P @ delta)) / denom) if denom > 0 else 0.0
    return SurvivingExponent(exponent, delta, inv_frac)


def water_filling(masking_eigs: np.ndarray, D: float) -> np.ndarray:
    r"""Water-fill :math:`\max \sum \tfrac12\log(1+q_i)` s.t. :math:`\sum \lambda_i q_i\le D`.

    Optimal ``q_i = max(0, 1/(2μλ_i) - 1)`` with the multiplier ``μ`` set so the budget
    binds.  Solved by bisection on the water level ``1/(2μ)``.
    """
    lam = np.asarray(masking_eigs, dtype=float)
    lam = np.clip(lam, 1e-15, None)

    def spent(level: float) -> float:
        q = np.clip(level - 1.0, 0.0, None)  # q_i>0 where water level > 1 (host var = 1)
        # level here is 1/(2 μ λ_i)? No — see derivation below; handled per-mode.
        return float(np.sum(lam * q))

    # Per-mode threshold: q_i = max(0, ν/λ_i - 1) where ν = 1/(2μ) is the common level.
    def spent_nu(nu: float) -> float:
        q = np.clip(nu / lam - 1.0, 0.0, None)
        return float(np.sum(lam * q))

    lo, hi = 0.0, 1.0
    while spent_nu(hi) < D:
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if spent_nu(mid) < D:
            lo = mid
        else:
            hi = mid
    nu = 0.5 * (lo + hi)
    return np.clip(nu / lam - 1.0, 0.0, None)


def subspace_capacity(
    channel: ResynthesisChannel,
    masking: MaskingBudget,
    basis: np.ndarray,
) -> CapacityResult:
    r"""Blind-detector achievable rate of the sub-channel spanned by ``basis`` after ``W``.

    In the whitened model the clean host has unit variance per orthonormal coordinate, so
    the surviving invariant sub-channel is the AWGN channel :math:`y=u+\eta`,
    :math:`\eta\sim\N(0,I)`, with input cost :math:`\tr(M_{\mathrm{row}}Q)\le D`.  Only the
    invariant component of ``basis`` carries payload through :math:`W`.  This is the rate of
    a *blind* (non-informed) embedder; informed/dirty-paper coding could exceed it.
    """
    basis = np.asarray(basis, dtype=float)
    P, M = channel.P, masking.M
    # Restrict to the surviving (invariant) part of the requested subspace.
    Pb = P @ basis
    # Orthonormal basis of the surviving image span(P basis):
    q_img, r_img = np.linalg.qr(Pb)
    keep = np.abs(np.diag(r_img)) > 1e-9 if r_img.size else np.array([], dtype=bool)
    surv = q_img[:, : keep.sum()] if keep.size else q_img[:, :0]
    if surv.shape[1] == 0:
        return CapacityResult(0.0, np.zeros(0), np.zeros(0))
    Mr = surv.T @ M @ surv
    evals, _ = linalg.eigh(0.5 * (Mr + Mr.T))
    evals = np.clip(evals, 1e-12, None)
    q = water_filling(evals, masking.D)
    R = 0.5 * float(np.sum(np.log1p(q)))    # unit-variance host: rate = 1/2 log(1+q_i)
    return CapacityResult(R, q, evals)


def invariant_subchannel_capacity(
    channel: ResynthesisChannel, masking: MaskingBudget
) -> CapacityResult:
    r"""Surviving blind-detector rate :math:`R^\*` of the full invariant subspace."""
    return subspace_capacity(channel, masking, channel.row_basis())
