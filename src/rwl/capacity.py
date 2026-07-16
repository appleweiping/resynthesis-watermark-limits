r"""What survives the linear-Gaussian resynthesis channel, and a rate LOWER BOUND.

Everything in this module is specific to the linear-Gaussian surrogate of
:mod:`rwl.channel`; nothing here is claimed for general nonlinear vocoders/codecs
(for those, only the data-processing inequality of Proposition 1 applies).

Two survivable quantities, both under the masking budget :math:`\delta^\top M\delta\le D`:

1. **Surviving detection exponent** (zero-rate / one-bit provenance).  The post-laundering
   Chernoff information for a fixed additive shift is
   :math:`\tfrac18\|P\delta\|^2 = \tfrac18\,\delta^\top P\delta`
   (:math:`P` is an orthogonal projector, so :math:`P^\top P = P`).  Maximizing it under
   the budget is a generalized Rayleigh quotient

   .. math::
       \max_{\delta^\top M\delta \le D} \tfrac18\,\delta^\top P\delta
       = \tfrac{D}{8}\,\lambda_{\max}(P, M),

   attained at the top generalized eigenvector of :math:`(P, M)`.  The channel
   transmits only the equivalence class :math:`[\delta]\in\mathbb R^n/\ker A`: the
   effective surviving perturbation is :math:`P\delta`.  Perturbations in
   :math:`\ker A` are erased exactly; mixed perturbations survive *partially*
   (exponent :math:`\tfrac18\|P\delta\|^2 > 0` whenever :math:`P\delta\ne0`).
   Row-space-only perturbations do not waste budget on erased components, but they
   are **not** the only surviving perturbations.

2. **Gaussian invariant-channel achievable lower bound** :math:`R_{\mathrm{LB}}`
   (multi-bit payload).  Writing the surviving shift as :math:`u = P\delta` in an
   orthonormal basis of :math:`\mathrm{row}(A)`, a *blind* (uninformed) detector faces
   the clean host as interference, :math:`y = u + \eta`, :math:`\eta\sim\mathcal N(0, I)`.
   With input covariance :math:`Q\succeq0` and masking cost
   :math:`\mathrm{tr}(M_{\mathrm{row}}Q)\le D` (a per-block average budget over i.i.d.
   blocks, each block the :math:`k`-dim invariant vector; units: nats per use of the
   :math:`k`-dimensional invariant vector channel --- a *total* over the :math:`k`
   surviving coordinates, not a per-coordinate rate),

   .. math::
       R_{\mathrm{LB}} = \max_{\mathrm{tr}(M_{\mathrm{row}}Q)\le D}\
       \tfrac12\log\det(I + Q),

   a water-filling over the eigenmodes of :math:`M_{\mathrm{row}} =
   V_{\mathrm{row}}^\top M\,V_{\mathrm{row}}`.

   **What this is and is not.**  :math:`R_{\mathrm{LB}}` is an achievable rate for the
   restricted class of embedders that (i) do not observe the host (uninformed encoder),
   (ii) use Gaussian codebooks keyed by :math:`\kappa`, and (iii) face a blind decoder
   knowing :math:`\kappa` but not the host.  It is a **lower bound** on what is possible:
   an informed (host-aware) embedder in the Costa / Moulin–O'Sullivan sense can exceed
   it by dirty-paper coding against the host interference.  We prove **no rate
   converse**; this module makes no "watermarking capacity" claim.  A nullspace
   watermark has :math:`R_{\mathrm{LB}} = 0` after the channel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg

from .channel import ResynthesisChannel
from .masking import MaskingBudget

__all__ = [
    "SurvivingExponent",
    "RateLowerBound",
    "surviving_detection_exponent",
    "subspace_detection_exponent",
    "invariant_subchannel_rate_lb",
    "subspace_rate_lb",
    "water_filling",
]


@dataclass(frozen=True)
class SurvivingExponent:
    exponent: float          # surviving Chernoff information (nats), LG surrogate only
    delta: np.ndarray        # optimal budgeted perturbation
    invariant_fraction: float  # ||P delta||^2 / ||delta||^2 (0 = fully in ker A)


@dataclass(frozen=True)
class RateLowerBound:
    R_lb: float              # nats per invariant VECTOR-channel use (total over k coords)
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

    Feed ``channel.null_basis()`` to obtain the nullspace watermark's surviving
    exponent (identically zero) or ``channel.row_basis()`` for the invariant-aligned case.
    Mixed subspaces yield partial survival through their :math:`P\delta` component.
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

    Optimal ``q_i = max(0, nu/λ_i - 1)`` with the water level ``nu`` set so the budget
    binds.  Solved by bisection on ``nu``.
    """
    lam = np.asarray(masking_eigs, dtype=float)
    lam = np.clip(lam, 1e-15, None)

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


def subspace_rate_lb(
    channel: ResynthesisChannel,
    masking: MaskingBudget,
    basis: np.ndarray,
) -> RateLowerBound:
    r"""Blind-embedder achievable rate of the sub-channel spanned by ``basis`` after ``W``.

    In the whitened model the clean host has unit variance per orthonormal coordinate, so
    the surviving invariant sub-channel is the AWGN channel :math:`y=u+\eta`,
    :math:`\eta\sim\mathcal N(0,I)`, with input cost :math:`\mathrm{tr}(M_{\mathrm{row}}Q)\le D`.
    Only the invariant component of ``basis`` carries payload through :math:`W`.  This is
    an achievable rate for an *uninformed* (host-blind) embedder — a lower bound;
    informed/dirty-paper coding could exceed it, and no converse is claimed.
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
        return RateLowerBound(0.0, np.zeros(0), np.zeros(0))
    Mr = surv.T @ M @ surv
    evals, _ = linalg.eigh(0.5 * (Mr + Mr.T))
    evals = np.clip(evals, 1e-12, None)
    q = water_filling(evals, masking.D)
    R = 0.5 * float(np.sum(np.log1p(q)))    # unit-variance host: rate = 1/2 log(1+q_i)
    return RateLowerBound(R, q, evals)


def invariant_subchannel_rate_lb(
    channel: ResynthesisChannel, masking: MaskingBudget
) -> RateLowerBound:
    r"""Achievable lower bound :math:`R_{\mathrm{LB}}` on the full invariant subspace."""
    return subspace_rate_lb(channel, masking, channel.row_basis())
