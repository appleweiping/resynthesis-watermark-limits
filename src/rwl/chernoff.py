r"""Chernoff information and the data-processing inequality.

The optimal (Bayesian) error exponent for deciding whether an observation was drawn
from the clean law :math:`P_0` or the watermarked law :math:`P_1` is the **Chernoff
information**

.. math::
    C(P_0, P_1) \;=\; \max_{0\le s\le 1}\; k(s),
    \qquad
    e^{-k(s)} \;=\; \int p_0(x)^{1-s}\, p_1(x)^{s}\, dx .

If :math:`C = 0` the two laws are indistinguishable: **no** detector, at **any** sample
size, beats a coin flip.  Theorem 1 (the converse) is exactly the statement that the
resynthesis channel drives :math:`C` to zero for nullspace watermarks; this module gives
the closed forms (Gaussian) and Monte-Carlo estimators that let the test suite verify it.

For two Gaussians :math:`\mathcal N(\mu_i, \Sigma_i)` with
:math:`\Sigma(s) = (1-s)\Sigma_0 + s\,\Sigma_1`,

.. math::
    k(s) = \tfrac{s(1-s)}{2}\,\Delta\mu^\top \Sigma(s)^{-1} \Delta\mu
           + \tfrac12 \ln\frac{\det\Sigma(s)}{\det\Sigma_0^{\,1-s}\det\Sigma_1^{\,s}},
    \qquad \Delta\mu = \mu_1-\mu_0 .

At equal covariance this collapses to the Bhattacharyya form
:math:`C = \tfrac18\,\Delta\mu^\top\Sigma^{-1}\Delta\mu` (optimum at :math:`s=\tfrac12`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg
from scipy.optimize import minimize_scalar

__all__ = [
    "ChernoffResult",
    "chernoff_gaussian_equal_cov",
    "bhattacharyya_gaussian",
    "chernoff_gaussian",
    "chernoff_exponent_at_s",
    "chernoff_mc_equal_cov",
    "dpi_gap",
]


def _as_2d_cov(sigma: np.ndarray, dim: int) -> np.ndarray:
    """Broadcast a scalar / 1-D (diagonal) / 2-D covariance to a dense SPD matrix."""
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 0:
        return np.eye(dim) * float(sigma)
    if sigma.ndim == 1:
        return np.diag(sigma)
    return sigma


def _symmetrize(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


def chernoff_gaussian_equal_cov(dmu: np.ndarray, sigma: np.ndarray) -> float:
    r"""Closed-form Chernoff information for equal-covariance Gaussians.

    :math:`C = \tfrac18\,\Delta\mu^\top \Sigma^{-1} \Delta\mu`.  This is the exponent
    that governs how fast a provenance detector's error can shrink; it is *exact* for the
    mean-shift watermark model (embedding adds a key-dependent shift to a Gaussian source).
    """
    dmu = np.asarray(dmu, dtype=float).ravel()
    cov = _as_2d_cov(sigma, dmu.size)
    sol = linalg.solve(cov, dmu, assume_a="pos")
    return float(0.125 * dmu @ sol)


def chernoff_exponent_at_s(
    mu0: np.ndarray,
    sigma0: np.ndarray,
    mu1: np.ndarray,
    sigma1: np.ndarray,
    s: float,
) -> float:
    r"""The Chernoff-bound exponent :math:`k(s)` for two general Gaussians."""
    mu0 = np.asarray(mu0, dtype=float).ravel()
    mu1 = np.asarray(mu1, dtype=float).ravel()
    dim = mu0.size
    s0 = _as_2d_cov(sigma0, dim)
    s1 = _as_2d_cov(sigma1, dim)
    dmu = mu1 - mu0
    ss = _symmetrize((1.0 - s) * s0 + s * s1)

    sol = linalg.solve(ss, dmu, assume_a="pos")
    quad = 0.5 * s * (1.0 - s) * float(dmu @ sol)

    sign_s, logdet_ss = np.linalg.slogdet(ss)
    sign_0, logdet_s0 = np.linalg.slogdet(s0)
    sign_1, logdet_s1 = np.linalg.slogdet(s1)
    if min(sign_s, sign_0, sign_1) <= 0:
        raise ValueError("covariances must be positive definite")
    logterm = 0.5 * (logdet_ss - (1.0 - s) * logdet_s0 - s * logdet_s1)
    return quad + logterm


def bhattacharyya_gaussian(
    mu0: np.ndarray, sigma0: np.ndarray, mu1: np.ndarray, sigma1: np.ndarray
) -> float:
    r"""Bhattacharyya distance :math:`k(\tfrac12)` — a lower bound on Chernoff info."""
    return chernoff_exponent_at_s(mu0, sigma0, mu1, sigma1, 0.5)


@dataclass(frozen=True)
class ChernoffResult:
    """Chernoff information and the optimizing tilt ``s``."""

    C: float
    s_star: float


def chernoff_gaussian(
    mu0: np.ndarray,
    sigma0: np.ndarray,
    mu1: np.ndarray,
    sigma1: np.ndarray,
) -> ChernoffResult:
    r"""Chernoff information :math:`\max_{s\in[0,1]} k(s)` for two Gaussians.

    Uses the equal-covariance closed form (optimum at ``s=1/2``) when the covariances
    match, otherwise maximizes :math:`k(s)` numerically on ``[0, 1]``.
    """
    mu0 = np.asarray(mu0, dtype=float).ravel()
    mu1 = np.asarray(mu1, dtype=float).ravel()
    dim = mu0.size
    s0 = _as_2d_cov(sigma0, dim)
    s1 = _as_2d_cov(sigma1, dim)

    if np.allclose(s0, s1, atol=1e-12):
        return ChernoffResult(chernoff_gaussian_equal_cov(mu1 - mu0, s0), 0.5)

    res = minimize_scalar(
        lambda s: -chernoff_exponent_at_s(mu0, s0, mu1, s1, s),
        bounds=(1e-6, 1.0 - 1e-6),
        method="bounded",
        options={"xatol": 1e-8},
    )
    return ChernoffResult(float(-res.fun), float(res.x))


def chernoff_mc_equal_cov(
    dmu: np.ndarray,
    sigma: np.ndarray,
    n: int = 200_000,
    rng: np.random.Generator | None = None,
) -> float:
    r"""Monte-Carlo estimate of the Chernoff information for equal-covariance Gaussians.

    Uses :math:`k(\tfrac12) = -\log \mathbb E_{P_0}\!\big[e^{\frac12 \mathrm{LLR}}\big]`
    with the exact Gaussian log-likelihood ratio, providing an independent check of
    :func:`chernoff_gaussian_equal_cov`.
    """
    rng = np.random.default_rng() if rng is None else rng
    dmu = np.asarray(dmu, dtype=float).ravel()
    dim = dmu.size
    cov = _as_2d_cov(sigma, dim)
    chol = linalg.cholesky(cov, lower=True)

    x = rng.standard_normal((n, dim)) @ chol.T  # samples from P0 = N(0, cov)
    # LLR = log p1/p0 for P1 = N(dmu, cov): (x-0)ᵀΣ⁻¹dmu - 1/2 dmuᵀΣ⁻¹dmu
    sol = linalg.solve(cov, dmu, assume_a="pos")
    llr = x @ sol - 0.5 * float(dmu @ sol)
    # k(1/2) = -log E_{P0}[exp(LLR/2)]; use log-sum-exp for stability.
    a = 0.5 * llr
    m = a.max()
    log_mean = m + np.log(np.mean(np.exp(a - m)))
    return float(-log_mean)


def dpi_gap(
    chernoff_before: float, chernoff_after: float, tol: float = 1e-9
) -> float:
    """Return ``before - after``; the data-processing inequality requires it be ``>= -tol``.

    Chernoff information is monotone under any (stochastic) channel, so passing both laws
    through the resynthesis channel can only *lose* detectability.  A negative gap beyond
    ``tol`` signals a bug in a channel or estimator, never a real violation.
    """
    return float(chernoff_before - chernoff_after)
