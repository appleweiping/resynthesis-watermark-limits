r"""The analysis--resynthesis channel :math:`W = S \circ A`.

An attacker "launders" audio by running it through a neural vocoder / codec / voice-
conversion model.  We model any such blind generator as

.. math::
    W(x) \;=\; S(A(x)),

an **analysis** map :math:`A` producing a representation :math:`z = A(x)`, followed by
a **synthesis** map :math:`S` (possibly stochastic, with randomness independent of the
input) that re-generates a waveform consistent with :math:`z`.

General model (Proposition 1 — what holds without further assumptions)
----------------------------------------------------------------------
For *any* channel :math:`W` the Chernoff information satisfies the data-processing
inequality :math:`C(W_\#P_0, W_\#P_1)\le C(P_0,P_1)`: laundering can only lose
detectability.  **Exact erasure** holds under a sufficient condition: if the embedder
:math:`E_\kappa` satisfies :math:`A(E_\kappa(x)) = A(x)` :math:`P_0`-a.s. and :math:`S`
depends on the input only through :math:`z` (plus independent randomness), then
:math:`W_\#P_1 = W_\#P_0` and :math:`C(W_\#P_0,W_\#P_1)=0` for every detector and
sample size.  Nothing stronger is claimed for general nonlinear vocoders/codecs;
in particular the closed-form constants below do NOT transfer to them.

Linear-Gaussian surrogate (this module — where the exact constants live)
------------------------------------------------------------------------
Work in the *whitened* coordinates where the clean prior is isotropic,
:math:`x \sim \mathcal N(0, I_n)` (all perceptual weighting is carried by the masking
metric in :mod:`rwl.masking`).  Let :math:`A \in \mathbb R^{k\times n}` have full row
rank and let :math:`P = A^{+}A` be the orthogonal projector onto its row space
:math:`\mathrm{row}(A)`; :math:`I-P` projects onto :math:`\ker(A)` (the subspace
:math:`S` resamples).  Then

.. math::
    W(x) \;=\; P x \;+\; (I-P)\, w, \qquad w \sim \mathcal N(0, I_n)\ \text{fresh.}

Consequences (surrogate only):

* :math:`W_\#\mathcal N(0, I) = \mathcal N(0, I)` and, for a fixed additive shift
  :math:`x\mapsto x+\delta`, :math:`W_\#\mathcal N(\delta, I) = \mathcal N(P\delta, I)`,
  so :math:`C(W_\#P_0,W_\#P_1)=\tfrac18\|P\delta\|^2` exactly.
* The channel transmits only the equivalence class
  :math:`[\delta]\in\mathbb R^n/\ker A`; the effective surviving perturbation is
  :math:`P\delta`.  Perturbations in :math:`\ker A` are erased exactly; **mixed**
  perturbations (row + null components) survive partially; row-space-only
  perturbations spend no budget on erased components but are not the unique
  surviving set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ResynthesisChannel"]


@dataclass
class ResynthesisChannel:
    r"""Linear-Gaussian resynthesis channel defined by an analysis map ``A`` (``k×n``).

    Parameters
    ----------
    A:
        Analysis map, shape ``(k, n)`` with ``k <= n`` and full row rank.  Its row space
        is the invariant subspace preserved by resynthesis; its nullspace is resampled.
    rank_tol:
        Relative singular-value tolerance for identifying the numerical row space.
    """

    A: np.ndarray
    rank_tol: float = 1e-9

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=float)
        if self.A.ndim != 2:
            raise ValueError("A must be a 2-D array of shape (k, n)")
        self.k, self.n = self.A.shape
        # Orthonormal bases for row space (invariant) and null space (surface) via SVD.
        U, sv, Vt = np.linalg.svd(self.A, full_matrices=True)
        tol = self.rank_tol * (sv[0] if sv.size else 1.0)
        self.rank = int(np.sum(sv > tol))
        self._V_row = Vt[: self.rank].T          # (n, rank) — invariant subspace basis
        self._V_null = Vt[self.rank :].T         # (n, n-rank) — surface subspace basis
        self.P = self._V_row @ self._V_row.T     # orthogonal projector onto row(A)

    # -- geometry ---------------------------------------------------------------
    @property
    def rowspace_dim(self) -> int:
        """Dimension of the invariant subspace (what resynthesis preserves)."""
        return self.rank

    @property
    def nullspace_dim(self) -> int:
        """Dimension of the surface subspace (what resynthesis resamples)."""
        return self.n - self.rank

    def row_basis(self) -> np.ndarray:
        """Orthonormal basis of the invariant subspace, shape ``(n, rank)``."""
        return self._V_row

    def null_basis(self) -> np.ndarray:
        """Orthonormal basis of the surface subspace, shape ``(n, n-rank)``."""
        return self._V_null

    def project_invariant(self, delta: np.ndarray) -> np.ndarray:
        r"""Return :math:`P\delta`, the surviving (invariant) component of a perturbation."""
        return np.asarray(delta, dtype=float) @ self.P.T

    def project_surface(self, delta: np.ndarray) -> np.ndarray:
        r"""Return :math:`(I-P)\delta`, the erased (surface) component of a perturbation."""
        delta = np.asarray(delta, dtype=float)
        return delta - self.project_invariant(delta)

    # -- the channel ------------------------------------------------------------
    def apply(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        r"""Sample :math:`W(x) = P x + (I-P) w`, ``w`` fresh standard normal.

        Accepts a single vector ``(n,)`` or a batch ``(m, n)``; the surface coordinates
        are resampled independently per row.
        """
        rng = np.random.default_rng() if rng is None else rng
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        xb = x[None, :] if single else x
        w = rng.standard_normal(xb.shape)
        out = xb @ self.P.T + (w - w @ self.P.T)  # P x + (I-P) w
        return out[0] if single else out

    def mean_shift(self, delta: np.ndarray) -> np.ndarray:
        r"""Deterministic surviving shift :math:`P\delta` induced at the channel output."""
        return self.project_invariant(delta)

    def pushforward_gaussian(
        self, mu: np.ndarray, cov: np.ndarray | float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Mean/covariance of :math:`W_\#\mathcal N(\mu, \Sigma)`.

        :math:`W x = P x + (I-P) w` with independent ``w~N(0,I)`` gives mean
        :math:`P\mu` and covariance :math:`P\Sigma P + (I-P)`.
        """
        mu = np.asarray(mu, dtype=float).ravel()
        P, I = self.P, np.eye(self.n)
        if np.isscalar(cov) or np.ndim(cov) == 0:
            sigma = float(cov) * I
        elif np.ndim(cov) == 1:
            sigma = np.diag(np.asarray(cov, dtype=float))
        else:
            sigma = np.asarray(cov, dtype=float)
        mu_out = P @ mu
        cov_out = P @ sigma @ P + (I - P)
        return mu_out, 0.5 * (cov_out + cov_out.T)

    # -- constructors -----------------------------------------------------------
    @classmethod
    def from_random(
        cls, n: int, k: int, rng: np.random.Generator | None = None
    ) -> "ResynthesisChannel":
        """A channel whose invariant subspace is a random ``k``-plane in ``R^n``."""
        rng = np.random.default_rng() if rng is None else rng
        A = rng.standard_normal((k, n))
        return cls(A)

    @classmethod
    def from_invariant_basis(cls, V_row: np.ndarray) -> "ResynthesisChannel":
        """Build a channel whose invariant subspace is ``span(V_row)`` (columns, ``n×k``)."""
        V_row = np.asarray(V_row, dtype=float)
        # Any A with row space = span(V_row); use A = V_row^T (rows span the subspace).
        return cls(V_row.T)
