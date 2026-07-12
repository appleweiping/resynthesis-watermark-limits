r"""The psychoacoustic imperceptibility budget.

A watermark must be *inaudible*.  In the whitened surrogate we encode audibility as a
quadratic masking metric: a perturbation :math:`\delta` costs

.. math::
    \operatorname{cost}(\delta) = \delta^\top M \delta \le D,

where :math:`M \succ 0` is a masking matrix (small weight = perceptually cheap =
inaudible direction) and :math:`D` is the budget.  Real audio replaces :math:`M` with a
signal-adaptive masking threshold on the STFT (see :mod:`rwl` experiments); the metric
interface here is identical.

The tension that drives the whole paper: **the perceptually cheapest directions (phase,
masked bins) are exactly the ones a resynthesis channel discards.**  A watermark that
minimizes audible distortion naively drifts into :math:`\ker(A)` and is laundered away;
surviving requires paying for an *invariant* direction.  :meth:`MaskingBudget.
cheapest_direction` and :mod:`rwl.capacity` make that trade-off quantitative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg

__all__ = ["MaskingBudget"]


@dataclass
class MaskingBudget:
    r"""Quadratic masking metric ``M`` (SPD) with an energy budget ``D``."""

    M: np.ndarray
    D: float = 1.0

    def __post_init__(self) -> None:
        self.M = np.asarray(self.M, dtype=float)
        if self.M.ndim == 1:
            self.M = np.diag(self.M)
        if self.M.ndim != 2 or self.M.shape[0] != self.M.shape[1]:
            raise ValueError("M must be a square SPD matrix (or 1-D diagonal)")
        self.n = self.M.shape[0]
        # Cholesky doubles as an SPD check.
        self._L = linalg.cholesky(0.5 * (self.M + self.M.T), lower=True)

    def cost(self, delta: np.ndarray) -> np.ndarray:
        r"""Masking cost :math:`\delta^\top M \delta` (per row for a batch)."""
        delta = np.asarray(delta, dtype=float)
        md = delta @ self.M.T
        return np.sum(delta * md, axis=-1)

    def scale_to_budget(self, delta: np.ndarray) -> np.ndarray:
        r"""Rescale ``delta`` so that :math:`\delta^\top M \delta = D` (fills the budget)."""
        delta = np.asarray(delta, dtype=float)
        c = float(self.cost(delta))
        if c <= 0:
            raise ValueError("zero-cost perturbation cannot be scaled to the budget")
        return delta * np.sqrt(self.D / c)

    def cheapest_direction(self, basis: np.ndarray) -> np.ndarray:
        r"""Unit (Euclidean) direction of least masking cost inside ``span(basis)``.

        ``basis`` has shape ``(n, d)`` with orthonormal columns.  Returns the column-space
        vector :math:`v = basis\,u` minimizing :math:`v^\top M v / v^\top v`, i.e. the
        eigenvector of the reduced metric :math:`basis^\top M\,basis` with the smallest
        eigenvalue — the most inaudible direction available in that subspace.
        """
        basis = np.asarray(basis, dtype=float)
        reduced = basis.T @ self.M @ basis
        evals, evecs = linalg.eigh(0.5 * (reduced + reduced.T))
        return basis @ evecs[:, 0]

    def budgeted_vector(self, direction: np.ndarray) -> np.ndarray:
        r"""Return the multiple of ``direction`` that exactly spends the budget ``D``."""
        return self.scale_to_budget(direction)

    @classmethod
    def isotropic(cls, n: int, D: float = 1.0) -> "MaskingBudget":
        """Masking metric ``M = I`` (all directions equally audible)."""
        return cls(np.eye(n), D)

    @classmethod
    def diagonal(cls, weights: np.ndarray, D: float = 1.0) -> "MaskingBudget":
        """Diagonal masking metric; small ``weights[i]`` = inaudible coordinate ``i``."""
        return cls(np.diag(np.asarray(weights, dtype=float)), D)
