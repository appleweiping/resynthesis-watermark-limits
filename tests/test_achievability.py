"""Theorem 2 (achievability): what survives, and at what rate.

An invariant-aligned watermark keeps a positive post-laundering detection exponent and a
positive payload rate R*, while a nullspace watermark's surviving rate is exactly zero.
The two theorems meet: survivable = non-nullspace of A.
"""

import numpy as np
import pytest

from rwl.capacity import (
    invariant_subchannel_capacity,
    subspace_capacity,
    subspace_detection_exponent,
    surviving_detection_exponent,
    water_filling,
)
from rwl.channel import ResynthesisChannel
from rwl.masking import MaskingBudget
from rwl.watermark import invariant_aligned_watermark, simulate_auc


@pytest.fixture
def setup():
    rng = np.random.default_rng(11)
    ch = ResynthesisChannel.from_random(n=16, k=6, rng=rng)
    # Non-isotropic masking so the trade-off is nontrivial.
    w = np.exp(0.5 * rng.standard_normal(ch.n))
    mask = MaskingBudget(np.diag(w), D=2.0)
    return ch, mask, rng


def test_invariant_watermark_survives(setup):
    ch, mask, rng = setup
    delta = invariant_aligned_watermark(ch, mask)
    exp = surviving_detection_exponent(ch, mask)
    assert exp.exponent > 0.05
    assert exp.invariant_fraction > 0.0
    after = simulate_auc(ch, delta, n_trials=40_000, after=True, rng=rng)
    assert after.auc_empirical > 0.6
    assert abs(after.auc_empirical - after.auc_theory) < 0.02


def test_surviving_exponent_orders_correctly(setup):
    ch, mask, _ = setup
    full = surviving_detection_exponent(ch, mask).exponent
    row = subspace_detection_exponent(ch, mask, ch.row_basis()).exponent
    null = subspace_detection_exponent(ch, mask, ch.null_basis()).exponent
    assert null == pytest.approx(0.0, abs=1e-9)   # post-hoc dies
    assert row > 0.05                              # invariant survives
    assert full >= row - 1e-9                      # full space can only help


def test_invariant_capacity_positive_nullspace_zero(setup):
    ch, mask, _ = setup
    r_star = invariant_subchannel_capacity(ch, mask)
    r_null = subspace_capacity(ch, mask, ch.null_basis())
    assert r_star.R_star > 0.0
    assert r_null.R_star == pytest.approx(0.0, abs=1e-12)
    assert np.all(r_star.power >= -1e-12)


def test_capacity_monotone_in_budget(setup):
    ch, mask, _ = setup
    rs = []
    for D in [0.5, 1.0, 2.0, 4.0]:
        m = MaskingBudget(mask.M, D=D)
        rs.append(invariant_subchannel_capacity(ch, m).R_star)
    assert all(b >= a - 1e-9 for a, b in zip(rs, rs[1:]))  # non-decreasing
    assert rs[-1] > rs[0]


def test_water_filling_binds_budget():
    lam = np.array([0.5, 1.0, 2.0, 4.0])
    D = 3.0
    q = water_filling(lam, D)
    assert np.all(q >= -1e-12)
    assert np.isclose(np.sum(lam * q), D, rtol=1e-6)  # budget is fully spent
