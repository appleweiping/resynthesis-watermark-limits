"""Surrogate sanity checks: surviving exponent and the achievable lower bound R_LB.

These tests verify the numerical implementation against the closed-form surrogate
quantities (they validate the code, not the model). An invariant-aligned watermark
keeps a positive post-laundering detection exponent and a positive achievable rate
R_LB, while a nullspace watermark's surviving R_LB is exactly zero. No capacity or
rate-converse claim is made.
"""

import numpy as np
import pytest

from rwl.capacity import (
    invariant_subchannel_rate_lb,
    subspace_rate_lb,
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
    assert null == pytest.approx(0.0, abs=1e-9)   # nullspace mark dies
    assert row > 0.05                              # invariant survives
    assert full >= row - 1e-9                      # full space can only help


def test_mixed_perturbation_survives_partially(setup):
    """Quotient-space framing: a mixed row+null perturbation survives partially."""
    ch, mask, _ = setup
    v_row = ch.row_basis()[:, 0]
    v_null = ch.null_basis()[:, 0]
    mixed = (v_row + v_null) / np.sqrt(2.0)
    surviving = ch.project_invariant(mixed)
    # Effective perturbation is P delta: nonzero but strictly smaller than ||delta||.
    assert 0.0 < np.linalg.norm(surviving) < np.linalg.norm(mixed)
    exponent = 0.125 * float(mixed @ (ch.P @ mixed))
    assert exponent == pytest.approx(0.125 * 0.5, rel=1e-9)  # half the energy survives


def test_invariant_rate_lb_positive_nullspace_zero(setup):
    ch, mask, _ = setup
    r_inv = invariant_subchannel_rate_lb(ch, mask)
    r_null = subspace_rate_lb(ch, mask, ch.null_basis())
    assert r_inv.R_lb > 0.0
    assert r_null.R_lb == pytest.approx(0.0, abs=1e-12)
    assert np.all(r_inv.power >= -1e-12)


def test_rate_lb_monotone_in_budget(setup):
    ch, mask, _ = setup
    rs = []
    for D in [0.5, 1.0, 2.0, 4.0]:
        m = MaskingBudget(mask.M, D=D)
        rs.append(invariant_subchannel_rate_lb(ch, m).R_lb)
    assert all(b >= a - 1e-9 for a, b in zip(rs, rs[1:]))  # non-decreasing
    assert rs[-1] > rs[0]


def test_water_filling_binds_budget():
    lam = np.array([0.5, 1.0, 2.0, 4.0])
    D = 3.0
    q = water_filling(lam, D)
    assert np.all(q >= -1e-12)
    assert np.isclose(np.sum(lam * q), D, rtol=1e-6)  # budget is fully spent
