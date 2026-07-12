"""Theorem 1 (converse) + the data-processing inequality.

A watermark in the analysis nullspace is erased by the resynthesis channel: its
post-laundering detection exponent is exactly zero and the optimal detector's AUC
collapses to chance.  For any watermark, detectability is monotone non-increasing.
"""

import numpy as np
import pytest

from rwl import chernoff
from rwl.channel import ResynthesisChannel
from rwl.masking import MaskingBudget
from rwl.watermark import nullspace_watermark, simulate_auc


@pytest.fixture
def setup():
    rng = np.random.default_rng(7)
    ch = ResynthesisChannel.from_random(n=16, k=6, rng=rng)
    mask = MaskingBudget(np.eye(ch.n), D=2.0)
    return ch, mask, rng


def test_nullspace_watermark_has_zero_surviving_exponent(setup):
    ch, mask, rng = setup
    delta = nullspace_watermark(ch, mask)
    # Exactly in the nullspace: P delta = 0.
    assert np.linalg.norm(ch.project_invariant(delta)) < 1e-9
    pre = chernoff.chernoff_gaussian_equal_cov(delta, np.eye(ch.n))
    post = chernoff.chernoff_gaussian_equal_cov(ch.project_invariant(delta), np.eye(ch.n))
    assert pre > 0.1               # detectable before laundering
    assert post < 1e-12            # provably erased after laundering


def test_nullspace_watermark_auc_collapses_to_chance(setup):
    ch, mask, rng = setup
    delta = nullspace_watermark(ch, mask)
    before = simulate_auc(ch, delta, n_trials=40_000, after=False, rng=rng)
    after = simulate_auc(ch, delta, n_trials=40_000, after=True, rng=rng)
    assert before.auc_empirical > 0.7          # clearly detectable pre-attack
    assert abs(after.auc_empirical - 0.5) < 0.02  # chance post-attack
    assert after.deflection < 1e-9


def test_data_processing_inequality_holds_for_random_watermarks(setup):
    ch, _, rng = setup
    for _ in range(200):
        delta = rng.standard_normal(ch.n)
        pre = chernoff.chernoff_gaussian_equal_cov(delta, np.eye(ch.n))
        post = chernoff.chernoff_gaussian_equal_cov(
            ch.project_invariant(delta), np.eye(ch.n)
        )
        assert chernoff.dpi_gap(pre, post) >= -1e-9   # post <= pre always


def test_surviving_detectability_shrinks_with_nullspace_fraction(setup):
    ch, mask, rng = setup
    row_dir = ch.row_basis()[:, 0]
    null_dir = ch.null_basis()[:, 0]
    aucs = []
    for alpha in np.linspace(0.0, 1.0, 6):  # alpha = fraction of energy in nullspace
        d = np.sqrt(1 - alpha) * row_dir + np.sqrt(alpha) * null_dir
        d = mask.scale_to_budget(d)
        aucs.append(simulate_auc(ch, d, n_trials=30_000, after=True, rng=rng).auc_empirical)
    # Monotone decreasing toward chance as the watermark drifts into the nullspace.
    assert aucs[0] > aucs[-1]
    assert abs(aucs[-1] - 0.5) < 0.03
    assert aucs[0] > 0.7
