"""End-to-end: the closed-form ROC predicts the simulated detector across seeds.

This is the manuscript gate for the headline claim — the theory *predicts the measured
numbers*, before and after laundering, for both watermark families.
"""

import numpy as np
import pytest

from rwl.channel import ResynthesisChannel
from rwl.masking import MaskingBudget
from rwl.watermark import (
    auc_from_deflection,
    detection_deflection,
    invariant_aligned_watermark,
    nullspace_watermark,
    simulate_auc,
)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_simulated_auc_matches_closed_form_after_laundering(seed):
    rng = np.random.default_rng(seed)
    ch = ResynthesisChannel.from_random(n=14, k=5, rng=rng)
    mask = MaskingBudget.isotropic(ch.n, D=2.0)

    inv = invariant_aligned_watermark(ch, mask)
    nul = nullspace_watermark(ch, mask)

    for delta in (inv, nul):
        for after in (False, True):
            sim = simulate_auc(ch, delta, n_trials=50_000, after=after, rng=rng)
            theory = auc_from_deflection(detection_deflection(ch, delta, after))
            assert abs(sim.auc_empirical - theory) < 0.02


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_invariant_beats_nullspace_after_laundering(seed):
    rng = np.random.default_rng(seed)
    ch = ResynthesisChannel.from_random(n=14, k=5, rng=rng)
    mask = MaskingBudget.isotropic(ch.n, D=2.0)

    inv = simulate_auc(ch, invariant_aligned_watermark(ch, mask),
                       n_trials=50_000, after=True, rng=rng)
    nul = simulate_auc(ch, nullspace_watermark(ch, mask),
                       n_trials=50_000, after=True, rng=rng)
    assert inv.auc_empirical > 0.6
    assert abs(nul.auc_empirical - 0.5) < 0.02
    assert inv.auc_empirical - nul.auc_empirical > 0.1
