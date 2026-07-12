"""Chernoff-information estimators: closed form vs Monte-Carlo, and the s-optimization."""

import numpy as np
import pytest

from rwl import chernoff


def test_equal_cov_closed_form_matches_montecarlo():
    rng = np.random.default_rng(0)
    dim = 6
    cov = np.eye(dim) + 0.3 * np.ones((dim, dim))
    cov = 0.5 * (cov + cov.T)
    dmu = 0.4 * rng.standard_normal(dim)

    c_closed = chernoff.chernoff_gaussian_equal_cov(dmu, cov)
    c_mc = chernoff.chernoff_mc_equal_cov(dmu, cov, n=400_000, rng=rng)
    assert c_closed > 0
    assert abs(c_closed - c_mc) / c_closed < 0.03


def test_equal_cov_optimum_is_half():
    dim = 4
    cov = np.diag([1.0, 2.0, 0.5, 1.5])
    dmu = np.array([0.3, -0.2, 0.5, 0.1])
    res = chernoff.chernoff_gaussian(np.zeros(dim), cov, dmu, cov)
    assert np.isclose(res.s_star, 0.5, atol=1e-6)
    assert np.isclose(res.C, chernoff.chernoff_gaussian_equal_cov(dmu, cov), rtol=1e-9)


def test_general_chernoff_ge_bhattacharyya():
    rng = np.random.default_rng(1)
    dim = 5
    a = rng.standard_normal((dim, dim))
    s0 = a @ a.T + np.eye(dim)
    b = rng.standard_normal((dim, dim))
    s1 = b @ b.T + 2 * np.eye(dim)
    mu0 = np.zeros(dim)
    mu1 = 0.5 * rng.standard_normal(dim)

    res = chernoff.chernoff_gaussian(mu0, s0, mu1, s1)
    bhat = chernoff.bhattacharyya_gaussian(mu0, s0, mu1, s1)
    assert res.C >= bhat - 1e-9
    assert 0.0 <= res.s_star <= 1.0
    # k(s) at the optimum dominates k at the endpoints (which vanish for the mean/cov terms).
    assert res.C >= chernoff.chernoff_exponent_at_s(mu0, s0, mu1, s1, 0.25) - 1e-9


def test_zero_shift_gives_zero_information():
    dim = 3
    cov = np.eye(dim)
    assert chernoff.chernoff_gaussian_equal_cov(np.zeros(dim), cov) == 0.0
