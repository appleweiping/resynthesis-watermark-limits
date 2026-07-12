"""Geometry and pushforward correctness of the resynthesis channel."""

import numpy as np
import pytest

from rwl.channel import ResynthesisChannel


@pytest.fixture
def channel():
    rng = np.random.default_rng(0)
    return ResynthesisChannel.from_random(n=12, k=5, rng=rng)


def test_projector_is_symmetric_idempotent(channel):
    P = channel.P
    assert np.allclose(P, P.T, atol=1e-10)
    assert np.allclose(P @ P, P, atol=1e-10)
    assert channel.rowspace_dim == 5
    assert channel.nullspace_dim == 7
    assert np.isclose(np.trace(P), 5.0, atol=1e-9)


def test_bases_orthonormal_and_null(channel):
    Vr, Vn = channel.row_basis(), channel.null_basis()
    assert np.allclose(Vr.T @ Vr, np.eye(Vr.shape[1]), atol=1e-9)
    assert np.allclose(Vn.T @ Vn, np.eye(Vn.shape[1]), atol=1e-9)
    # Null basis is annihilated by A; row and null subspaces are orthogonal.
    assert np.allclose(channel.A @ Vn, 0.0, atol=1e-9)
    assert np.allclose(Vr.T @ Vn, 0.0, atol=1e-9)


def test_invariant_surface_decomposition(channel):
    rng = np.random.default_rng(1)
    d = rng.standard_normal(channel.n)
    assert np.allclose(channel.project_invariant(d) + channel.project_surface(d), d)
    # Surface component is truly in the nullspace: A annihilates it.
    assert np.allclose(channel.A @ channel.project_surface(d), 0.0, atol=1e-9)


def test_pushforward_gaussian_identity_and_shift(channel):
    mu0, cov0 = channel.pushforward_gaussian(np.zeros(channel.n), 1.0)
    assert np.allclose(mu0, 0.0, atol=1e-10)
    assert np.allclose(cov0, np.eye(channel.n), atol=1e-10)

    rng = np.random.default_rng(2)
    delta = rng.standard_normal(channel.n)
    mu1, cov1 = channel.pushforward_gaussian(delta, 1.0)
    assert np.allclose(mu1, channel.project_invariant(delta), atol=1e-10)
    assert np.allclose(cov1, np.eye(channel.n), atol=1e-10)


def test_apply_matches_pushforward_empirically(channel):
    rng = np.random.default_rng(3)
    delta = channel.project_surface(rng.standard_normal(channel.n))  # a nullspace shift
    m = 60_000
    x1 = rng.standard_normal((m, channel.n)) + delta
    y1 = channel.apply(x1, rng=rng)
    # A nullspace shift is invisible at the output: empirical mean ~ 0, cov ~ I.
    assert np.linalg.norm(y1.mean(axis=0)) < 0.05
    emp_cov = np.cov(y1, rowvar=False)
    assert np.linalg.norm(emp_cov - np.eye(channel.n)) / channel.n < 0.05
