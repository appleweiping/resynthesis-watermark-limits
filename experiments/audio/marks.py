"""Watermark perturbation constructions w.r.t. THE unified analysis map.

All constructions are host-adaptive (the embedder sees the host — standard informed
embedding) and keyed. They are built in the STFT domain at the analysis grid of
`analysis.MelAnalysis` and returned as additive waveform perturbations.

* nullspace  : quadrature (pure-phase) increments D_k = i * e^{i phi_k} * g_k — to first
               order |X + D| = |X|, so A(x+delta) = A(x) for the magnitude-mel A.
               ISTFT consistency makes this only approximately exact; the residual is
               MEASURED and reported (verify()), never assumed away.
* rowspace   : in-phase increments along mel filterbank patterns
               D_k = e^{i phi_k} * (FB @ u)_k — moves A(x) directly.
* inphase_rand: in-phase increments with a random (non-mel-aligned) magnitude pattern —
               moves |STFT| in a random direction; its mel fraction falls where it falls.
* mixture    : unit-norm combination sqrt(1-b)*nullspace + sqrt(b)*rowspace.

Scoring (diagnostic, genie-aided): the separability probe correlates the attacked signal
with the KNOWN per-utterance perturbation, t(y) = <y, delta>/||delta||. This measures
channel geometry; it is NOT a deployable blind detector, and is labeled as such
everywhere. Deployed-detector claims use the real baselines (audioseal/wavmark/...).
"""

from __future__ import annotations

import numpy as np
import torch

from .analysis import MelAnalysis

ENERGY_FLOOR_DB = -50.0   # bins below this (rel. max) carry no mark energy


def _keyed_rng(key: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(key))
    return g


def _active_mask(mag: torch.Tensor) -> torch.Tensor:
    """Bins with energy above ENERGY_FLOOR_DB relative to the clip max."""
    floor = mag.max() * (10.0 ** (ENERGY_FLOOR_DB / 20.0))
    return mag > floor


def _smooth_frames(pattern: torch.Tensor, width: int = 9) -> torch.Tensor:
    """Moving-average smoothing along frames so patterns are low-rate (robust)."""
    if width <= 1:
        return pattern
    kernel = torch.ones(1, 1, width, device=pattern.device) / width
    p = pattern[None, :, :]                          # (1, rows, frames)
    p = torch.nn.functional.conv1d(
        p.transpose(0, 1), kernel, padding=width // 2
    ).transpose(0, 1)
    return p[0, :, : pattern.shape[1]]


def nullspace_direction(
    an: MelAnalysis, x: torch.Tensor, key: int, n_proj: int = 60
) -> torch.Tensor:
    """First-order kernel of the magnitude analysis, via alternating projections.

    The pointwise-quadrature spectrograms (D = i e^{i phi} r, r real) and the
    consistent spectrograms (range of STFT) are both real-linear subspaces; their
    intersection is exactly the waveform perturbations that leave |STFT| (hence mel)
    unchanged to first order. Alternating projections converge onto it. The residual
    leakage is MEASURED by verify_direction(), never assumed zero.
    """
    X = an.stft(x)
    mag, phase = X.abs(), torch.angle(X)
    u = torch.exp(1j * phase)                        # host phase direction
    mask = _active_mask(mag)
    g = torch.randn(X.shape, generator=_keyed_rng(key)).to(x.device)
    g = _smooth_frames(g) * mask
    D = 1j * u * (g * mag)                            # quadrature seed
    delta = an.istft(D, length=x.shape[-1])           # ISTFT is linear
    for _ in range(n_proj):
        E = an.stft(delta)                            # -> consistent subspace
        quad = (E * u.conj()).imag                    # quadrature (real) component
        E_q = 1j * u * (quad * mask)                  # kill in-phase + inactive bins
        delta = an.istft(E_q, length=x.shape[-1])
    n = torch.linalg.norm(delta)
    return delta / n.clamp_min(1e-12)


def rowspace_direction(
    an: MelAnalysis, x: torch.Tensor, key: int
) -> torch.Tensor:
    """In-phase mel-pattern perturbation: moves A(x) along the filterbank rows."""
    X = an.stft(x)
    mag, phase = X.abs(), torch.angle(X)
    n_frames = X.shape[-1]
    u = torch.randn((an.n_mels, n_frames), generator=_keyed_rng(key)).to(x.device)
    u = _smooth_frames(u)
    pat = an.fb @ u                                   # (freq, frames)
    D = torch.exp(1j * phase) * (pat * mag) * _active_mask(mag)
    delta = an.istft(X + D, length=x.shape[-1]) - x
    n = torch.linalg.norm(delta)
    return delta / n.clamp_min(1e-12)


def inphase_random_direction(
    an: MelAnalysis, x: torch.Tensor, key: int
) -> torch.Tensor:
    """In-phase perturbation with a random per-bin magnitude pattern."""
    X = an.stft(x)
    mag, phase = X.abs(), torch.angle(X)
    h = torch.randn(X.shape, generator=_keyed_rng(key)).to(x.device)
    h = _smooth_frames(h) * _active_mask(mag)
    D = torch.exp(1j * phase) * (h * mag)
    delta = an.istft(X + D, length=x.shape[-1]) - x
    n = torch.linalg.norm(delta)
    return delta / n.clamp_min(1e-12)


def mixture_direction(
    an: MelAnalysis, x: torch.Tensor, key: int, beta: float
) -> torch.Tensor:
    """sqrt(1-b) * nullspace + sqrt(b) * rowspace, renormalized to unit norm."""
    d0 = nullspace_direction(an, x, key)
    d1 = rowspace_direction(an, x, key + 7919)
    d = np.sqrt(max(0.0, 1.0 - beta)) * d0 + np.sqrt(max(0.0, beta)) * d1
    n = torch.linalg.norm(d)
    return d / n.clamp_min(1e-12)


DIRECTION_BUILDERS = {
    "nullspace": lambda an, x, key, beta=0.0: nullspace_direction(an, x, key),
    "rowspace": lambda an, x, key, beta=1.0: rowspace_direction(an, x, key),
    "inphase_rand": lambda an, x, key, beta=0.5: inphase_random_direction(an, x, key),
    "mixture": mixture_direction,
}


# ---- verification -------------------------------------------------------------
def verify_direction(an: MelAnalysis, x: torch.Tensor, delta: torch.Tensor) -> dict:
    """Measured analysis change of a direction, absolute and vs a random control.

    Returns ratio = ||A(x+d)-A(x)||/||d|| at the direction's ACTUAL scale (finite-
    amplitude leakage, what matters operationally) and ratio_rel = ratio / ratio
    (random waveform direction of the same norm). A true nullspace direction must
    show ratio_rel << 1; a rowspace direction ratio_rel >~ 1. These numbers are
    REPORTED, not assumed.
    """
    ratio = an.analysis_change(x, delta)
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    rnd = torch.randn(x.shape, generator=g).to(x.device)
    rnd = rnd / torch.linalg.norm(rnd).clamp_min(1e-12) * torch.linalg.norm(delta)
    ratio_rnd = an.analysis_change(x, rnd)
    return {
        "delta_norm_rel": float(torch.linalg.norm(delta) / torch.linalg.norm(x)),
        "analysis_change_per_unit": ratio,
        "analysis_change_random_control": ratio_rnd,
        "ratio_rel": ratio / max(ratio_rnd, 1e-12),
    }


# ---- budgets --------------------------------------------------------------------
def scale_to_snr(x: torch.Tensor, direction: torch.Tensor, snr_db: float) -> torch.Tensor:
    """alpha * direction with 10 log10(||x||^2/||alpha d||^2) = snr_db."""
    alpha = torch.linalg.norm(x) * (10.0 ** (-snr_db / 20.0))
    return direction / torch.linalg.norm(direction).clamp_min(1e-12) * alpha


def scale_to_pesq(
    x: torch.Tensor, direction: torch.Tensor, sr: int,
    target: float = 4.2, tol: float = 0.05, max_iter: int = 12,
) -> tuple[torch.Tensor, float, float]:
    """Bisection on the embedding gain so PESQ-WB(x, x+delta) hits `target` +- `tol`.

    Returns (delta, achieved_pesq, snr_db). All marks calibrated this way share the
    same perceptual budget across CONSTRUCTED marks (PESQ-matched, not SNR-matched); the
    # deployed baselines saturate above this target and run at native strength.
    """
    from pesq import pesq as pesq_fn

    ref = x.detach().cpu().numpy().astype(np.float64)
    d_unit = (direction / torch.linalg.norm(direction).clamp_min(1e-12))

    def pesq_at(alpha: float) -> float:
        deg = (x + alpha * d_unit).detach().cpu().numpy().astype(np.float64)
        return float(pesq_fn(sr, ref, deg, "wb"))

    # bracket: grow alpha until PESQ drops below target
    lo, hi = 0.0, float(0.003 * torch.linalg.norm(x))
    p_hi = pesq_at(hi)
    grow = 0
    while p_hi > target and grow < 14:
        lo, hi = hi, hi * 2.0
        p_hi = pesq_at(hi)
        grow += 1
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p = pesq_at(mid)
        if abs(p - target) <= tol:
            lo = hi = mid
            break
        if p > target:
            lo = mid
        else:
            hi = mid
    alpha = 0.5 * (lo + hi)
    delta = alpha * d_unit
    achieved = pesq_at(alpha)
    snr = float(20.0 * torch.log10(
        torch.linalg.norm(x) / torch.linalg.norm(delta).clamp_min(1e-12)))
    return delta, achieved, snr


# ---- diagnostic separability probes ---------------------------------------------
def probe_score(y: torch.Tensor, delta: torch.Tensor) -> float:
    """WAVEFORM-domain genie probe t(y) = <y, delta>/||delta||.

    Sensitive to attacks that re-derive phase/fine structure even when the
    magnitude pattern survives — i.e., it measures waveform-domain transmission.
    """
    d = delta / torch.linalg.norm(delta).clamp_min(1e-12)
    return float(torch.dot(y.flatten(), d.flatten()))


def mel_probe_direction(an, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Unit-norm mel-domain pattern written by delta: (A(x+d)-A(x)) / ||.||."""
    dmel = an.mel(x + delta) - an.mel(x)
    n = torch.linalg.norm(dmel)
    return dmel / n.clamp_min(1e-12)


def probe_score_mel(an, y_mel: torch.Tensor, dmel_unit: torch.Tensor) -> float:
    """MEL-domain genie probe t(y) = <A(y), dmel_unit> (fixed mel-reading detector).

    Measures transmission of the embedded magnitude pattern through the channel;
    blind to phase-domain content by construction. Survival is a property of the
    (channel, detector-domain) pair — both probes are reported.
    """
    m = min(y_mel.shape[-1], dmel_unit.shape[-1])
    return float(torch.sum(y_mel[..., :m] * dmel_unit[..., :m]))
