"""Detection metrics and theory-linked measurements for real audio.

Two quantities connect the experiment to Theorem 1:

* ``invariant_energy_fraction`` — the fraction of a watermark perturbation's STFT energy
  that lies in the **magnitude** (the subspace Griffin–Lim, and any magnitude-based
  vocoder, preserves).  This is a pure signal-geometry analog of ||Pδ||²/||δ||²,
  bounded in [0,1] and independent of any detector.  A phase/surface watermark has it
  near 0; a magnitude watermark near 1.

* ``detector_survival`` — the bounded fraction of the detector's mean watermark response
  that survives laundering, (Δ_after)/(Δ_before).  Theorem 1 predicts detection dies as
  either quantity → 0.

Note: || W(x+δ) − W(x) ||² is *not* a valid surrogate for ||Pδ||² when W is a nonlinear
generator (Griffin–Lim, neural codecs amplify tiny perturbations), so we never use it.
"""

from __future__ import annotations

import numpy as np

N_FFT = 1024
HOP = 256


def empirical_auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """Mann-Whitney (rank) AUC = P(score_pos > score_neg)."""
    scores_pos = np.asarray(scores_pos, dtype=float)
    scores_neg = np.asarray(scores_neg, dtype=float)
    allv = np.concatenate([scores_pos, scores_neg])
    ranks = allv.argsort().argsort().astype(float) + 1.0
    n1, n0 = scores_pos.size, scores_neg.size
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    return float(u1 / (n1 * n0))


def tpr_at_fpr(scores_pos: np.ndarray, scores_neg: np.ndarray, fpr: float = 0.01) -> float:
    """True-positive rate at a fixed false-positive rate (threshold from negatives)."""
    scores_neg = np.sort(np.asarray(scores_neg, dtype=float))
    k = max(0, int(np.ceil((1.0 - fpr) * scores_neg.size)) - 1)
    thr = scores_neg[min(k, scores_neg.size - 1)]
    return float(np.mean(np.asarray(scores_pos, dtype=float) > thr))


def bit_accuracy(decoded: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of correctly recovered payload bits (chance = 0.5)."""
    return float(np.mean(np.asarray(decoded).astype(int) == np.asarray(truth).astype(int)))


def _stft(x: np.ndarray):
    import librosa
    return librosa.stft(np.asarray(x, dtype=np.float32), n_fft=N_FFT, hop_length=HOP)


def invariant_energy_fraction(x: np.ndarray, x_wm: np.ndarray) -> float:
    """Fraction of the watermark's STFT energy carried by the magnitude (GL-invariant).

    f = || |S(x_wm)| − |S(x)| ||² / || S(x_wm) − S(x) ||²  ∈ [0, 1].
    """
    S0 = _stft(x)
    n = min(S0.shape[1], _stft(x_wm).shape[1])
    S1 = _stft(x_wm)
    m = min(S0.shape[1], S1.shape[1])
    S0, S1 = S0[:, :m], S1[:, :m]
    dmag = np.abs(S1) - np.abs(S0)
    dcomplex = S1 - S0
    num = float(np.sum(dmag ** 2))
    den = float(np.sum(np.abs(dcomplex) ** 2))
    return num / den if den > 0 else 0.0


_MELB = {}


def mel_invariant_fraction(x: np.ndarray, x_wm: np.ndarray, n_mels: int = 80) -> float:
    """Fraction of the watermark's STFT change energy in the mel envelope (the vocoder invariant).

    f = || (Mp Mb)(|S1|−|S0|) ||² / || S1 − S0 ||²  ∈ [0, 1] — coarse (mel-representable)
    magnitude change over total complex change (phase + sub-mel detail count as nullspace).
    """
    import librosa
    if n_mels not in _MELB:
        Mb = librosa.filters.mel(sr=16000, n_fft=N_FFT, n_mels=n_mels).astype(np.float32)
        _MELB[n_mels] = (Mb, np.linalg.pinv(Mb).astype(np.float32))
    Mb, Mp = _MELB[n_mels]
    S0, S1 = _stft(x), _stft(x_wm)
    m = min(S0.shape[1], S1.shape[1])
    S0, S1 = S0[:, :m], S1[:, :m]
    dmag = np.abs(S1) - np.abs(S0)
    coarse = Mp @ (Mb @ dmag)              # project magnitude change onto mel-representable set
    num = float(np.sum(coarse ** 2))
    den = float(np.sum(np.abs(S1 - S0) ** 2))
    return num / den if den > 0 else 0.0


def detector_survival(delta_before: float, delta_after: float) -> float:
    """Bounded fraction of the detector's mean watermark response surviving the attack."""
    if abs(delta_before) < 1e-12:
        return 0.0
    return float(np.clip(delta_after / delta_before, -0.2, 1.2))


def snr_db(clean: np.ndarray, perturbed: np.ndarray) -> float:
    """SNR of the watermark perturbation, in dB (imperceptibility proxy)."""
    clean = np.asarray(clean, dtype=float)
    noise = np.asarray(perturbed, dtype=float) - clean
    ps = float(np.sum(clean ** 2)) + 1e-12
    pn = float(np.sum(noise ** 2)) + 1e-12
    return 10.0 * np.log10(ps / pn)


def pesq_wb(clean: np.ndarray, perturbed: np.ndarray, sr: int = 16000) -> float:
    """Wideband PESQ MOS-LQO (perceptual quality of the watermarked signal). NaN on failure."""
    try:
        from pesq import pesq
        return float(pesq(sr, np.asarray(clean, dtype=np.float32),
                          np.asarray(perturbed, dtype=np.float32), "wb"))
    except Exception:
        return float("nan")


def auc_bootstrap_ci(scores_pos, scores_neg, n_boot: int = 1000, alpha: float = 0.05,
                     rng: np.random.Generator | None = None):
    """Percentile bootstrap CI for the AUC (resampling utterances with replacement)."""
    rng = np.random.default_rng(0) if rng is None else rng
    pos = np.asarray(scores_pos, dtype=float)
    neg = np.asarray(scores_neg, dtype=float)
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        p = pos[rng.integers(0, pos.size, pos.size)]
        n = neg[rng.integers(0, neg.size, neg.size)]
        aucs[b] = empirical_auc(p, n)
    lo, hi = np.quantile(aucs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
