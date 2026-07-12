"""Watermarks and blind detectors for real speech (16 kHz mono).

Three embedders, keyed by a secret integer ``key``:

  * ``SurfaceSpreadSpectrum``   — a fine high-band time-domain carrier detected by waveform
    correlation.  Its energy is phase/fine-structure (the *surface / nullspace*): a
    magnitude-preserving resynthesizer re-derives it away.  Dies under laundering.
  * ``MagnitudeSpreadSpectrum`` — a coarse multiplicative pattern in the STFT *magnitude*
    (the invariant a vocoder/codec preserves), detected by whitened magnitude correlation.
    The constructive realization of the achievability theorem; survives.
  * ``AudioSealWatermark``      — Meta's deployed neural post-hoc watermark (real baseline).

Surface and magnitude marks are scaled to a common waveform SNR (equal imperceptibility).
Detectors are blind (no clean reference), matching the threat model.
"""

from __future__ import annotations

import numpy as np

N_FFT = 1024
HOP = 256
SR = 16000
EPS = 1e-8


def _stft(x):
    import librosa
    return librosa.stft(np.asarray(x, dtype=np.float32), n_fft=N_FFT, hop_length=HOP)


def _istft(S, length):
    import librosa
    return librosa.istft(S, hop_length=HOP, length=length).astype(np.float32)


def _snr(x, x_wm):
    return 10 * np.log10((np.sum(x ** 2) + 1e-12) / (np.sum((x_wm - x) ** 2) + 1e-12))


_MEL_CACHE = {}


def _mel_basis(n_mels=80):
    import librosa
    if n_mels not in _MEL_CACHE:
        Mb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=n_mels).astype(np.float32)  # (M,F)
        _MEL_CACHE[n_mels] = (Mb, np.linalg.pinv(Mb).astype(np.float32))               # (F,M)
    return _MEL_CACHE[n_mels]


# ------------------------------------------------------- surface (nullspace) SS
class SurfaceSpreadSpectrum:
    """Fine high-band time-domain carrier — perceptually cheap, laundered away."""

    name = "surface_ss"

    def __init__(self, snr_db: float = 24.0, band_hz: float = 5000.0):
        self.snr_db = snr_db
        self.band_hz = band_hz

    def _carrier(self, length: int, key: int) -> np.ndarray:
        from scipy.signal import butter, sosfilt
        rng = np.random.default_rng(key)
        c = rng.standard_normal(length).astype(np.float32)
        sos = butter(8, self.band_hz / (SR / 2), btype="highpass", output="sos")
        c = sosfilt(sos, c).astype(np.float32)
        return c / (np.linalg.norm(c) + 1e-12)

    def embed(self, x: np.ndarray, key: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        c = self._carrier(x.size, key)
        ps = float(np.sum(x ** 2))
        target = ps / (10 ** (self.snr_db / 10))
        return x + c * np.sqrt(target)  # ||c||=1

    def score(self, y: np.ndarray, key: int) -> float:
        y = np.asarray(y, dtype=np.float32)
        c = self._carrier(y.size, key)
        n = min(y.size, c.size)
        return float(np.dot(y[:n], c[:n]) / (np.linalg.norm(y[:n]) + 1e-12))


# ------------------------------------------------------ mel-envelope (invariant) SS
class MagnitudeSpreadSpectrum:
    """Coarse multiplicative pattern in the mel envelope — the vocoder/codec invariant.

    Embedded as a smooth mel-domain gain applied to the *true* magnitude (fine detail is
    preserved, so imperceptible), then detected by whitened mel-envelope correlation.
    """

    name = "invariant_ss"

    def __init__(self, snr_db: float = 24.0, n_mels: int = 80, m_lo: int = 4,
                 m_hi: int = 74, bm: int = 2, bt: int = 4):
        self.snr_db, self.n_mels = snr_db, n_mels
        self.m_lo, self.m_hi = m_lo, m_hi
        self.bm, self.bt = bm, bt   # mel/time block sizes (coarseness -> codec robustness)

    def _pattern(self, T: int, key: int) -> np.ndarray:
        rng = np.random.default_rng(key)
        P = np.zeros((self.n_mels, T), dtype=np.float32)
        nm = self.m_hi - self.m_lo
        base = rng.standard_normal((nm // self.bm + 1, T // self.bt + 1)).astype(np.float32)
        up = np.kron(base, np.ones((self.bm, self.bt), dtype=np.float32))[:nm, :T]
        P[self.m_lo:self.m_hi, :] = up
        P -= P.mean()
        P /= P.std() + 1e-12   # unit per-cell std: detection deflection grows as sqrt(DOF)
        return P

    def _gain_from_pattern(self, mag, P, a):
        """Smooth per-linear-bin gain realizing the mel-envelope perturbation ``a*P``."""
        Mb, Mp = _mel_basis(self.n_mels)
        mel = Mb @ mag                              # (M,T)
        Pt = P[:, :mel.shape[1]]
        recon0 = Mp @ mel                           # linear reconstruction of the envelope
        recon1 = Mp @ (mel * (1.0 + a * Pt))
        return np.clip(recon1 / (recon0 + 1e-6), 0.3, 3.0)

    def _gain(self, mag, key, a):
        return self._gain_from_pattern(mag, self._pattern(mag.shape[1], key), a)

    def embed(self, x: np.ndarray, key: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        S = _stft(x)
        mag, phase = np.abs(S), np.angle(S)
        lo, hi, x_wm = 0.0, 4.0, x
        for _ in range(20):
            a = 0.5 * (lo + hi)
            x_wm = _istft(mag * self._gain(mag, key, a) * np.exp(1j * phase), x.size)
            if _snr(x, x_wm) > self.snr_db:
                lo = a
            else:
                hi = a
        return x_wm

    def score(self, y: np.ndarray, key: int) -> float:
        Mb, _ = _mel_basis(self.n_mels)
        mel = Mb @ np.abs(_stft(np.asarray(y, dtype=np.float32)))
        logmel = np.log(mel + 1e-6)
        # Double-center: remove host spectral shape (per-mel time-mean) and level (per-frame).
        resid = (logmel - logmel.mean(axis=1, keepdims=True)
                 - logmel.mean(axis=0, keepdims=True) + logmel.mean())
        P = self._pattern(resid.shape[1], key)
        T = min(P.shape[1], resid.shape[1])
        return float(np.sum(resid[:, :T] * P[:, :T]))


# --------------------------------------------------- mixture family (f-sweep)
class MixedSpreadSpectrum:
    """Blend of magnitude (invariant) and surface (nullspace) energy, fraction ``beta``.

    Embeds ``sqrt(beta)`` of its energy as a magnitude-modulation direction and
    ``sqrt(1-beta)`` as a surface carrier, at a fixed waveform SNR.  Sweeping ``beta``
    sweeps the invariant-energy fraction ``f`` from ~0 (surface) to ~1 (invariant); the
    two-component detector fires on both before laundering but only on the magnitude part
    after — realizing the converse's monotone AUC-vs-``f`` curve on real speech.
    """

    def __init__(self, beta: float, snr_db: float = 24.0):
        self.beta = float(beta)
        self.snr_db = snr_db
        self.name = f"mixed{beta:.2f}"
        self._mag = MagnitudeSpreadSpectrum(snr_db=snr_db)
        self._surf = SurfaceSpreadSpectrum(snr_db=snr_db)

    def _mag_direction(self, x: np.ndarray, key: int) -> np.ndarray:
        """Unit-norm mel-envelope perturbation direction for (x, key)."""
        S = _stft(x)
        mag, phase = np.abs(S), np.angle(S)
        g = self._mag._gain(mag, key, 0.15)
        x_wm = _istft(mag * g * np.exp(1j * phase), x.size)
        d = x_wm - x
        return d / (np.linalg.norm(d) + 1e-12)

    def embed(self, x: np.ndarray, key: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        d_mag = self._mag_direction(x, key)
        d_surf = self._surf._carrier(x.size, key)
        n = min(d_mag.size, d_surf.size)
        delta = np.sqrt(self.beta) * d_mag[:n] + np.sqrt(1 - self.beta) * d_surf[:n]
        ps = float(np.sum(x[:n] ** 2))
        target = ps / (10 ** (self.snr_db / 10))
        delta = delta * np.sqrt(target / (np.sum(delta ** 2) + 1e-12))
        out = x.copy()
        out[:n] = x[:n] + delta
        return out

    def score_components(self, y: np.ndarray, key: int):
        """Return (magnitude-detector score, surface-detector score)."""
        return self._mag.score(y, key), self._surf.score(y, key)


# --------------------------------------------------------------------- AudioSeal
class AudioSealWatermark:
    """Wrapper for Meta's AudioSeal 16-bit neural post-hoc watermark."""

    name = "audioseal"

    def __init__(self, device: str = "cpu"):
        from audioseal import AudioSeal
        self.device = device
        self.gen = AudioSeal.load_generator("audioseal_wm_16bits").to(device)
        self.det = AudioSeal.load_detector("audioseal_detector_16bits").to(device)

    def _msg(self, key: int):
        import torch
        rng = np.random.default_rng(key)
        bits = rng.integers(0, 2, size=16)
        return torch.tensor(bits, dtype=torch.int32, device=self.device).unsqueeze(0), bits

    def embed(self, x: np.ndarray, key: int) -> np.ndarray:
        import torch
        msg, _ = self._msg(key)
        wav = torch.as_tensor(np.asarray(x, dtype=np.float32),
                              device=self.device).view(1, 1, -1)
        with torch.no_grad():
            wm = self.gen.get_watermark(wav, sample_rate=SR, message=msg)
            out = (wav + wm).view(-1).cpu().numpy()
        return out.astype(np.float32)

    def score(self, y: np.ndarray, key: int) -> float:
        import torch
        wav = torch.as_tensor(np.asarray(y, dtype=np.float32),
                              device=self.device).view(1, 1, -1)
        with torch.no_grad():
            prob, _ = self.det.detect_watermark(wav, sample_rate=SR)
        return float(prob)

    def bits(self, y: np.ndarray, key: int):
        import torch
        _, truth = self._msg(key)
        wav = torch.as_tensor(np.asarray(y, dtype=np.float32),
                              device=self.device).view(1, 1, -1)
        with torch.no_grad():
            _, msg = self.det.detect_watermark(wav, sample_rate=SR)
        return msg.view(-1).cpu().numpy()[:16], truth
