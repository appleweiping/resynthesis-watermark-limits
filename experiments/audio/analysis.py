"""THE unified analysis map A: one STFT/mel implementation shared by everything.

Every component that refers to "the mel analysis" — the mel-inversion attacker, the
invariant/row-space mark construction, the nullspace-mark verification, and the
mel-fraction predictor — MUST import this module and use the same `MelAnalysis`
instance parameters. No librosa/torchaudio mixing, no per-module parameter drift.

Definition (fixed for the whole paper):
    sr=16000, n_fft=1024, win=1024 (Hann), hop=256, center=True,
    n_mels=80, fmin=0, fmax=8000, mel_scale='slaney', norm='slaney', power=1
    (magnitude mel: A(x) = FB @ |STFT(x)|).

For attackers with their OWN analysis (Vocos 24k/100-mel, EnCodec/DAC/SNAC latents),
this A is NOT reused; channel-relative sensitivity lives in attackers.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

SR = 16_000
N_FFT = 1024
HOP = 256
N_MELS = 80
F_MIN = 0.0
F_MAX = 8000.0


@dataclass
class MelAnalysis:
    """Magnitude-mel analysis A(x) = FB @ |STFT(x)| with one fixed parameterization."""

    device: str = "cpu"
    sr: int = SR
    n_fft: int = N_FFT
    hop: int = HOP
    n_mels: int = N_MELS
    _win: torch.Tensor = field(init=False, repr=False)
    _fb: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        import torchaudio

        self._win = torch.hann_window(self.n_fft, device=self.device)
        # Slaney scale + Slaney area-normalization (librosa-compatible), one source.
        self._fb = torchaudio.functional.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=F_MIN,
            f_max=F_MAX,
            n_mels=self.n_mels,
            sample_rate=self.sr,
            norm="slaney",
            mel_scale="slaney",
        ).to(self.device)                                   # (n_freqs, n_mels)

    # ---- primitives -----------------------------------------------------------
    def stft(self, x: torch.Tensor) -> torch.Tensor:
        """Complex STFT, shape (freq, frames). x: (T,) on self.device."""
        return torch.stft(
            x, self.n_fft, hop_length=self.hop, win_length=self.n_fft,
            window=self._win, center=True, return_complex=True,
        )

    def istft(self, X: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            X, self.n_fft, hop_length=self.hop, win_length=self.n_fft,
            window=self._win, center=True, length=length,
        )

    def mel_of_mag(self, mag: torch.Tensor) -> torch.Tensor:
        """FB^T @ |STFT|: (freq, frames) -> (n_mels, frames)."""
        return self._fb.T @ mag

    def mel(self, x: torch.Tensor) -> torch.Tensor:
        """A(x): magnitude mel, shape (n_mels, frames)."""
        return self.mel_of_mag(self.stft(x).abs())

    @property
    def fb(self) -> torch.Tensor:
        """(n_freqs, n_mels) Slaney filterbank."""
        return self._fb

    # ---- measurements ----------------------------------------------------------
    def analysis_change(self, x: torch.Tensor, delta: torch.Tensor) -> float:
        """||A(x+delta) - A(x)||_F / ||delta||_2 — the nullspace-verification ratio."""
        num = torch.linalg.norm(self.mel(x + delta) - self.mel(x))
        den = torch.linalg.norm(delta).clamp_min(1e-12)
        return float(num / den)

    def mel_fraction(self, x: torch.Tensor, delta: torch.Tensor) -> float:
        """Fraction of the STFT-magnitude change captured by the mel filterbank.

        Measures ||FB_proj(dMag)||^2/||dMag||^2 where dMag = |STFT(x+d)|-|STFT(x)| and
        FB_proj is the orthogonal projector onto the row space of the mel filterbank.
        This is the channel-relative preserved fraction FOR A MEL-READING CHANNEL ONLY.
        """
        d_mag = (self.stft(x + delta).abs() - self.stft(x).abs())  # (freq, frames)
        fb = self._fb                                              # (freq, mels)
        # Orthogonal projection of each frame's dMag onto span(fb columns).
        gram = fb.T @ fb + 1e-10 * torch.eye(fb.shape[1], device=fb.device)
        coef = torch.linalg.solve(gram, fb.T @ d_mag)              # (mels, frames)
        proj = fb @ coef
        num = float(torch.sum(proj**2))
        den = float(torch.sum(d_mag**2)) + 1e-20
        return num / den
