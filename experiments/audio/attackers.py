"""Resynthesis-channel attackers W = S∘A for real speech (16 kHz mono).

Each attacker exposes ``apply(wav)`` at 16 kHz and instantiates a blind launderer with a
different analysis map A (hence a different nullspace):

  * ``StftGriffinLim`` — A = full linear-STFT magnitude.  A near-lossless *control*: its
    nullspace is only the phase, so it launders little (the small-nullspace regime).
  * ``MelGriffinLim``  — A = mel-spectrogram (n_mels bins) magnitude, S = mel→linear
    (filterbank pseudo-inverse) + Griffin-Lim.  A genuinely lossy vocoder-style channel:
    its nullspace is phase *plus* all sub-mel spectral detail (the large-nullspace regime).
  * ``EncodecAttacker`` — A = EnCodec encoder codes, S = decoder; lower bandwidth ⇒ larger
    nullspace ⇒ more laundering.

The theory predicts laundering strength grows with the size of $\\ker A$; these attackers
sweep exactly that axis.
"""

from __future__ import annotations

import numpy as np
import torch

SR = 16000
N_FFT = 1024
HOP = 256


def _t(wav, device):
    return torch.as_tensor(np.asarray(wav, dtype=np.float32), device=device)


def _fit(y, ref_len):
    out = np.zeros(ref_len, dtype=np.float32)
    n = min(len(y), ref_len)
    out[:n] = np.asarray(y, dtype=np.float32)[:n]
    return out


class StftGriffinLim:
    """Near-lossless control: full linear-STFT magnitude, phase re-derived."""

    name = "stft_gl"
    sample_rate = SR

    def __init__(self, n_iter: int = 60, device: str = "cpu"):
        import torchaudio
        self.device = device
        self.spec = torchaudio.transforms.Spectrogram(N_FFT, hop_length=HOP, power=1.0).to(device)
        self.gl = torchaudio.transforms.GriffinLim(N_FFT, hop_length=HOP, power=1.0,
                                                   n_iter=n_iter).to(device)

    @torch.no_grad()
    def apply(self, wav):
        x = _t(wav, self.device)
        y = self.gl(self.spec(x))
        return _fit(y.cpu().numpy(), len(wav))


class MelGriffinLim:
    """Lossy vocoder-style channel: A = mel magnitude, S = mel→linear + Griffin-Lim."""

    sample_rate = SR

    def __init__(self, n_mels: int = 80, n_iter: int = 60, device: str = "cpu"):
        import torchaudio
        self.device = device
        self.name = f"mel{n_mels}_gl"
        self.spec = torchaudio.transforms.Spectrogram(N_FFT, hop_length=HOP, power=1.0).to(device)
        fb = torchaudio.functional.melscale_fbanks(
            N_FFT // 2 + 1, f_min=0.0, f_max=SR / 2, n_mels=n_mels, sample_rate=SR)  # (F, M)
        self.fb = fb.to(device)                       # linear->mel: mel = fb^T @ mag
        self.fb_pinv = torch.linalg.pinv(fb).to(device)  # mel->linear: (M, F)
        self.gl = torchaudio.transforms.GriffinLim(N_FFT, hop_length=HOP, power=1.0,
                                                   n_iter=n_iter).to(device)

    @torch.no_grad()
    def apply(self, wav):
        x = _t(wav, self.device)
        mag = self.spec(x)                    # (F, T)
        mel = self.fb.T @ mag                 # (M, T)  analysis: keep mel envelope
        lin = (self.fb_pinv.T @ mel).clamp_min(0.0)  # synthesis: back to linear magnitude
        y = self.gl(lin)
        return _fit(y.cpu().numpy(), len(wav))


class NeuralVocoderAttacker:
    """Real neural vocoder: A = the vocoder's mel, S = its learned decoder (Vocos).

    Vocos is a high-fidelity GAN-based neural vocoder; unlike Griffin-Lim it *learns* to
    resynthesize the waveform (phase and fine detail) from the mel envelope, so this is the
    genuinely generative regime the theory targets.  Operates via 16->24->16 kHz resampling.
    """

    sample_rate = SR
    name = "vocos"

    def __init__(self, device: str = "cpu"):
        import torchaudio
        from vocos import Vocos
        self.device = device
        self.model = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        self.model = self.model.to(device).eval()
        self.up = torchaudio.transforms.Resample(SR, 24000).to(device)
        self.dn = torchaudio.transforms.Resample(24000, SR).to(device)

    @torch.no_grad()
    def apply(self, wav):
        x = _t(wav, self.device).view(1, -1)
        x24 = self.up(x)
        mel = self.model.feature_extractor(x24)      # A: vocoder mel
        y24 = self.model.decode(mel)                 # S: learned resynthesis
        y = self.dn(y24).view(-1)
        return _fit(y.cpu().numpy(), len(wav))


class EncodecAttacker:
    """EnCodec 24 kHz neural-codec round-trip at a chosen bandwidth (kbps)."""

    sample_rate = SR

    def __init__(self, bandwidth: float = 6.0, device: str = "cpu"):
        import torchaudio
        from encodec import EncodecModel
        self.device = device
        self.name = f"encodec{bandwidth:g}k"
        self.model = EncodecModel.encodec_model_24khz().to(device)
        self.model.set_target_bandwidth(bandwidth)
        self.model.eval()
        self.up = torchaudio.transforms.Resample(SR, 24000).to(device)
        self.dn = torchaudio.transforms.Resample(24000, SR).to(device)

    @torch.no_grad()
    def apply(self, wav):
        x = _t(wav, self.device).view(1, 1, -1)
        enc = self.model.encode(self.up(x))
        y = self.dn(self.model.decode(enc)).view(-1)
        return _fit(y.cpu().numpy(), len(wav))


def build_attackers(device: str = "cpu", include_control: bool = True) -> list:
    att = []
    if include_control:
        try:
            att.append(StftGriffinLim(device=device))
        except Exception as exc:  # pragma: no cover
            print(f"[attackers] StftGriffinLim unavailable: {exc}")
    try:
        att.append(MelGriffinLim(device=device))
    except Exception as exc:  # pragma: no cover
        print(f"[attackers] MelGriffinLim unavailable: {exc}")
    try:
        att.append(NeuralVocoderAttacker(device=device))
    except Exception as exc:  # pragma: no cover
        print(f"[attackers] NeuralVocoder (Vocos) unavailable: {exc}")
    for bw in (6.0, 3.0, 1.5):
        try:
            att.append(EncodecAttacker(bandwidth=bw, device=device))
        except Exception as exc:  # pragma: no cover
            print(f"[attackers] EnCodec {bw}k unavailable: {exc}")
            break
    return att
