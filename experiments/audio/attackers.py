"""Analysis-resynthesis attackers W = S∘A for real speech (16 kHz mono).

Each attacker exposes:
  * ``apply(wav)``  — attack a 16 kHz float32 waveform, same length out;
  * ``repr_of(x)``  — the attacker's OWN analysis representation A_W(x) (torch);
  * ``sensitivity(x, delta)`` — channel-relative preserved-perturbation measure
    s_W = ||A_W(x+delta) - A_W(x)|| / ||delta||, using THAT attacker's analysis.
    A single mel fraction is NEVER reused across attackers with different A_W.

Attackers:
  * stft_gl      — |STFT| kept, phase re-derived (near-lossless control)
  * mel80_gl     — THE unified mel analysis (analysis.MelAnalysis) + pinv + Griffin-Lim
  * vocos        — Vocos mel->waveform neural vocoder (its own 24k/100-mel analysis)
  * vocos_encodec— Vocos decoding EnCodec tokens (2nd learned reconstruction pipeline)
  * encodec{6,3,1.5}k — EnCodec codec round-trips
  * dac          — Descript Audio Codec 16 kHz round-trip
  * snac         — SNAC 24 kHz round-trip
  * knnvc_self{4,8} — kNN-VC self-voice-conversion (WavLM features -> kNN match ->
                   HiFi-GAN), topk 4 / 8: genuine content/speaker-preserving
                   re-synthesis pipelines.

Registry: ``build_attackers(names, device, strict=True)`` RAISES if a requested
attacker cannot be constructed — formal runs must fail loudly, never silently skip.
"""

from __future__ import annotations

import numpy as np
import torch

from .analysis import MelAnalysis, HOP, N_FFT, SR


def _t(wav, device):
    return torch.as_tensor(np.asarray(wav, dtype=np.float32), device=device)


def _fit(y, ref_len):
    out = np.zeros(ref_len, dtype=np.float32)
    n = min(len(y), ref_len)
    out[:n] = np.asarray(y, dtype=np.float32)[:n]
    return out


class _Base:
    sample_rate = SR
    name = "base"
    device = "cpu"

    def apply(self, wav: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def repr_of(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    @torch.no_grad()
    def sensitivity(self, x: torch.Tensor, delta: torch.Tensor) -> float:
        """||A_W(x+d) - A_W(x)||_F / ||d|| with THIS attacker's analysis."""
        r0 = self.repr_of(x)
        r1 = self.repr_of(x + delta)
        return float(torch.linalg.norm((r1 - r0).float())
                     / torch.linalg.norm(delta).clamp_min(1e-12))


class StftGriffinLim(_Base):
    """Near-lossless control: full linear-STFT magnitude kept, phase re-derived."""

    name = "stft_gl"

    def __init__(self, n_iter: int = 32, device: str = "cpu"):
        import torchaudio

        self.device = device
        self.an = MelAnalysis(device=device)
        self.gl = torchaudio.transforms.GriffinLim(
            N_FFT, hop_length=HOP, power=1.0, n_iter=n_iter).to(device)

    @torch.no_grad()
    def apply(self, wav):
        x = _t(wav, self.device)
        y = self.gl(self.an.stft(x).abs())
        return _fit(y.cpu().numpy(), len(wav))

    def repr_of(self, x):
        return self.an.stft(x).abs()


class MelGriffinLim(_Base):
    """Lossy spectral-inversion channel built on THE unified mel analysis."""

    name = "mel80_gl"

    def __init__(self, n_iter: int = 32, device: str = "cpu"):
        import torchaudio

        self.device = device
        self.an = MelAnalysis(device=device)
        self.fb_pinv = torch.linalg.pinv(self.an.fb).to(device)   # (mels, freqs)
        self.gl = torchaudio.transforms.GriffinLim(
            N_FFT, hop_length=HOP, power=1.0, n_iter=n_iter).to(device)

    @torch.no_grad()
    def apply(self, wav):
        x = _t(wav, self.device)
        mel = self.an.mel(x)                                # unified A
        lin = (self.fb_pinv.T @ mel).clamp_min(0.0)
        y = self.gl(lin)
        return _fit(y.cpu().numpy(), len(wav))

    def repr_of(self, x):
        return self.an.mel(x)


class VocosAttacker(_Base):
    """Vocos neural vocoder: A = ITS OWN 24 kHz/100-mel features, S = learned decoder."""

    name = "vocos"

    def __init__(self, device: str = "cpu"):
        import torchaudio
        from vocos import Vocos

        self.device = device
        self.model = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
        self.up = torchaudio.transforms.Resample(SR, 24000).to(device)
        self.dn = torchaudio.transforms.Resample(24000, SR).to(device)

    @torch.no_grad()
    def apply(self, wav):
        x = _t(wav, self.device).view(1, -1)
        mel = self.model.feature_extractor(self.up(x))
        y = self.dn(self.model.decode(mel)).view(-1)
        return _fit(y.cpu().numpy(), len(wav))

    @torch.no_grad()
    def repr_of(self, x):
        return self.model.feature_extractor(self.up(x.view(1, -1)))


class VocosEncodecAttacker(_Base):
    """Second learned reconstruction pipeline: Vocos decoding EnCodec tokens."""

    def __init__(self, bandwidth: float = 6.0, device: str = "cpu"):
        import torchaudio
        from vocos import Vocos

        self.device = device
        self.name = f"vocos_encodec{bandwidth:g}k"
        self.bandwidth = bandwidth
        self.model = Vocos.from_pretrained("charactr/vocos-encodec-24khz").to(device).eval()
        self.up = torchaudio.transforms.Resample(SR, 24000).to(device)
        self.dn = torchaudio.transforms.Resample(24000, SR).to(device)

    @torch.no_grad()
    def apply(self, wav):
        x = _t(wav, self.device).view(1, -1)
        bw = torch.tensor([{1.5: 0, 3.0: 1, 6.0: 2, 12.0: 3}[self.bandwidth]],
                          device=self.device)
        y24 = self.model(self.up(x), bandwidth_id=bw)
        y = self.dn(y24).view(-1)
        return _fit(y.cpu().numpy(), len(wav))

    @torch.no_grad()
    def repr_of(self, x):
        return self.model.feature_extractor(self.up(x.view(1, -1)))


class EncodecAttacker(_Base):
    """EnCodec 24 kHz neural-codec round-trip; repr = continuous encoder output."""

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

    @torch.no_grad()
    def repr_of(self, x):
        return self.model.encoder(self.up(x.view(1, 1, -1)))


class DacAttacker(_Base):
    """Descript Audio Codec (16 kHz) round-trip; repr = continuous encoder output."""

    name = "dac"

    def __init__(self, device: str = "cpu"):
        import dac as dac_lib

        self.device = device
        path = dac_lib.utils.download(model_type="16khz")
        self.model = dac_lib.DAC.load(path).to(device).eval()

    @torch.no_grad()
    def apply(self, wav):
        x = _t(wav, self.device).view(1, 1, -1)
        x = self.model.preprocess(x, SR)
        z, *_ = self.model.encode(x)
        y = self.model.decode(z).view(-1)
        return _fit(y.cpu().numpy(), len(wav))

    @torch.no_grad()
    def repr_of(self, x):
        xx = self.model.preprocess(x.view(1, 1, -1), SR)
        return self.model.encoder(xx)


class SnacAttacker(_Base):
    """SNAC 24 kHz round-trip; repr = continuous encoder output."""

    name = "snac"

    def __init__(self, device: str = "cpu"):
        import torchaudio
        from snac import SNAC

        self.device = device
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
        self.up = torchaudio.transforms.Resample(SR, 24000).to(device)
        self.dn = torchaudio.transforms.Resample(24000, SR).to(device)

    @torch.no_grad()
    def apply(self, wav):
        x = _t(wav, self.device).view(1, 1, -1)
        codes = self.model.encode(self.up(x))
        y = self.dn(self.model.decode(codes)).view(-1)
        return _fit(y.cpu().numpy(), len(wav))

    @torch.no_grad()
    def repr_of(self, x):
        return self.model.encoder(self.up(x.view(1, 1, -1)))


class KnnVcSelfAttacker(_Base):
    """kNN-VC self-conversion: WavLM features -> kNN match (self set) -> HiFi-GAN.

    A genuine content/speaker/prosody-preserving RE-SYNTHESIS: every output sample is
    generated by the vocoder from matched self features; nothing of the input waveform
    passes through directly. topk controls how much feature averaging (regeneration).
    """

    def __init__(self, topk: int = 4, device: str = "cpu"):
        import os

        self.device = device
        self.topk = topk
        self.name = f"knnvc_self{topk}"
        # Prefer a local clone (checkpoints pre-fetched into the torch hub cache);
        # the GitHub fetch inside torch.hub is unreliable from this network.
        local = os.environ.get("KNNVC_REPO", "/root/autodl-tmp/knn-vc")
        if os.path.isdir(local):
            self.model = torch.hub.load(
                local, "knn_vc", source="local", prematched=True,
                pretrained=True, device=device)
        else:
            self.model = torch.hub.load(
                "bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True,
                pretrained=True, device=device)

    @torch.no_grad()
    def apply(self, wav):
        import tempfile, soundfile as sf, os

        x = np.asarray(wav, dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "in.wav")
            sf.write(p, x, SR)
            # vad_trigger_level=0 disables kNN-VC's silence trimming: manifest
            # clips are already voice-active, and the VAD can trim low-energy
            # clips to zero length (WavLM conv crash). Self matching set = the
            # clip's own features (same semantics as get_matching_set([p])).
            q = self.model.get_features(p, vad_trigger_level=0)
            y = self.model.match(q, q, topk=self.topk)
        y = y.cpu().numpy().ravel()
        # level-match to the input so the channel effect is not confounded with gain
        rms_in = float(np.sqrt(np.mean(x**2)) + 1e-12)
        rms_out = float(np.sqrt(np.mean(y**2)) + 1e-12)
        return _fit(y * (rms_in / rms_out), len(wav))

    @torch.no_grad()
    def repr_of(self, x):
        import tempfile, soundfile as sf, os

        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "in.wav")
            sf.write(p, x.detach().cpu().numpy().astype(np.float32), SR)
            return self.model.get_features(p, vad_trigger_level=0)


_FACTORIES = {
    "stft_gl": lambda device: StftGriffinLim(device=device),
    "mel80_gl": lambda device: MelGriffinLim(device=device),
    "vocos": lambda device: VocosAttacker(device=device),
    "vocos_encodec6k": lambda device: VocosEncodecAttacker(6.0, device),
    "encodec6k": lambda device: EncodecAttacker(6.0, device),
    "encodec3k": lambda device: EncodecAttacker(3.0, device),
    "encodec1.5k": lambda device: EncodecAttacker(1.5, device),
    "dac": lambda device: DacAttacker(device=device),
    "snac": lambda device: SnacAttacker(device=device),
    "knnvc_self4": lambda device: KnnVcSelfAttacker(4, device),
    "knnvc_self8": lambda device: KnnVcSelfAttacker(8, device),
}

FULL_SET = list(_FACTORIES)


def build_attackers(names: list[str], device: str = "cpu", strict: bool = True) -> list:
    """Construct the requested attackers. strict=True (formal runs): missing attacker
    RAISES RuntimeError — results must never silently drop a channel."""
    out, errors = [], []
    for n in names:
        if n not in _FACTORIES:
            raise KeyError(f"unknown attacker '{n}'; known: {FULL_SET}")
        try:
            out.append(_FACTORIES[n](device))
        except Exception as exc:
            errors.append(f"{n}: {type(exc).__name__}: {exc}")
    if errors:
        msg = "attacker construction FAILED:\n  " + "\n  ".join(errors)
        if strict:
            raise RuntimeError(msg)
        print("[attackers] WARNING (non-strict dev mode):", msg)
    return out
