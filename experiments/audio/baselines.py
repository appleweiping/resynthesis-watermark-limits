"""Deployed watermark baselines with one uniform interface.

Each baseline exposes:
  * ``embed(wav, key)``   -> watermarked wav (np.float32, same length)
  * ``score(wav, key)``   -> host-blind detector score (higher = "watermarked");
                             the score direction is FIXED once (raw orientation)
  * ``strength``          -> scalar in (0, 1]; watermark residual is post-scaled by
                             it so every method can be tuned to the SAME perceptual
                             target (matched PESQ), not matched SNR.

Scores are continuous so AUC/thresholds are meaningful:
  * AudioSeal     — detector watermark probability
  * WavMark       — 16 - Hamming(decoded, keyed payload)  (payload known to detector)
  * SilentCipher  — decoder confidence for the keyed message

Formal runs construct baselines with build_baselines(names, strict=True): a missing
baseline RAISES — never silently skipped.
"""

from __future__ import annotations

import numpy as np
import torch

SR = 16_000


def _np32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _keyed_bits(key: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(key)
    return rng.integers(0, 2, size=n).astype(np.int64)


class AudioSealBaseline:
    name = "audioseal"

    def __init__(self, device: str = "cpu", strength: float = 1.0):
        from audioseal import AudioSeal

        self.device = device
        self.strength = float(strength)
        self.gen = AudioSeal.load_generator("audioseal_wm_16bits").to(device).eval()
        self.det = AudioSeal.load_detector("audioseal_detector_16bits").to(device).eval()

    @torch.no_grad()
    def embed(self, wav: np.ndarray, key: int) -> np.ndarray:
        x = torch.as_tensor(_np32(wav), device=self.device).view(1, 1, -1)
        msg = torch.as_tensor(_keyed_bits(key, 16), device=self.device).view(1, 16)
        delta = self.gen.get_watermark(x, sample_rate=SR, message=msg)
        y = x + self.strength * delta
        return _np32(y.view(-1).cpu().numpy())

    @torch.no_grad()
    def score(self, wav: np.ndarray, key: int) -> float:
        x = torch.as_tensor(_np32(wav), device=self.device).view(1, 1, -1)
        result, _ = self.det.detect_watermark(x, sample_rate=SR)
        return float(result)

    @torch.no_grad()
    def decode_bits(self, wav: np.ndarray) -> np.ndarray | None:
        x = torch.as_tensor(_np32(wav), device=self.device).view(1, 1, -1)
        _, msg = self.det.detect_watermark(x, sample_rate=SR)
        return None if msg is None else msg.view(-1).cpu().numpy().astype(np.int64)


class WavMarkBaseline:
    name = "wavmark"

    def __init__(self, device: str = "cpu", strength: float = 1.0):
        import wavmark

        self.device = device
        self.strength = float(strength)
        self.model = wavmark.load_model().to(device).eval()

    def embed(self, wav: np.ndarray, key: int) -> np.ndarray:
        import wavmark

        x = _np32(wav)
        payload = _keyed_bits(key, 16)
        y, _ = wavmark.encode_watermark(self.model, x, payload, show_progress=False)
        y = _np32(y)
        n = min(len(y), len(x))
        out = x.copy()
        out[:n] = x[:n] + self.strength * (y[:n] - x[:n])   # post-scaled residual
        return out

    def score(self, wav: np.ndarray, key: int) -> float:
        import wavmark

        decoded, _ = wavmark.decode_watermark(self.model, _np32(wav), show_progress=False)
        if decoded is None:
            return 0.0                                     # no sync found: min score
        truth = _keyed_bits(key, 16)
        m = min(len(decoded), 16)
        return float(np.sum(decoded[:m] == truth[:m]))     # 0..16 matches

    def decode_bits(self, wav: np.ndarray) -> np.ndarray | None:
        import wavmark

        decoded, _ = wavmark.decode_watermark(self.model, _np32(wav), show_progress=False)
        return None if decoded is None else np.asarray(decoded, dtype=np.int64)


class SilentCipherBaseline:
    name = "silentcipher"

    def __init__(self, device: str = "cpu", strength: float = 1.0,
                 message_sdr: float = 47.0):
        import silentcipher

        self.device = device
        self.strength = float(strength)
        self.message_sdr = float(message_sdr)
        self.model = silentcipher.get_model(model_type="16k", device=device)

    def _msg(self, key: int) -> list[int]:
        rng = np.random.default_rng(key)
        return [int(v) for v in rng.integers(0, 256, size=5)]

    def embed(self, wav: np.ndarray, key: int) -> np.ndarray:
        x = _np32(wav)
        y, _ = self.model.encode_wav(x, SR, self._msg(key),
                                     message_sdr=self.message_sdr)
        y = _np32(y)
        n = min(len(y), len(x))
        out = x.copy()
        out[:n] = x[:n] + self.strength * (y[:n] - x[:n])
        return out

    def score(self, wav: np.ndarray, key: int) -> float:
        res = self.model.decode_wav(_np32(wav), SR, phase_shift_decoding=False)
        if not res.get("status"):
            return 0.0
        truth = self._msg(key)
        best = 0.0
        for msg, conf in zip(res.get("messages", []), res.get("confidences", [])):
            match = sum(int(a == b) for a, b in zip(msg, truth)) / max(1, len(truth))
            best = max(best, float(conf) * match)
        return best

    def decode_bits(self, wav: np.ndarray) -> np.ndarray | None:
        res = self.model.decode_wav(_np32(wav), SR, phase_shift_decoding=False)
        if not res.get("status") or not res.get("messages"):
            return None
        return np.asarray(res["messages"][0], dtype=np.int64)


_FACTORIES = {
    "audioseal": AudioSealBaseline,
    "wavmark": WavMarkBaseline,
    "silentcipher": SilentCipherBaseline,
}

FULL_SET = list(_FACTORIES)


def build_baselines(names: list[str], device: str = "cpu", strict: bool = True) -> list:
    out, errors = [], []
    for n in names:
        if n not in _FACTORIES:
            raise KeyError(f"unknown baseline '{n}'; known: {FULL_SET}")
        try:
            out.append(_FACTORIES[n](device=device))
        except Exception as exc:
            errors.append(f"{n}: {type(exc).__name__}: {exc}")
    if errors:
        msg = "baseline construction FAILED:\n  " + "\n  ".join(errors)
        if strict:
            raise RuntimeError(msg)
        print("[baselines] WARNING (non-strict dev mode):", msg)
    return out


def calibrate_strength_to_pesq(
    baseline, wavs: list[np.ndarray], target: float = 4.2, tol: float = 0.08,
    key: int = 1234, max_iter: int = 8,
) -> float:
    """Set baseline.strength so median PESQ(x, embed(x)) over `wavs` hits `target`.

    The matched-perceptual-budget condition: every method is tuned to the same
    median PESQ before any attack is applied.
    """
    from pesq import pesq as pesq_fn

    def med_pesq(s: float) -> float:
        baseline.strength = s
        vals = []
        for w in wavs:
            y = baseline.embed(w, key)
            vals.append(pesq_fn(SR, np.asarray(w, np.float64),
                                np.asarray(y, np.float64), "wb"))
        return float(np.median(vals))

    lo, hi = 0.05, 1.0
    p_hi = med_pesq(hi)
    if p_hi >= target - tol:          # already transparent at full strength
        baseline.strength = hi
        return hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p = med_pesq(mid)
        if abs(p - target) <= tol:
            baseline.strength = mid
            return mid
        if p < target:                # too audible -> weaken
            hi = mid
        else:
            lo = mid
    baseline.strength = 0.5 * (lo + hi)
    return baseline.strength
