"""E3 — multi-bit PROOF OF CONCEPT (no rate/capacity claim).

A B-bit payload is embedded in the unified mel domain with frame-differential BPSK:
bit i occupies a block of frames and sign-modulates a keyed mel pattern with
alternating polarity across frame pairs. The decoder is BLIND (no host, no genie):
it reads A(y), takes even-odd frame differences within each block (host mel is
locally smooth, so the host largely cancels), and correlates with the keyed pattern.

Reported: BER and block-error rate (payload-level) with Clopper-Pearson CIs, per
attacker, at the matched perceptual budget. This is a demonstration that the
invariant sub-channel can carry bits through resynthesis — NOT a measurement of an
achievable rate; no bits/s at target reliability is claimed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .analysis import MelAnalysis
from .attackers import build_attackers
from .data_io import iter_split, load_manifest
from .marks import scale_to_pesq
from .metrics_audio import clopper_pearson, pesq_wb

SR = 16_000
ROOT = Path(__file__).resolve().parents[2]
ATTACKERS = ["mel80_gl", "vocos", "encodec6k", "dac", "snac"]


def _keyed_pattern(an: MelAnalysis, key: int, device) -> torch.Tensor:
    """Smooth keyed spectral pattern: random combination of low-order cosine
    profiles over mel bins — survives codec spectral smearing far better than a
    per-bin random pattern."""
    g = torch.Generator(device="cpu"); g.manual_seed(key)
    coeff = torch.randn(8, generator=g)
    bins = torch.arange(an.n_mels, dtype=torch.float32)
    basis = torch.stack([torch.cos(np.pi * (k + 1) * (bins + 0.5) / an.n_mels)
                         for k in range(8)])           # (8, n_mels)
    pattern = (coeff[:, None] * basis).sum(0)
    return (pattern / torch.linalg.norm(pattern)).to(device)


def embed_payload(an: MelAnalysis, x: torch.Tensor, key: int, bits: np.ndarray,
                  pesq_target: float) -> tuple[torch.Tensor, float, float]:
    X = an.stft(x)
    mag, phase = X.abs(), torch.angle(X)
    n_frames = X.shape[-1]
    B = len(bits)
    block = n_frames // B
    pattern = _keyed_pattern(an, key, x.device)
    D = torch.zeros_like(X.real)
    for i, b in enumerate(bits):
        sl = slice(i * block, (i + 1) * block)
        pol = torch.ones(block, device=x.device)
        pol[1::2] = -1.0                      # frame-differential polarity
        sgn = 1.0 if b else -1.0
        pat = (an.fb @ pattern[:, None]) * pol[None, :]      # (freq, block)
        D[:, sl] = sgn * pat * mag[:, sl]
    delta_dir = an.istft(X + torch.exp(1j * phase) * D, length=x.shape[-1]) - x
    delta, achieved, snr = scale_to_pesq(x, delta_dir, SR, target=pesq_target)
    return x + delta, achieved, snr


def decode_payload(an: MelAnalysis, y: torch.Tensor, key: int, B: int) -> np.ndarray:
    mel = an.mel(y)
    n_frames = mel.shape[-1]
    block = n_frames // B
    pattern = _keyed_pattern(an, key, y.device)
    bits = []
    for i in range(B):
        sl = mel[:, i * block:(i + 1) * block]
        m = sl.shape[-1] - (sl.shape[-1] % 2)
        diff = sl[:, 0:m:2] - sl[:, 1:m:2]        # host cancels, mark adds
        # average over frame pairs, whiten per mel bin (noise-scaled correlation)
        d_mean = diff.mean(dim=1)
        d_std = diff.std(dim=1).clamp_min(1e-6)
        stat = float(torch.sum(pattern * d_mean / d_std))
        bits.append(1 if stat > 0 else 0)
    return np.array(bits, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--pesq-target", type=float, default=4.2)
    ap.add_argument("--strict", action="store_true", default=True)
    ap.add_argument("--no-strict", dest="strict", action="store_false")
    ap.add_argument("--out", default=str(ROOT / "results" / "e3_payload_poc.json"))
    args = ap.parse_args()

    man = load_manifest(args.manifest)
    an = MelAnalysis(device=args.device)
    attackers = build_attackers(ATTACKERS, args.device, args.strict)
    rng = np.random.default_rng(7)

    rows = {a.name: {"bit_err": 0, "bit_tot": 0, "blk_err": 0, "blk_tot": 0}
            for a in attackers}
    rows["none"] = {"bit_err": 0, "bit_tot": 0, "blk_err": 0, "blk_tot": 0}
    pesqs, snrs = [], []
    for i, r, x_np in iter_split(man, "test", args.n_test):
        x = torch.as_tensor(x_np, device=args.device)
        key = 31_000 + i
        bits = rng.integers(0, 2, size=args.bits)
        y, p, s = embed_payload(an, x, key, bits, args.pesq_target)
        pesqs.append(p); snrs.append(s)

        def tally(name: str, y_att: torch.Tensor) -> None:
            dec = decode_payload(an, y_att, key, args.bits)
            err = int(np.sum(dec != bits))
            rows[name]["bit_err"] += err
            rows[name]["bit_tot"] += args.bits
            rows[name]["blk_err"] += int(err > 0)
            rows[name]["blk_tot"] += 1

        tally("none", y)
        for att in attackers:
            tally(att.name, torch.as_tensor(att.apply(y.cpu().numpy()),
                                            device=args.device))
        if (i + 1) % 25 == 0:
            print(f"[E3] {i + 1}/{args.n_test}")

    out = {"bits": args.bits, "n_clips": args.n_test,
           "pesq_median": float(np.median(pesqs)),
           "snr_db_median": float(np.median(snrs)),
           "channels": {}}
    for name, c in rows.items():
        ber = c["bit_err"] / max(1, c["bit_tot"])
        bler = c["blk_err"] / max(1, c["blk_tot"])
        out["channels"][name] = {
            "ber": ber, "ber_ci": list(clopper_pearson(c["bit_err"], c["bit_tot"])),
            "bler": bler, "bler_ci": list(clopper_pearson(c["blk_err"], c["blk_tot"])),
        }
        print(f"[E3] {name}: BER={ber:.3f} BLER={bler:.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"[E3] wrote {args.out}")


if __name__ == "__main__":
    main()
