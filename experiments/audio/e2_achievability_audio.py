r"""E2 — achievability on real speech (GPU): payload survival and the rate-survival curve.

The invariant magnitude watermark carries a multi-bit payload through the *invariant
sub-channel*.  We embed ``B`` bits with ``B`` coarse magnitude patterns (BPSK) and recover
them by whitened correlation, versus a surface (nullspace) payload of the same size at equal
SNR.  We report payload bit-accuracy before vs after laundering, and sweep the imperceptibility
budget (SNR) to trace the surviving rate — positive for the invariant mark (realizing R*),
chance for the surface mark.

Outputs: results/e2_audio.json, paper/figures/fig_e2_rate_survival.pdf.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attackers import EncodecAttacker, MelGriffinLim
from metrics_audio import bit_accuracy, snr_db
from watermarks_audio import (
    _istft, _mel_basis, _stft, MagnitudeSpreadSpectrum, SurfaceSpreadSpectrum,
)

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "paper" / "figures"
RESDIR = ROOT / "results"
SR = 16000
CLIP_SEC = 3.0


class PayloadMagnitude:
    """B-bit invariant payload in the mel envelope, BPSK, double-centered-corr decode."""

    name = "magnitude"

    def __init__(self, nbits: int = 16, snr_db: float = 28.0):
        self.nbits, self.snr_db = nbits, snr_db
        self._base = MagnitudeSpreadSpectrum(snr_db=snr_db)

    def _patterns(self, T, key):
        return [self._base._pattern(T, key * 131 + b + 1) for b in range(self.nbits)]

    def embed(self, x, key, bits):
        x = np.asarray(x, dtype=np.float32)
        S = _stft(x)
        mag, phase = np.abs(S), np.angle(S)
        Ps = self._patterns(mag.shape[1], key)
        signs = 2 * np.asarray(bits) - 1
        combo = sum(s * P for s, P in zip(signs, Ps))
        lo, hi, x_wm = 0.0, 4.0, x
        for _ in range(18):
            a = 0.5 * (lo + hi)
            x_wm = _istft(mag * self._base._gain_from_pattern(mag, combo, a)
                          * np.exp(1j * phase), x.size)
            if snr_db(x, x_wm) > self.snr_db:
                lo = a
            else:
                hi = a
        return x_wm

    def decode(self, y, key):
        Mb, _ = _mel_basis(self._base.n_mels)
        mel = Mb @ np.abs(_stft(y))
        logmel = np.log(mel + 1e-6)
        resid = (logmel - logmel.mean(axis=1, keepdims=True)
                 - logmel.mean(axis=0, keepdims=True) + logmel.mean())
        Ps = self._patterns(resid.shape[1], key)
        T = min(resid.shape[1], Ps[0].shape[1])
        return np.array([1 if np.sum(resid[:, :T] * P[:, :T]) > 0 else 0 for P in Ps])


class PayloadSurface:
    """B-bit surface payload: B high-band time-domain carriers, BPSK."""

    name = "surface"

    def __init__(self, nbits: int = 16, snr_db: float = 28.0):
        self.nbits, self.snr_db = nbits, snr_db
        self._base = SurfaceSpreadSpectrum(snr_db=snr_db)

    def _carriers(self, length, key):
        return [self._base._carrier(length, key * 131 + b + 1) for b in range(self.nbits)]

    def embed(self, x, key, bits):
        x = np.asarray(x, dtype=np.float32)
        cs = self._carriers(x.size, key)
        signs = 2 * np.asarray(bits) - 1
        delta = sum(s * c for s, c in zip(signs, cs))
        ps = float(np.sum(x ** 2))
        delta = delta * np.sqrt(ps / (10 ** (self.snr_db / 10)) / (np.sum(delta ** 2) + 1e-12))
        return x + delta

    def decode(self, y, key):
        y = np.asarray(y, dtype=np.float32)
        cs = self._carriers(y.size, key)
        return np.array([1 if np.dot(y[:c.size], c) > 0 else 0 for c in cs])


def run(utts, device, snrs, nbits=16):
    gl = MelGriffinLim(device=device)
    attackers = [("clean", None), ("mel_gl", gl)]
    try:
        from attackers import NeuralVocoderAttacker
        attackers.append(("vocos", NeuralVocoderAttacker(device=device)))
    except Exception as exc:
        print(f"[E2] Vocos unavailable: {exc}")
    for bw in (6.0, 3.0):
        try:
            attackers.append((f"encodec{bw:g}k", EncodecAttacker(bandwidth=bw, device=device)))
        except Exception as exc:
            print(f"[E2] EnCodec {bw}k unavailable: {exc}")
    rng = np.random.default_rng(7)
    results = {"nbits": nbits, "curves": {}, "table": []}

    for snr in snrs:
        for maker in (PayloadMagnitude, PayloadSurface):
            wm = maker(nbits=nbits, snr_db=snr)
            accs = {a[0]: [] for a in attackers}
            for i, x in enumerate(utts):
                key = 3000 + i
                bits = rng.integers(0, 2, size=nbits)
                xw = wm.embed(x, key, bits)
                for aname, att in attackers:
                    y = xw if att is None else att.apply(xw)
                    accs[aname].append(bit_accuracy(wm.decode(y, key), bits))
            row = {"watermark": wm.name, "snr_db": snr}
            for aname in accs:
                row[f"bitacc_{aname}"] = float(np.mean(accs[aname]))
            results["table"].append(row)
            results["curves"].setdefault(wm.name, []).append(
                {"snr_db": snr, "bitacc_gl": row["bitacc_mel_gl"],
                 "bitacc_clean": row["bitacc_clean"]})
            print(f"[E2] {wm.name:9s} SNR={snr:>4} "
                  + " ".join(f"{k.split('_')[1]}={row[k]:.3f}" for k in row if k.startswith('bitacc')))
    return results


def make_figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                                "savefig.bbox": "tight", "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for wm, style in (("magnitude", ("C0", "-o")), ("surface", ("C3", "-s"))):
        c = sorted(results["curves"][wm], key=lambda r: r["snr_db"])
        snr = [r["snr_db"] for r in c]
        ax.plot(snr, [r["bitacc_gl"] for r in c], style[1], color=style[0],
                ms=4, label=f"{wm} (after GL)")
    ax.axhline(0.5, color="0.4", lw=0.8, ls=":")
    ax.set_xlabel("imperceptibility budget (SNR, dB) — lower = more budget")
    ax.set_ylabel("payload bit-accuracy after laundering")
    ax.invert_xaxis()
    ax.set_ylim(0.45, 1.03)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_e2_rate_survival.pdf")
    plt.close(fig)


def load_utterances(n, root):
    import torchaudio
    ds = torchaudio.datasets.LIBRISPEECH(root=root, url="test-clean", download=False)
    out, target = [], int(CLIP_SEC * SR)
    for i in range(len(ds)):
        wav, sr, *_ = ds[i]
        wav = wav.mean(0).numpy().astype(np.float32)
        if wav.size < target:
            continue
        seg = wav[:target] / (np.max(np.abs(wav[:target])) + 1e-8) * 0.95
        out.append(seg.astype(np.float32))
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-root", default="/root/autodl-tmp/data")
    ap.add_argument("--n-utt", type=int, default=60)
    ap.add_argument("--nbits", type=int, default=16)
    ap.add_argument("--quick", action="store_true")
    args, _ = ap.parse_known_args()

    utts = load_utterances(args.n_utt, args.data_root)
    print(f"[E2] loaded {len(utts)} utterances")
    snrs = [34, 28, 22] if args.quick else [37, 34, 31, 28, 25, 22]
    results = run(utts, args.device, snrs, args.nbits)
    (RESDIR / "e2_audio.json").write_text(json.dumps(results, indent=2))
    make_figure(results)
    print("[E2] wrote results/e2_audio.json and figure")
    return results


if __name__ == "__main__":
    main()
