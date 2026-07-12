"""Fast isolated tuning of the invariant watermark's detectability (no attackers)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from metrics_audio import empirical_auc, mel_invariant_fraction, snr_db
from watermarks_audio import MagnitudeSpreadSpectrum


def load(n, root):
    import torchaudio
    ds = torchaudio.datasets.LIBRISPEECH(root=root, url="test-clean", download=False)
    out, tgt = [], 48000
    for i in range(len(ds)):
        w, sr, *_ = ds[i]
        w = w.mean(0).numpy().astype(np.float32)
        if w.size < tgt:
            continue
        out.append((w[:tgt] / (np.max(np.abs(w[:tgt])) + 1e-8) * 0.95).astype(np.float32))
        if len(out) >= n:
            break
    return out


def main():
    utts = load(30, "/root/autodl-tmp/data")
    for snr in [28, 24, 20, 17]:
        wm = MagnitudeSpreadSpectrum(snr_db=snr)
        keys = list(range(len(utts)))
        xwm = [wm.embed(x, k) for x, k in zip(utts, keys)]
        sc = [wm.score(x, k) for x, k in zip(utts, keys)]
        sw = [wm.score(w, k) for w, k in zip(xwm, keys)]
        auc = empirical_auc(sw, sc)
        f = np.mean([mel_invariant_fraction(x, w) for x, w in zip(utts, xwm)])
        realsnr = np.mean([snr_db(x, w) for x, w in zip(utts, xwm)])
        print(f"snr={snr} realSNR={realsnr:.1f} AUC_before={auc:.3f} f={f:.3f}")


if __name__ == "__main__":
    main()
