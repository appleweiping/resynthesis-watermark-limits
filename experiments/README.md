# Experiments

## E0 — surrogate (CPU, always reproducible)

```bash
uv run python experiments/e0_surrogate.py     # or: python experiments/run_all.py
```
Validates both theorems on the linear-Gaussian surrogate and writes
`paper/figures/fig_{converse_collapse,rate_survival,theory_vs_sim}.pdf` and
`results/e0_surrogate.json`. This is what the `pytest` suite gates.

## E1 / E2 — real speech (GPU)

These instantiate the resynthesis channel with pretrained neural models and need
`pip install -e ".[audio]"` plus a GPU. Data is LibriSpeech `test-clean` (auto-downloaded by
`torchaudio`, or place it at `--data-root`).

```bash
python experiments/audio/e1_converse_audio.py --device cuda --data-root <dir> --n-utt 80
python experiments/audio/e2_achievability_audio.py --device cuda --data-root <dir> --n-utt 60
python scripts/make_paper_macros.py            # inject numbers into the paper
```

**Attackers** (`experiments/audio/attackers.py`) instantiate `W = S∘A` with growing nullspace:
`StftGriffinLim` (near-lossless control, `ker A` = phase), `MelGriffinLim` (lossy vocoder,
`ker A` = phase + sub-mel detail), `EncodecAttacker` (neural codec at 6/3/1.5 kbps).

**Watermarks** (`experiments/audio/watermarks_audio.py`): `SurfaceSpreadSpectrum` (nullspace),
`MagnitudeSpreadSpectrum` (invariant mel envelope — the constructive achievability scheme),
`AudioSealWatermark` (Meta's deployed neural watermark), and `MixedSpreadSpectrum` (the
surface→invariant sweep for the converse curve).

**AudioSeal offline note.** AudioSeal's loader hardcodes `huggingface.co`. Behind a firewall,
pre-place the weights (from a mirror) at `~/.cache/audioseal/<sha1(url_path)[:24]>` — see
`scripts/prefetch_audioseal.py`.
