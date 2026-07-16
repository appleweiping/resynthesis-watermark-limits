# Experiments

## E0 — linear-Gaussian surrogate (CPU, always reproducible)

```bash
uv run python experiments/e0_surrogate.py     # figures + results/e0_surrogate.json
uv run pytest                                  # sanity-checks the code vs closed forms
```
Exercises the surrogate of `src/rwl/`: the exact surviving shift/exponent (Thm 1) and the
`R_LB` achievable lower bound. This is a code sanity check against closed forms, **not** a
validation of the model on real audio (that is E1–E3).

## E1 / E2 / E3 — real speech (GPU)

Need `pip install -e ".[audio]"` (see `requirements-audio.lock` for the exact frozen
environment + model revisions) and a GPU. One command builds manifests (incl. the
dev-other/test-other calibration extras) and runs everything:

```bash
python experiments/run_audio_all.py --root <LibriSpeech parent> --device cuda
python scripts/make_paper_macros.py    # inject numbers into the paper (also run by the above)
```

Individual drivers:
```bash
python -m experiments.audio.e1_survival   --manifest data/manifest_seed0.json --strict \
       --extra-calib data/manifest_calib_dev-other.json,data/manifest_calib_test-other.json
python -m experiments.audio.e2_predictor  --manifest data/manifest_seed0.json --strict
python -m experiments.audio.e3_payload_poc --manifest data/manifest_seed0.json --strict
```

**Attackers** (`experiments/audio/attackers.py`) instantiate `W = S∘A`; each exposes its
own analysis representation `A_W` for the channel-relative sensitivity `s_W`:
`stft_gl` (near-lossless control), `mel80_gl` (spectral inversion of THE unified 80-mel
analysis), `vocos` and `vocos_encodec6k` (neural vocoders), `encodec6k/3k`, `dac`, `snac`
(neural codecs), and `knnvc_self4/8` (self voice-conversion). Missing attackers **fail
loudly** under `--strict`.

**Deployed baselines** (`experiments/audio/baselines.py`): AudioSeal, WavMark,
SilentCipher, with a uniform host-blind scoring interface. They run at native full
strength (all transparent but **not** PESQ-equalized — the strength/energy confound is
disclosed in the paper).

**Constructed geometry marks** (`experiments/audio/marks.py`): a `nullspace` (kernel of
the magnitude analysis, alternating projections, measured leakage), a `rowspace`
(in-phase mel pattern), `inphase_rand`, and `mixture` sweeps — all PESQ-matched to 4.2.
Their genie-aided matched-filter probe is a labeled **diagnostic** of channel geometry,
not a deployable detector; deployed-detector conclusions come only from the baselines.

## What each experiment claims

- **E1**: deployed-watermark survival across channels — oriented AUC + calibrated
  operating point (TPR + achieved FPR from a 7,157-negative independent calibration).
- **E2**: does a geometric quantity (channel-relative `s_W`, or the static magnitude
  fraction) predict post-attack detectability out-of-sample? Held-out fit(dev-clean)→
  test(test-clean), within-attacker rank-pooled. Every constructed mark is at a common
  PESQ-4.2 budget, so achieved PESQ/SI-SDR/SNR enter as competitor predictors alongside
  spectral centroid. Inference is cluster-aware: a common whole-direction permutation
  across attackers, plus direction / speaker / two-way (direction×utterance) bootstrap CIs.
- **E3**: multi-bit proof of concept (blind decoder, BER/BLER + CIs) — **not** a rate
  measurement.

**Model note.** SilentCipher uses its supported 44.1 kHz checkpoint (the released 16 kHz
model is unreachable through the package's byte API); DAC weights install with `--no-deps`;
behind a firewall use `HF_ENDPOINT=https://hf-mirror.com` for HF models and pre-fetch the
GitHub-release checkpoints (DAC, kNN-VC/WavLM) into the torch hub cache.
