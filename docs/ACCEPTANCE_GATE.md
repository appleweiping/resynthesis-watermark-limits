# Acceptance gate (from the external review) — item-by-item evidence

Status legend: ✅ done · 🔶 in progress · ❌ not done. This file is updated as the
final runs land; nothing is marked done without pointing at code/results.

| # | Gate item | Status | Evidence |
|---|---|---|---|
| 1 | Raw AUC<0.5 no longer read as erasure | ✅ | `metrics_audio.py` (raw+oriented, `sign_inverted` flag); paper §Experiments wording "operational failure … separability remaining"; abstract |
| 2 | Matched rate converse/achievability deleted or proved | ✅ | Deleted. `R_LB` = blind-embedder achievable lower bound only; no converse claimed (`src/rwl/capacity.py`, paper §3, THEORY.md §4) |
| 3 | True attack-specific nullspace mark, constructed & verified | ✅ | `marks.nullspace_direction` (alternating projections, quadrature∩consistent); measured leakage 0.06× random (first-order) / 0.16× (operating amplitude) reported in results + paper |
| 4 | Surface/nullspace mark at reasonable perceptual quality | ✅ | All constructed marks PESQ-matched (4.2 target) via `scale_to_pesq`; old audible 5 kHz carrier deleted |
| 5 | Rate experiment → reliable bps/BLER or fully downgraded | ✅ | Downgraded: E3 = 8-bit blind PoC, BER/BLER + binomial CIs, explicit "no achievable-rate measurement" (e3_payload_poc.py, paper §E3) |
| 6 | 1% FPR from independent calibration set (≥5000) | ✅ | 7,157 clean negatives: dev-clean 1,928 + dev-other 2,568 + test-other 2,661 (all speaker-disjoint from test-clean); thresholds only from calibration (`threshold_at_fpr` raises below 1,000; rethreshold_e1.py); Clopper–Pearson CIs |
| 7 | ≥1000 clips, speaker-disjoint splits, multi-key/seed | ✅/🔶 | test = 1,000 clips / 40 speakers / 20F+20M (manifest stats); calib/fit vs test speaker-disjoint by partition; keys: 2 (audioseal, wavmark), 1 (silentcipher; disclosed); seeds: manifest ×3, E2 seed-robustness runs 🔶 |
| 8 | Voice conversion + multiple codecs included | ✅ | kNN-VC self-VC top-k∈{4,8} (WavLM+HiFi-GAN, level-matched); EnCodec 6/3/1.5k + DAC + SNAC; Vocos + Vocos-EnCodec |
| 9 | Latent-Mark / Feature-Aligned related work + strong baselines | ✅/🔶 | Cited & differentiated (refs verified via arXiv; RELATED_WORK.md); baselines AudioSeal + WavMark + SilentCipher (Timbre not included — noted as limitation) 🔶 pending final E1 rows |
| 10 | Full repro environment; fail-loudly on missing attackers | ✅ | requirements-audio.lock (exact versions + model sources); strict registries raise; per-clip score arrays saved; one-command runner |
| 11 | Surrogate theorems not extrapolated to all speech generators | ✅ | Prop 1 (general) vs Thm 1 (surrogate) separation in paper/THEORY.md/docstrings; "closed forms do NOT transfer" stated |
| 12 | Metric consistency (one primary metric; both reported) | ✅ | Oriented AUC = separability (theory-linked); TPR@1%FPR = operational; recalibration diagnostic labeled; all rows carry all metrics |
| 13 | Matched perceptual budget (not matched SNR) | ✅ | All baselines strength-calibrated to median PESQ 4.2 (`calibrate_strength_to_pesq`); PESQ+SI-SDR reported; attack-on-clean PESQ reported |
| 14 | Predictor validated out-of-sample w/ competitors + permutation | ✅ | E2: 108 dirs, fit dev-clean → test test-clean; within-attacker ρ .48–.80, pooled .72 CI [.60,.80], perm p=.001; SNR/centroid controls fail; necessity max dev .045 |
| 15 | Unified analysis map (no librosa/torchaudio mixing) | ✅ | `analysis.MelAnalysis` single implementation; per-attacker sensitivity uses each attacker's own representation |
| 16 | Content/speaker/prosody invariants claim | ✅ | Empirically grounded via kNN-VC self-resynthesis; elsewhere called spectro-temporal/codec-latent invariants (THEORY.md §1) |

## Known residual limitations (disclosed, not hidden)
- SilentCipher uses its supported 44.1 kHz pipeline on 16 kHz speech (the released
  16 kHz model is unreachable through the package's byte API — a documented upstream
  bug); resampling is part of its embed/decode path.
- kNN-VC attack definition was fixed mid-campaign (VAD disabled); SilentCipher's two
  kNN-VC rows are being re-run with the final definition for consistency. 🔶
- E3's blind decoder fails under SNAC (BER ≈ 0.47) — reported as a failure case.
- Timbre-style watermark baseline not included.
- Adaptive/key-aware attacks out of scope (stated).
