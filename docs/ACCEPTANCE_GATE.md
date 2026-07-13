# Acceptance gate (from the external review) — item-by-item evidence

Status legend: ✅ done · 🔶 partial/disclosed · ❌ not done. This file is updated as the
final runs land; nothing is marked done without pointing at code/results.

**Adversarial-polish convergence.** After the P0/P1 rework, the paper + code were run
through repeated multi-agent adversarial reviews (4 lenses + adversarial verification
each). Confirmed-major count per round: **13 → 2 → 1 → 1 → 1 → 2 → 2 → 0 → 0**. Two
consecutive clean rounds (rounds 8–9) establish convergence; every finding produced a
genuine honesty improvement (e.g. dropping a non-robust waveform sign-reversal, relabeling
a degenerate constant-score detector, switching separability claims to raw two-sided CIs,
scoping the predictor to within-attacker ordinal ranking). The paper spine (Prop 1 /
Thm 1 / R_LB / E1 taxonomy / E2 mel-domain predictor / E3) was unaffected throughout.

| # | Gate item | Status | Evidence |
|---|---|---|---|
| 1 | Raw AUC<0.5 no longer read as erasure | ✅ | `metrics_audio.py` (raw+oriented, `sign_inverted` flag); paper §Experiments wording "operational failure … separability remaining"; abstract |
| 2 | Matched rate converse/achievability deleted or proved | ✅ | Deleted. `R_LB` = blind-embedder achievable lower bound only; no converse claimed (`src/rwl/capacity.py`, paper §3, THEORY.md §4) |
| 3 | True attack-specific nullspace mark, constructed & verified | ✅ | `marks.nullspace_direction` (alternating projections, quadrature∩consistent); measured leakage 0.06× random (first-order) / 0.16× (operating amplitude) reported in results + paper |
| 4 | Surface/nullspace mark at reasonable perceptual quality | ✅ | All constructed marks PESQ-matched (4.2 target) via `scale_to_pesq`; old audible 5 kHz carrier deleted |
| 5 | Rate experiment → reliable bps/BLER or fully downgraded | ✅ | Downgraded: E3 = 8-bit blind PoC, BER/BLER + binomial CIs, explicit "no achievable-rate measurement" (e3_payload_poc.py, paper §E3) |
| 6 | 1% FPR from independent calibration set (≥5000) | ✅ | 7,157 clean negatives: dev-clean 1,928 + dev-other 2,568 + test-other 2,661 (all speaker-disjoint from test-clean); thresholds only from calibration (`threshold_at_fpr` raises below 1,000); **all three baselines uniformly thresholded on the 7,157 pool**; Clopper–Pearson CIs |
| 7 | ≥1000 clips, speaker-disjoint splits, multi-key/seed | ✅ | test = 1,000 clips / 40 speakers / 20F+20M (manifest stats); calib/fit vs test speaker-disjoint by partition; keys: 2 (audioseal, wavmark), 1 (silentcipher; disclosed); E2 replicated on 3 independent manifest seeds (ρ 0.72/0.66/0.60, all p<10⁻³) |
| 8 | Voice conversion + multiple codecs included | ✅ | kNN-VC self-VC top-k∈{4,8} (WavLM+HiFi-GAN, level-matched); EnCodec 6/3k + DAC + SNAC; Vocos + Vocos-EnCodec |
| 9 | Latent-Mark / Feature-Aligned related work + strong baselines | ✅ | Cited & differentiated (refs verified via arXiv; RELATED_WORK.md); 3 deployed baselines AudioSeal + WavMark + SilentCipher all run at N=1000 (Timbre not included — noted as limitation) |
| 10 | Full repro environment; fail-loudly on missing attackers | ✅ | requirements-audio.lock (exact versions + model sources); strict registries raise; per-clip score arrays saved; one-command runner |
| 11 | Surrogate theorems not extrapolated to all speech generators | ✅ | Prop 1 (general) vs Thm 1 (surrogate) separation in paper/THEORY.md/docstrings; "closed forms do NOT transfer" stated |
| 12 | Metric consistency (one primary metric; both reported) | ✅ | Oriented AUC = separability (theory-linked); TPR@1%FPR = operational; recalibration diagnostic labeled; all rows carry all metrics |
| 13 | Perceptual budget reported (not matched SNR) | 🔶 | Not matched-PESQ: all 3 baselines saturate at native full strength above the 4.2 target (PESQ 4.42/4.28/4.60, SI-SDR 26/37/34 dB) — reported per-baseline; the strength/energy confound (AudioSeal loudest) is disclosed in Scope & limitations rather than force-equalized (which would distort each scheme's operating point) |
| 14 | Predictor validated out-of-sample w/ competitors + permutation | ✅ | E2 (5 attackers: mel-GL/Vocos/EnCodec/DAC/SNAC): 108 dirs, fit dev-clean → test test-clean; within-attacker ρ .48–.80, pooled .72 CI [.60,.80], perm p=.001; SNR/centroid controls fail; necessity max dev .045 |
| 15 | Unified analysis map (no librosa/torchaudio mixing) | ✅ | `analysis.MelAnalysis` single implementation; per-attacker sensitivity uses each attacker's own representation |
| 16 | Content/speaker/prosody invariants claim | ✅ | Empirically grounded via kNN-VC self-resynthesis; elsewhere called spectro-temporal/codec-latent invariants (THEORY.md §1) |

## Known residual limitations (disclosed, not hidden)
- SilentCipher uses its supported 44.1 kHz pipeline on 16 kHz speech (the released
  16 kHz model is unreachable through the package's byte API — a documented upstream
  bug); resampling is part of its embed/decode path.
- kNN-VC attack definition was fixed mid-campaign (VAD disabled); SilentCipher's two
  kNN-VC rows were re-run with the final definition and spliced in (consistent across
  all baselines). ✅
- E3's blind decoder fails under SNAC (BER ≈ 0.47) — reported as a failure case.
- Timbre-style watermark baseline not included.
- Adaptive/key-aware attacks out of scope (stated).
