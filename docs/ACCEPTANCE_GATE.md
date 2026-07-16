# Acceptance gate (from the external review) — item-by-item evidence

Status legend: ✅ done · 🔶 partial/disclosed · ❌ not done. This file is updated as the
final runs land; nothing is marked done without pointing at code/results.

> **Pre-submission fix batch (in progress).** A second review required, before the repo
> can be called submission-ready: (P0-1) E2 on a common PESQ-4.2 budget with achieved
> PESQ/SI-SDR/SNR as competitor predictors — **code done** (`e2_predictor.py`,
> `scale_to_pesq`); (P0-2) cluster-aware E2 inference — common whole-direction
> permutation + two-way bootstrap — **code done** (`e2_stats.py`); (P0-3) stable SHA-256
> keys, fixed seeds, `rand_init=False` Griffin–Lim, pinned `run_audio_all` params,
> byte-level macro check, GitHub Actions CI — **code done** (`repro.py`,
> `check_macros_reproducible.py`, `.github/workflows/ci.yml`); (P0-4) model pinning by
> HF revision + checkpoint hash — **machinery done** (`model_lock.py`,
> `capture_model_lock.py`), SHA values captured on the run host; (P0-5) R_LB units
> relabelled to *per invariant vector-channel use*; (P0-6) kNN-VC relabelled a
> low-fidelity self-VC stress test with per-attacker clean PESQ reported.
> **Pending:** the GPU E2 re-run (seeds 0/1/2) that regenerates the headline
> ρ/CI/p under the corrected protocol, the `models.lock.json` capture, and the E1
> speaker-cluster CI recompute. Until those land, the E2 numbers are marked provisional
> in the README/paper and the repo is **not** marked submission-ready.

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
