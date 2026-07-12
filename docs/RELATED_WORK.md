# Related work and exact differentiation

What this project claims — and, just as importantly, what it does **not** claim —
relative to each strand of prior work.

## 1. Empirical attack studies (the collapse is known)

- **O'Reilly et al., "Deep Audio Watermarks are Shallow"** (ICLR-W 2025): post-hoc
  speech watermarks fail under reconstruction attacks.
- **Wen et al., SoK** (arXiv:2503.19176): systematic benchmark; neural codecs are the
  dominant threat.
- **Ding et al., "Learning to Evade"** (arXiv:2606.22310): *adaptive* attacks — out of
  scope for our model, and we say so explicitly.

These document *that* watermarks fail. We provide a stated channel model in which the
failure (and its exceptions) can be derived at the right level of generality, plus a
measured predictor of *how much* separability survives.

## 2. Invariant/latent-aligned watermarks (the idea is established — NOT ours)

- **Latent-Mark** (arXiv:2603.05310): optimizes waveforms to shift codec latents;
  cross-codec transfer; zero-bit.
- **Feature-Aligned Speech Watermarking** (arXiv:2606.11828): watermark aligned to the
  speech feature distribution for reconstruction robustness.
- **VoiceMark** (Li et al., Interspeech 2025): speaker-latent marks robust to zero-shot VC.
- **Images**: Zhao et al. (NeurIPS 2024) prove pixel-space removability and prescribe
  semantic embedding; Tree-Ring and Gaussian Shading realize it.

**Differentiation.** We do not claim invariant alignment as a contribution. What these
method papers do not provide, and we do:
1. a **general proposition** (Chernoff DPI + exact-erasure sufficient condition) that
   scopes exactly what is guaranteed for *any* blind analysis–resynthesis channel;
2. **closed-form** surviving shift/exponent in a stated surrogate (with its limits
   stated);
3. a **channel-relative, measurable survival predictor** validated on held-out,
   speaker-disjoint data with competitor predictors and permutation controls;
4. a **calibrated operational evaluation protocol** (independent calibration split,
   raw-vs-oriented separation) that corrects measurement practices which conflate
   score inversion with erasure.

## 3. Classical watermarking theory (why R_LB is NOT capacity)

- **Costa (1983); Moulin–O'Sullivan (2003); Chen–Wornell QIM (2001)**: information-
  hiding capacity with *informed* embedding against fixed/additive attack channels.

Our `R_LB` is an achievable rate for **blind (host-uninformed) Gaussian embedders**
against the resynthesis channel — a *lower bound only*. Informed embedding can exceed
it; we prove **no rate converse**; we make **no capacity claim**. The classical results
do not cover the resynthesis attack geometry; we do not cover their informed-embedder
optimality. The two are complementary, and the paper says exactly this.

## 4. What would falsify our empirical claim

The predictor study (E2) is designed so the answer could have been "no":
- if post-attack oriented separability did not track the attacker's own analysis
  sensitivity out-of-sample (Spearman ≈ 0, permutation p large), or
- if generic budget measures (SNR) predicted survival equally well,
the subspace picture would have no empirical content for real channels. Those are the
comparisons we run and report, with CIs.
