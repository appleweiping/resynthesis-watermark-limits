# The Resynthesis Channel — full theory

This note gives the model, assumptions, and proofs behind the two theorems. The
linear-Gaussian surrogate in `src/rwl/` makes every quantity closed-form; the test suite
(`tests/`) cross-checks each claim against Monte-Carlo before it enters the paper.

## 1. Setup

A provenance **watermark** embeds a key-dependent perturbation into a host signal,
`x ↦ x + δ_κ`, with `δ_κ` determined by a secret key `κ`. A **detector** that knows `κ`
decides between

- `H₀` (no watermark): observation drawn from the clean-speech law `P₀`,
- `H₁` (watermarked): observation drawn from `P₁`, the law of `x + δ_κ`.

The optimal Bayesian error probability after `N` i.i.d. observations decays as
`exp(-N·C(P₀,P₁))`, where `C` is the **Chernoff information**
`C(P₀,P₁) = max_{s∈[0,1]} −log ∫ p₀^{1−s} p₁^{s}`. `C=0` iff `P₀=P₁`, i.e. the watermark
is undetectable by *any* detector at *any* sample size.

**Attacker (blind resynthesis).** The attacker applies a channel `W = S∘A`:
- `A: 𝓧 → 𝓩` (analysis) maps a waveform to its perceptual invariants `z = A(x)` — the
  quantities a vocoder/codec/voice-conversion model is built to preserve (linguistic
  content, speaker identity, coarse prosody). `A` is a (near-)sufficient statistic for
  perceptual identity: two signals with the same `z` sound the same.
- `S: 𝓩 → 𝓧` (synthesis) samples a waveform consistent with `z`, drawing everything `A`
  discarded from the clean-speech prior `P₀(·|z)`.

The attacker does **not** know `κ` and does **not** optimize against the scheme
(key-aware/adaptive attacks are discussed in §5 as future work).

**Imperceptibility.** `δ` must be inaudible: `ρ(δ; x) ≤ D` for a perceptual distortion
`ρ` (psychoacoustic masking). In the surrogate, `ρ(δ) = δᵀMδ` with `M ≻ 0`.

## 2. Linear-Gaussian surrogate

Work in whitened coordinates where `P₀ = 𝒩(0, Iₙ)` (all perceptual weighting is carried
by `M`). Let `A ∈ ℝ^{k×n}` have full row rank, `P = A⁺A` the orthogonal projector onto
`row(A)` (the **invariant subspace**), `I−P` the projector onto `ker(A)` (the **surface
subspace**). Synthesis resamples the surface coordinates from the clean prior:

```
W(x) = P x + (I−P) w,   w ~ 𝒩(0, Iₙ)  fresh.
```

This is exactly "project onto the manifold of clean signals sharing the invariants `z`."
Then `W_#𝒩(0,I) = 𝒩(0,I)` and `W_#𝒩(δ,I) = 𝒩(Pδ, I)`.

## 3. Theorem 1 — Converse (nullspace erasure)

**Statement.** If `δ ∈ ker(A)` then `C(W_#P₀, W_#P₁) = 0`: after laundering, no detector
beats chance, for any `N`. More generally, for any `δ`,
`C(W_#P₀, W_#P₁) = ⅛‖Pδ‖² ≤ ⅛‖δ‖² = C(P₀,P₁)` (data-processing inequality), with
equality iff `δ ∈ row(A)`.

**Proof.** `W_#P₁ = 𝒩(Pδ, I)` and `W_#P₀ = 𝒩(0, I)` share covariance `I`, so their
Chernoff information is `⅛(Pδ)ᵀI⁻¹(Pδ) = ⅛‖Pδ‖²` (optimum at `s=½`). If `δ∈ker(A)` then
`Pδ = 0` and the two output laws coincide, giving `C=0`. Since `P` is an orthogonal
projector, `‖Pδ‖ ≤ ‖δ‖`, hence `C(W_#P₀,W_#P₁) ≤ C(P₀,P₁)`; this is the DPI specialized
to the channel and is tight exactly on `row(A)`. ∎

**Companion (distributional).** `W(x+δ) = P(x+δ) + (I−P)w`. For `δ∈ker(A)` this equals
`Px + (I−P)w = W(x)` in distribution: the laundered watermarked signal is distributed
identically to a laundered clean signal — not merely close, but equal.

**Why the nullspace is huge for audio.** Human hearing is (largely) phase-insensitive and
subject to simultaneous/temporal masking, so a vast set of perturbations is inaudible
(`ρ(δ)` small). Vocoders/codecs discard exactly these (they re-derive phase, quantize
masked bins). Thus the perceptually cheapest embedding directions are precisely those in
`ker(A)`. Post-hoc watermarks, tuned for imperceptibility, drift there — and die. This is
the theory's explanation of the universal empirical collapse.

## 4. Theorem 2 — Achievability (invariant sub-channel)

**Detection (zero-rate).** Maximizing the surviving exponent under the budget,

```
max_{δᵀMδ ≤ D} ⅛‖Pδ‖² = (D/8)·λ_max(P, M),
```

attained at the top generalized eigenvector of `(P, M)`. This is strictly positive
whenever `row(A) ⊄ ker(M‑cheap directions)` — i.e. whenever *some* invariant direction is
affordable. Restricting `δ∈ker(A)` gives quotient `0` (post-hoc dies); the optimizer
necessarily carries invariant energy that survives.

**Payload (rate).** Encode a message `m` as a surviving shift `u = Pδ(m) ∈ row(A)`. A
blind detector sees `y = u + η`, `η ~ 𝒩(0, I)` the clean host in orthonormal invariant
coordinates. With input covariance `Q ⪰ 0` and cost `tr(M_row Q) ≤ D`
(`M_row = V_rowᵀ M V_row`),

```
R* = max_{tr(M_row Q) ≤ D} ½ log det(I + Q),
```

a water-filling over the eigenmodes of `M_row`: `q_i = max(0, ν/λ_i − 1)` with `ν` set so
the budget binds. `R* > 0` whenever `D > 0` and the invariant subspace is nonempty; a
nullspace watermark has `R* = 0` after `W`.

**Meeting.** Theorem 1 says the survivable set is *at most* `row(A)`; Theorem 2 achieves a
positive rate *on* `row(A)`. The survivable set is therefore **exactly** the non-nullspace
of `A`, and its size is the invariant sub-channel capacity `R*`.

## 5. Assumptions, honesty, and scope

- **Blind, non-adaptive attacker.** The converse is against an attacker who runs a generic
  resynthesizer without the key. A key-aware attacker who *optimizes* to remove our
  specific watermark is out of scope (it becomes an empirical arms race); we treat it as a
  remark and a stress test (`E4`), not a core claim.
- **`A` as sufficient statistic.** Real analysis maps are only *approximately* sufficient;
  `S` is stochastic with a temperature. `E4` measures how the converse degrades as these
  idealizations break (imperfect `A`, low-temperature `S`).
- **Whitening / Gaussianity.** The surrogate is a modeling device for *validating* the
  proofs. Monotonicity (DPI) holds for the general `W = S∘A` with no Gaussian assumption;
  the *exact-zero* for `δ∈ker(A)` uses that `S` resamples the nullspace from the clean
  conditional. The Gaussian model supplies the exact constants and the achievability
  construction.
- **`R*` is a blind-embedder rate**, not the watermarking capacity: it treats the clean host
  as interference (no informed/dirty-paper coding), so informed embedding could exceed it.

## 6. What the experiments test

- **E1** instantiates `A`, `S` with STFT/mel-spectrogram inversion (Griffin–Lim) and the
  EnCodec neural codec, and shows watermarks collapse when their energy lies in the channel's
  nullspace — to the `‖Pδ‖≈0` floor predicted by Thm 1. (A learned neural vocoder / voice
  conversion is future work.)
- **E2** builds an invariant-aligned watermark realizing a surviving payload (Thm 2) and
  traces the rate–survival–SNR relation where the converse and achievable rate meet.
- **E4** stress-tests the idealizations (§5).
