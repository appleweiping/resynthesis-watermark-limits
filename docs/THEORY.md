# A Subspace Model of Speech-Watermark Survival Under Analysis–Resynthesis — theory notes

This note states precisely what is proved, at which level of generality. There are three
levels, and they must not be conflated:

1. a **general proposition** (any analysis–resynthesis channel): DPI + an exact-erasure
   sufficient condition;
2. a **linear-Gaussian surrogate theorem**: closed-form surviving mean shift and exponent;
3. a **Gaussian achievable lower bound** `R_LB` for a restricted embedder class — *not* a
   watermarking capacity.

The numerical implementation in `src/rwl/` is *sanity-checked against these closed-form
quantities* by the test suite (`tests/`). That check validates the code, not the model:
whether real attackers behave like the surrogate is an **empirical question** answered
(only) by the experiments.

## 1. Setup

A provenance **watermark** embeds a key-dependent perturbation into a host signal,
`x ↦ E_κ(x)` (additive case `E_κ(x) = x + δ_κ`), with `δ_κ` determined by a secret key
`κ`. A **detector** that knows `κ` decides between

- `H₀` (no watermark): observation drawn from the clean-speech law `P₀`,
- `H₁` (watermarked): observation drawn from `P₁ = (E_κ)_# P₀`.

The optimal Bayesian error probability after `N` i.i.d. observations decays as
`exp(-N·C(P₀,P₁))`, where `C` is the **Chernoff information**
`C(P₀,P₁) = max_{s∈[0,1]} −log ∫ p₀^{1−s} p₁^{s}`. `C=0` iff `P₀=P₁`.

**Attacker (blind resynthesis).** The attacker applies a channel `W = S∘A`:
- `A: 𝓧 → 𝓩` (analysis) maps a waveform to a representation `z = A(x)` — for the real
  systems we test, a mel magnitude envelope or a codec latent (we call these
  **spectro-temporal / codec-latent invariants**; whether they coincide with perceptual
  invariants like content/speaker/prosody is *not* assumed by the theory);
- `S: 𝓩 → 𝓧` (synthesis) generates a waveform from `z`, with randomness independent of
  the input.

The attacker does **not** know `κ` and does **not** optimize against the scheme
(key-aware/adaptive removal is out of scope).

**Imperceptibility.** `δ` must be inaudible: `ρ(δ; x) ≤ D` for a perceptual distortion
`ρ`. In the surrogate, `ρ(δ) = δᵀMδ` with `M ≻ 0`.

## 2. Proposition 1 (general analysis–resynthesis channels)

Let `W = S∘A` be *any* measurable channel of the above form (in particular, any neural
vocoder, codec, or voice-conversion pipeline used blindly).

**(a) Data-processing inequality.** For any channel `W` (Markov kernel),

```
C(W_# P₀, W_# P₁) ≤ C(P₀, P₁).
```

Laundering can only lose detectability. This — and only this — is what the theory
asserts about general nonlinear resynthesizers. It gives **no rate**, **no constant**,
and **no prediction of how much survives**.

**(b) Exact erasure (sufficient condition).** If the embedder preserves the analysis
output exactly,

```
A(E_κ(x)) = A(x)   P₀-almost surely,
```

and `S` depends on its input only through `z` (plus independent randomness), then
`W_# P₁ = W_# P₀`, hence `C(W_# P₀, W_# P₁) = 0`: no detector beats chance at any
sample size.

*Proof.* (a) is the DPI for f-divergences applied to the Chernoff family (each
`−log∫p₀^{1−s}p₁^{s}` is monotone under Markov kernels; the max over `s` preserves
monotonicity). (b): `W(E_κ(x)) = S(A(E_κ(x)), U) = S(A(x), U) = W(x)` a.s. with `U ⟂ x`,
so the two output laws coincide. ∎

Note what (b) does **not** say: it does not say a watermark with small-but-nonzero
`‖A(E_κ(x)) − A(x)‖` is *nearly* erased at any particular rate. Quantitative statements
require a model — that is the role of the surrogate.

## 3. Theorem 1 (linear-Gaussian surrogate: exact surviving shift and exponent)

**Model.** Whitened coordinates, `P₀ = 𝒩(0, Iₙ)`; `A ∈ ℝ^{k×n}` full row rank;
`P = A⁺A` the orthogonal projector onto `row(A)`; synthesis resamples the complement:

```
W(x) = P x + (I−P) w,   w ~ 𝒩(0, Iₙ)  fresh.
```

**Statement.** For a fixed additive shift `δ`:

```
W_# 𝒩(δ, I) = 𝒩(Pδ, I),   and   C(W_# P₀, W_# P₁) = ⅛‖Pδ‖².
```

Consequences:

- the channel transmits only the **equivalence class** `[δ] ∈ ℝⁿ/ker(A)`; the effective
  surviving perturbation is `Pδ`;
- `δ ∈ ker(A)` ⟹ exact erasure (`C = 0`), recovering Prop. 1(b);
- **mixed** perturbations (row + null components) survive **partially**, with exponent
  `⅛‖Pδ‖² > 0` whenever `Pδ ≠ 0`;
- row-space-only perturbations spend no budget on erased components — efficient, but
  *not* the unique surviving set. (We do **not** say "the survivable set is exactly the
  non-nullspace".)

**Budgeted optimum.** `max_{δᵀMδ≤D} ⅛‖Pδ‖² = (D/8)·λ_max(P, M)`, attained at the top
generalized eigenvector of `(P, M)`.

**Scope.** These constants are properties of the surrogate. For real nonlinear
attackers we *measure* an analogous channel-relative quantity (below) and test
empirically whether it predicts survival; nothing is claimed a priori.

## 4. `R_LB` — a Gaussian achievable lower bound (NOT capacity)

**Restricted class.** Uninformed (host-blind) encoder; Gaussian codebook keyed by `κ`;
decoder knows `κ` but not the host; per-block average distortion `tr(M_row Q) ≤ D` over
i.i.d. blocks, each block being the `k`-dimensional invariant vector; reliability =
vanishing block error probability as blocklength → ∞. Units: **nats per use of the
`k`-dimensional invariant vector channel** — a *total* over the `k` surviving coordinates,
not a per-coordinate rate (do not divide by `k`).

**Statement (achievability only).** Rates below

```
R_LB = max_{Q ⪰ 0, tr(M_row Q) ≤ D} ½ log det(I + Q)
```

are achievable within this class (AWGN water-filling over the eigenmodes of
`M_row = V_rowᵀ M V_row`; the clean host acts as unit-variance interference).

**What `R_LB` is not.**
- Not the watermarking capacity: an **informed** (host-aware) embedder in the
  Costa / Moulin–O'Sullivan sense can exceed it via dirty-paper coding. We do not
  analyze that setting.
- No **rate converse** is proved, at any level of generality. We therefore never write
  "the surviving payload is exactly R*".

A nullspace watermark has `R_LB = 0` after `W`; an invariant-aligned code attains
`R_LB > 0` whenever `D > 0` and the invariant subspace is nonempty.

## 5. Channel-relative survival predictor (what the experiments actually test)

For a real attacker `W` (mel inversion, neural vocoder, codec), define the measured,
channel-relative preserved fraction of a perturbation `δ` at host `x`:

```
f_W(δ; x)  from attack-specific sensitivity  —  e.g. finite differences
‖W(x+δ) − W(x)‖ / ‖δ‖ (deterministic W, matched seeds for stochastic W), or a
representation distance ‖A_W(x+δ) − A_W(x)‖ using that attacker's own analysis A_W.
```

The **empirical hypothesis** (not a theorem) is that oriented post-attack separability
increases with `f_W` for a fixed detector family. This is tested on held-out data with
alternative predictors and permutation controls; see the experiment protocol. A single
fixed mel fraction is *not* reused across attackers with different `A_W`.

## 6. Assumptions, honesty, and scope

- **Blind, non-adaptive attacker.** Key-aware or adaptive removal (an attacker
  optimizing against the specific scheme) is explicitly out of scope.
- **Detector orientation.** Theory statements concern *separability* (Chernoff
  exponent / oriented discriminability). A deployed detector with a fixed sign and
  threshold can fail *operationally* (score sign/order inversion) while substantial
  attack-conditioned separability remains; we report both and never conflate them.
- **Gaussianity/whitening** is a modeling device for closed forms; nothing Gaussian is
  assumed about real speech in the empirical claims.
- **The surrogate is not the world.** Real analysis maps are nonlinear and only
  approximately idempotent; codec latents are quantized. The experiments measure the
  degree to which the subspace picture predicts real survival — that measured
  correspondence, not the theorems, is the paper's empirical contribution.
