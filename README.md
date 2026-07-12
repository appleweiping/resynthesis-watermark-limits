# resynthesis-watermark-limits

**The Resynthesis Channel — Fundamental Limits of Speech Watermarking Against Generative Laundering.**

A complete, reproducible signal-processing-theory research project (targeting the IEEE ICASSP
*Signal Processing Theory & Methods* / *Information Forensics & Security* tracks). The **theory and
every theory-validation experiment are CPU-only and fully reproducible**; the empirical
instantiation on real speech uses pretrained neural generators and one small trained watermarker on
a **GPU**. Every theorem in the paper is cross-checked against Monte-Carlo simulation by the test
suite **before** it is allowed into the manuscript.

> **TL;DR** — Every audio watermark on the market (AudioSeal, WavMark, …) is silently erased when an
> attacker simply runs the audio through a neural vocoder, codec, or voice-conversion model. The
> whole field has documented this collapse **empirically**, over and over — but nobody has explained
> *why*, or *what could possibly survive*. We give the missing theory. Model the attacker as a
> **resynthesis channel** `W = S∘A`: an *analysis* map `A` that keeps only the perceptual invariants
> of speech (linguistic content, speaker identity, prosody), followed by a *synthesis* map `S` that
> re-generates the waveform. Two matched results follow:
>
> - **Converse (impossibility).** By a data-processing inequality, any watermark whose signal lives
>   in the **nullspace of `A`** — phase, psychoacoustically-masked spectral regions, the parts a
>   listener can't hear — has a post-attack detection error-exponent of **exactly zero**. It is
>   provably laundered. Because masking + phase-insensitivity make that nullspace *enormous*, this is
>   exactly where post-hoc watermarks hide, which is why they all die.
> - **Achievability (what survives).** Spend the imperceptibility budget on a *preserved invariant*
>   instead, and the watermark rides through `S`. The maximal surviving payload is
>   **R\* = the capacity of the invariant sub-channel**. We give a matching invariant-aligned scheme
>   that realizes a positive rate against all three attacker families.
>
> The upshot is a **predictive theory**: from where a watermark places its energy relative to `A`,
> we predict whether it survives generative laundering and at what rate — and the measured numbers
> track the bound.

---

## Why this is new (and not a re-skin of the image result)

Regeneration attacks are known to *provably* remove **image** watermarks
([Zhao et al., NeurIPS 2024](https://arxiv.org/abs/2306.01953)). Audio is **not** a port:

| | Images (prior work) | Speech (this work) |
|---|---|---|
| Imperceptibility model | pixel `L2` / LPIPS | **psychoacoustic masking + phase-insensitivity** → a huge inaudible nullspace |
| Attacker | diffusion / VAE regeneration on a generic natural-image manifold | **analysis–resynthesis** with *concrete, measurable* preserved invariants (content/speaker/prosody) |
| Result available | impossibility only (watermark removable) | **matched converse *and* achievability** — we characterize *what survives and at what rate* |

That structure — speech resynthesis exposes a **named, estimable** invariant sub-channel — is what
lets us prove a positive result the image papers cannot state. See
[`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) for a full differentiation.

---

## The theory in one page

Detection of provenance is a binary hypothesis test between the clean-speech law `P₀` and the
watermarked law `P₁`. The optimal achievable error-exponent of that test is the **Chernoff
information** `C(P₀, P₁)`. The attacker applies the resynthesis channel `W = S∘A`.

- **Data-processing inequality.** For any channel `W`, `C(W#P₀, W#P₁) ≤ C(P₀, P₁)` — laundering can
  only *lose* detectability. (`W#P` = pushforward of `P` through `W`.)
- **Converse / nullspace erasure (Thm 1).** If the watermark perturbation `δ` satisfies `A(x+δ) =
  A(x)` (it lives in the analysis nullspace), then `W#P₁ = W#P₀`, so `C(W#P₀, W#P₁) = 0`: the
  post-attack test is *no better than a coin flip*, for **any** detector and **any** number of
  observations. Companion Wasserstein bound: the laundered signal is distributed as clean speech
  conditioned on the same invariants, so it is `≈` an unwatermarked signal.
- **Achievability / invariant sub-channel (Thm 2).** Restrict the watermark to move `z = A(x)`
  within the imperceptibility budget `D`. The surviving payload is the capacity
  `R\* = max_{q: 𝔼[d(x,x+δ)] ≤ D} I(m; A(x+δ))` of the induced sub-channel, and it is achievable by
  an invariant-aligned code. `R\* > 0` whenever the budget buys a distinguishable shift of a
  preserved coordinate. Thm 1 and Thm 2 **meet**: the survivable set is *exactly* the non-nullspace
  of `A`.

Both theorems are validated on a tractable **linear-Gaussian surrogate** (closed-form Chernoff
information and capacity) by the Monte-Carlo test suite, then demonstrated on real speech.

---

## Repository layout

```
src/rwl/            # library
  channel.py        #   resynthesis channel W = S∘A (abstract + linear-Gaussian surrogate)
  chernoff.py       #   Chernoff information / error-exponent estimators + DPI checks
  masking.py        #   psychoacoustic masking budget (imperceptibility constraint)
  watermark.py      #   invariant-aligned watermark encoder/detector (surrogate + neural hook)
  capacity.py       #   R* = invariant sub-channel capacity
experiments/        # E1-E4 runners + run_all.py (reproducibility entry point)
tests/              # theory-vs-simulation tests (pytest); the manuscript gate
paper/              # spconf LaTeX (main.tex, refs.bib), figures/
scripts/            # data fetch, server sync, plotting helpers
docs/               # RELATED_WORK.md, THEORY.md (full proofs)
data/  results/     # inputs / outputs (gitignored except small summaries)
```

## Reproducing

```bash
uv venv && uv pip install -e ".[dev]"     # CPU: theory + tests
uv run pytest                              # every theorem cross-checked vs Monte-Carlo
uv run python experiments/run_all.py       # regenerates all paper figures/numbers
```

Empirical audio experiments (E1–E4) additionally require `".[audio]"` and pretrained generators; see
[`experiments/README.md`](experiments/README.md) for the GPU workflow.

## Results at a glance

**Surrogate (E0, CPU).** As a watermark drifts from the invariant subspace into `ker(A)`,
post-laundering AUC falls from 0.89 to 0.50 (chance), matching the closed form to RMSE 0.001;
surviving detection exponent and rate `R*` are positive for the invariant mark and identically
zero for the nullspace mark. (26/26 theorem tests green.)

**Real speech (E1, LibriSpeech test-clean, N=80).** Attackers sweep the size of `ker(A)`:

| Watermark | STFT-GL (control, phase-only null) | mel-GL (lossy) | EnCodec 6k / 3k / 1.5k |
|---|---|---|---|
| Surface (nullspace) | 1.00 → 0.48 | 1.00 → 0.44 | 0.53 / 0.54 / 0.56 |
| **Invariant (mel)** | 0.94 → 0.93 | 0.94 → **0.93** | 0.74 / 0.67 / 0.60 |
| AudioSeal (deployed) | 1.00 → 1.00 | 1.00 → **0.20** | 1.00 / 0.98 / 0.94 |

(AUC before → after laundering.) The surface mark dies under every attacker; AudioSeal resists
the codec it was hardened against but **collapses under the mel-vocoder** (bit-accuracy 0.48 ≈
chance) — survival is attacker-specific and predicted by each mark's invariant-energy fraction.
The invariant mark survives the mel-vocoder and degrades *gracefully* as the codec bandwidth
(the invariant subspace) shrinks. Post-laundering AUC is a single monotone function of the
invariant-energy fraction `f`, tracking the converse curve `Φ(√f · d₀)`.

**Achievability (E2).** A 16-bit invariant payload survives the mel-vocoder (bit-accuracy
0.82 → 0.80 at 22 dB SNR, growing with budget), while a surface payload of equal size collapses
from perfect (1.00) to chance (0.50).

## Status

- [x] Theory core + Monte-Carlo tests (Thm 1 converse, DPI monotonicity, Thm 2 achievability) — 26/26 green
- [x] E1 converse validation on real speech (3 attacker families, N=80)
- [x] E2 achievability (invariant payload survives; surface payload dies)
- [x] Paper (spconf, 4 pp, 0 overfull, verified refs)
- [ ] Adversarial multi-agent polish pass (in progress)

## Citation

```bibtex
@misc{yan2026resynthesis,
  title  = {The Resynthesis Channel: Fundamental Limits of Speech Watermarking
            Against Generative Laundering},
  author = {Yan, Weiping},
  year   = {2026},
  note   = {https://github.com/appleweiping/resynthesis-watermark-limits}
}
```

## License

MIT — see [LICENSE](LICENSE).
