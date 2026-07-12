# resynthesis-watermark-limits

**The Resynthesis Channel — Fundamental Limits of Speech Watermarking Against Generative Laundering.**

A complete, reproducible signal-processing-theory research project (targeting the IEEE ICASSP
*Signal Processing Theory & Methods* / *Information Forensics & Security* tracks; ICASSP 2027
deadline Sept 2026). The **theory and every theory-validation experiment are CPU-only and fully
reproducible**; the real-speech instantiation uses pretrained neural generators and a constructive
invariant-aligned scheme on a **GPU**. Every theorem is cross-checked against Monte-Carlo
simulation by the test suite **before** it is allowed into the manuscript.

> **TL;DR** — Audio watermarks (AudioSeal, WavMark, …) are increasingly erased when an attacker
> regenerates the audio through a vocoder or codec; the field documents this collapse **empirically**
> but, for speech, has no theory of *which* watermarks die, under *which* attacker, or *what
> survives*. We give it. Model the attacker as a **resynthesis channel** `W = S∘A`: an *analysis*
> map `A` that keeps only the perceptual invariants of speech (content, speaker, prosody), followed
> by a *synthesis* map `S` that re-generates the waveform. Two matched results:
>
> - **Converse.** By a data-processing inequality, any watermark whose signal lives in the
>   **nullspace of `A`** — phase, psychoacoustically-masked regions — has a post-attack detection
>   error-exponent of **exactly zero**. Because masking + phase-insensitivity make that nullspace
>   *enormous*, this is exactly where post-hoc watermarks hide, which is why they are laundered.
> - **Achievability (what survives).** Spend the imperceptibility budget on a *preserved invariant*
>   instead, and the watermark rides through `S`. The surviving payload is **R\* = the invariant
>   sub-channel's (blind-embedder) rate**; a constructive invariant-aligned scheme realizes it.
>
> The upshot is a **predictive theory**: from where a watermark places its energy relative to `A`,
> we predict whether it survives generative laundering and at what rate — and, for a fixed channel,
> the measured numbers track the bound.

---

## Why this is new (and not a re-skin of the image result)

The image domain already has both an impossibility (regeneration provably removes image
watermarks, [Zhao et al., NeurIPS 2024](https://arxiv.org/abs/2306.01953)) **and** the positive
prescription (embed in preserved *semantic* content — Tree-Ring, Gaussian Shading). So
"embed-in-what-survives" is not unique to audio. What is new here is a **provable, quantitative**
treatment for *speech* — an explicit surviving **rate `R*`** and exponent — tied to audio's
concrete invariants:

| | Image work | Speech (this work) |
|---|---|---|
| Imperceptibility model | pixel `L2` / LPIPS | **psychoacoustic masking + phase-insensitivity** → a huge inaudible nullspace |
| Attacker | diffusion / VAE regeneration | **analysis–resynthesis** with *concrete, measurable* preserved invariants (content/speaker/prosody) |
| What is provided | impossibility (Zhao); constructions (Tree-Ring, Gaussian Shading) | **matched converse *and* achievability** with an explicit surviving *rate* `R*` |

That structure — speech resynthesis exposes a **named, estimable** invariant sub-channel — is what
lets us give a *quantitative* rate `R*` for speech, rather than only the qualitative
"embed-in-what-survives" prescription the image literature already has. See
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
  within the imperceptibility budget `D`. The surviving payload is the **blind-embedder rate**
  `R* = max_{tr(M_row Q) ≤ D} ½ log det(I + Q)` of the induced sub-channel (host as interference;
  informed/dirty-paper coding could exceed it), achievable by an invariant-aligned code.
  `R* > 0` whenever the budget buys a distinguishable shift of a preserved coordinate. Thm 1 and
  Thm 2 **meet**: the survivable set is *exactly* the non-nullspace of `A`.

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

(AUC = separability, the primary metric — it is what the Chernoff exponent governs.) The surface
mark dies under every attacker. **AudioSeal** resists the EnCodec it was hardened against but is
**defeated operationally by mel-spectrogram inversion**: TPR@1%FPR 1.00→0.03 and payload
bit-accuracy 0.48 (chance); its AUC of 0.20 is a sign-inversion (residual separability),
consistent with its small but *nonzero* mel invariant-fraction 0.26. The **invariant mark's
separability** survives the mel channel (AUC 0.94→0.93) and degrades *gracefully* as the codec
bandwidth (invariant subspace) shrinks. Survival is channel-specific and predicted by each
mark's invariant-energy fraction `f`; for a fixed channel, post-laundering AUC is monotone in
`f`, well described by the converse's √f form `Φ(a√f+b)`.

> **Honest caveats.** Our synthesizers are STFT/mel-spectrogram inversion (Griffin–Lim) and a
> neural codec (EnCodec); a learned neural vocoder / voice conversion is future work. The
> constructive invariant embedder is a proof-of-concept — strong in AUC but weak at a stringent
> 1% FPR (TPR ≈ 0.11 even on clean audio); a learned embedder would tighten it. Marks are matched
> by SNR (24 dB), not a perceptual metric (PESQ/ViSQOL) — a limitation we flag.

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
