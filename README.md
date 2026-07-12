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

**Real speech (E1, LibriSpeech test-clean, N=80).** Attackers sweep the size of `ker(A)` — from a
near-lossless STFT control, through a **lossy mel-inversion and a real neural vocoder (Vocos)**,
to EnCodec at falling bitrates. Detection **AUC after** each channel (before ≈ 1.00, except the
invariant mark 0.94):

| Watermark | STFT† (control) | mel-inv. | **Vocos (neural vocoder)** | EnC 6k / 3k / 1.5k |
|---|---|---|---|---|
| Surface (phase/nullspace) | 0.45 | 0.54 | 0.49 | 0.53 / 0.54 / 0.56 |
| **Invariant (mel envelope)** | 0.93 | 0.93 | **0.93** | 0.74 / 0.67 / 0.60 |
| AudioSeal (deployed) | 0.98 | **0.15** | **0.33** | 0.99 / 0.98 / 0.94 |

(AUC = separability, the metric the Chernoff exponent governs.) The phase-domain surface mark
dies under every channel. **AudioSeal collapses under both the mel-inversion *and* the real
neural vocoder** (AUC 0.15 / 0.33, TPR@1%FPR 0.04 / 0.05, payload at chance) yet resists the
EnCodec it was hardened against — survival is channel-specific. The **invariant mark's
separability survives both vocoders** (0.94→0.93, 95% CI [0.88, 0.97]) and erodes *gracefully* as
the EnCodec bitrate shrinks the invariant subspace (`R*` in action). In a **single-fixed-detector
sweep**, post-laundering AUC is monotone in the invariant-energy fraction `f`, tracking the
converse's √f form `Φ(a√f+b)`.

**Perception (PESQ):** the invariant mark is **near-transparent (4.61**, vs AudioSeal's 4.53) —
its survival is *not* bought with audibility; at matched SNR it is far *less* audible than the
surface mark (1.99).

> **Honest caveats.** We test spectral inversion, a **neural vocoder (Vocos)**, and a neural codec
> (EnCodec); **voice conversion** is the main remaining laundering family (future work). The
> constructive invariant embedder is a proof-of-concept — strong in AUC and PESQ but weak at a
> stringent 1% FPR (TPR ≈ 0.11 even on clean audio); a learned embedder would tighten it.

**Achievability (E2).** A 16-bit invariant payload survives the mel-vocoder (bit-accuracy
0.82 → 0.80 at 22 dB SNR, growing with budget), while a surface payload of equal size collapses
from perfect (1.00) to chance (0.50).

## Status

- [x] Theory core + Monte-Carlo tests (Thm 1 converse, DPI monotonicity, Thm 2 achievability) — 26/26 green
- [x] E1 converse validation on real speech (STFT/mel inversion, **neural vocoder (Vocos)**, EnCodec; N=80)
- [x] E2 achievability (invariant payload survives both vocoders; surface payload dies)
- [x] PESQ perceptual quality + bootstrap AUC confidence intervals
- [x] Paper (spconf, 4 pp, 0 overfull, verified refs)
- [x] Adversarial multi-agent polish — **4 rounds, converged clean** (r1: 8 majors → r4: 0 findings)

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
