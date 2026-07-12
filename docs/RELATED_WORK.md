# Related work and differentiation

We separate three literatures and state precisely what is new here.

## 1. Empirical audio watermarking (schemes)

Neural and classical audio watermarks — **AudioSeal**, **WavMark**, **IDEAW**,
**TriniMark**, **AWARE**, **FakeMark**, and psychoacoustic/spread-spectrum methods — embed
a payload and report robustness to compression, noise, resampling, and filtering. These
are *constructions*; none proves a fundamental limit, and (see §2) they are the objects
our converse explains.

*Difference:* we do not propose another benchmark-chasing scheme. We prove **why** a broad
class of these (post-hoc, imperceptibility-tuned) schemes must fail against resynthesis,
and characterize what any scheme could retain.

## 2. Attacks / SoK on audio watermark robustness

Multiple 2024–2026 works show watermarks collapse under **neural regeneration**: vocoder
resynthesis, neural-codec round-trips, and **self voice conversion** drive detection to
near chance (e.g. reported AudioSeal accuracy of 0.09–0.28 under vocoder reconstruction).
SoK-style studies conclude *no* audio watermark is robust across all transformation
classes, and "post-hoc watermarks are shallow." Overwriting and re-watermarking attacks
add further removal strategies.

*Difference:* this literature documents the collapse **empirically, case by case**. We
give the **missing theory**: the collapse is a data-processing inequality for the analysis
map `A`, with the surviving detectability equal to `⅛‖Pδ‖²`. Our E1 reproduces the
collapse and shows it lands at the theory's predicted floor — not merely "it drops."

## 3. Provable watermark removal for images

Zhao et al., *"Invisible Image Watermarks Are Provably Removable Using Generative AI"*
(NeurIPS 2024), prove diffusion/VAE regeneration removes pixel-space image watermarks and
that the regenerated image is close to the clean one. Related image analyses (spectral
projection attacks; "when denoising becomes unsigning" for diffusion editing) study the
same phenomenon in vision.

*Difference (this is the crux of novelty):*

| Axis | Image work (Zhao et al.) | This work (speech) |
|---|---|---|
| Imperceptibility | pixel `L2`/LPIPS | psychoacoustic masking + phase-insensitivity ⇒ a **structured, enormous** nullspace |
| Attacker model | regeneration on a generic natural-image manifold | **analysis–resynthesis** `W=S∘A` with *named, estimable* invariants (content/speaker/prosody) |
| Result | impossibility only (removable) | **matched converse *and* achievability**: exactly what survives (`row(A)`) and at what rate (`R*`) |
| Prediction | watermark removable | *quantitative* surviving exponent `⅛‖Pδ‖²`; rate–survival curve verified on real speech |

A generic port of the image impossibility to audio would be stitching. The audio structure
— resynthesis exposes a concrete invariant sub-channel — is what lets us state a **positive
result the image papers cannot**: an invariant-aligned watermark that provably survives.

## 4. Information-theoretic watermarking (classical)

Costa's "writing on dirty paper," Moulin–O'Sullivan information-theoretic watermarking, and
QIM establish capacities under **fixed additive attacks** and (for informed embedding)
host-cancellation. Our attacker is not additive: it is a **nonlinear projection onto a
generative manifold**, and our detector is **blind** (host as interference). The novelty is
locating watermark survivability in the geometry of the resynthesis map `A`, giving a DPI
converse and a matching blind-detector rate `R*` specific to generative laundering.

## Positioning sentence

*We give the first signal-processing-theoretic account of why generative resynthesis
launders speech watermarks — a data-processing converse that pinpoints the erased
(nullspace) versus surviving (invariant) components — together with a matching achievable
rate and an invariant-aligned scheme that realizes it, validated on real neural vocoders,
codecs, and voice-conversion systems.*
