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

*Difference.* The image community already has both the impossibility (Zhao et al.) **and** the
positive prescription — Zhao et al. advocate embedding in preserved *semantic* content, and
Tree-Ring (Wen et al., NeurIPS 2023) and Gaussian Shading (Yang et al., CVPR 2024) realize
regeneration-robust semantic image watermarks. So "embed in what survives" is **not** unique to
audio. What is new here is a **quantitative, provable** treatment for *speech*:

| Axis | Image work | This work (speech) |
|---|---|---|
| Imperceptibility | pixel `L2`/LPIPS | psychoacoustic masking + phase-insensitivity ⇒ a **structured, enormous** nullspace |
| Attacker model | diffusion/VAE regeneration | **analysis–resynthesis** `W=S∘A` with *named, estimable* invariants (content/speaker/prosody) |
| What is provided | impossibility (Zhao); constructions (Tree-Ring, Gaussian Shading) | **matched converse *and* achievability** with an explicit surviving *rate* `R*` and exponent `⅛‖Pδ‖²` |

Our contribution is the **rate/limit** (`R*`, the surviving exponent) for the speech resynthesis
channel and its verification on real speech — not the qualitative "embed-in-what-survives" idea,
which the image literature already has.

## 4. Information-theoretic watermarking (classical)

Costa's "writing on dirty paper," Moulin–O'Sullivan information-theoretic watermarking, and
QIM establish capacities under **fixed additive attacks** and (for informed embedding)
host-cancellation. Our attacker is not additive: it is a **nonlinear projection onto a
generative manifold**, and our detector is **blind** (host as interference). The novelty is
locating watermark survivability in the geometry of the resynthesis map `A`, giving a DPI
converse and a matching blind-detector rate `R*` specific to generative laundering.

## Positioning sentence

*We give a provable, quantitative signal-processing account of why analysis–resynthesis
launders speech watermarks — a data-processing converse that pinpoints the erased (nullspace)
versus surviving (invariant) components — together with a matching achievable rate `R*` and an
invariant-aligned scheme that realizes it, validated on mel/STFT-spectrogram inversion and a
neural codec (EnCodec).*
