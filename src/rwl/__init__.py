"""rwl — The Resynthesis Channel.

Fundamental limits of speech watermarking against generative laundering.

This package provides:
  * `channel`   — the resynthesis channel W = S∘A (abstract + linear-Gaussian surrogate),
  * `chernoff`  — Chernoff-information / detection-error-exponent estimators and the
                  data-processing-inequality check that underpins the converse,
  * `masking`   — the psychoacoustic imperceptibility budget,
  * `capacity`  — the invariant sub-channel capacity R* (achievability),
  * `watermark` — nullspace (post-hoc) vs invariant-aligned watermark constructions
                  and their optimal detectors.

The linear-Gaussian surrogate in these modules makes both theorems closed-form so the
test suite can cross-check every claim against Monte-Carlo before it enters the paper.
"""

from . import channel, chernoff, masking, capacity, watermark

__all__ = ["channel", "chernoff", "masking", "capacity", "watermark"]
__version__ = "0.1.0"
