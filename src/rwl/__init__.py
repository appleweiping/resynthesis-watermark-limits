"""rwl — A subspace model of speech-watermark survival under analysis–resynthesis.

This package provides:
  * `channel`   — the analysis–resynthesis channel W = S∘A: general DPI/exact-erasure
                  proposition (docstring) + the linear-Gaussian surrogate (code),
  * `chernoff`  — Chernoff-information / detection-error-exponent estimators and the
                  data-processing-inequality check,
  * `masking`   — the perceptual imperceptibility budget (quadratic surrogate),
  * `capacity`  — R_LB, a Gaussian achievable lower bound for blind embedders
                  (NOT a watermarking capacity; no rate converse is claimed),
  * `watermark` — nullspace vs invariant-aligned watermark constructions and their
                  matched detectors, in the surrogate.

The numerical implementation is sanity-checked against the closed-form surrogate
quantities by the test suite; whether real attackers behave like the surrogate is an
empirical question answered by the experiments, not by these modules.
"""

from . import channel, chernoff, masking, capacity, watermark

__all__ = ["channel", "chernoff", "masking", "capacity", "watermark"]
__version__ = "0.2.0"
