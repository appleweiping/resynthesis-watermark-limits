"""Reproducibility controls shared by all audio experiments (P0-3).

`set_determinism(seed)` pins every RNG that can influence a run: Python hashing,
Python `random`, NumPy, and Torch (CPU + all CUDA devices), and switches Torch and
cuDNN into deterministic-algorithm mode. Griffin-Lim is separately forced to a fixed
(non-random) phase seed in `attackers.py` (`rand_init=False`). Call this ONCE at the
top of every experiment `main()` before any model or data touches an RNG.

PYTHONHASHSEED only takes effect if set before the interpreter starts, so we also
export it here and re-exec is NOT attempted; instead all keyed constructions use a
stable content hash (`stable_key`) that does not depend on Python's salted `hash()`.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch

PYTHONHASHSEED = "0"


def set_determinism(seed: int = 0, strict: bool = False) -> None:
    """Pin every RNG. ``strict=True`` also sets ``use_deterministic_algorithms(
    warn_only=False)`` so any nondeterministic operator RAISES rather than silently
    running — use it to *audit* whether a pipeline can be byte-identical. Pretrained
    GPU vocoders/codecs may include ops without deterministic kernels; when strict
    raises, the honest claim is STATISTICAL reproducibility (two runs compared on
    canonicalized JSON within a stated tolerance), not byte-identity. See
    scripts/check_macros_reproducible.py --hash for the two-run comparison."""
    os.environ["PYTHONHASHSEED"] = PYTHONHASHSEED
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # cuBLAS determinism
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=not strict)
    except Exception:
        if strict:
            raise


def stable_key(text: str, mod: int = 50_000) -> int:
    """Deterministic key from a string via SHA-256[:8] — independent of the
    process-salted builtin ``hash`` (which varies run to run unless PYTHONHASHSEED
    is set before interpreter start). Used to seed per-utterance mark construction."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "big") % mod
