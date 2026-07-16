r"""Reproducibility entry point — regenerate every figure and number in the paper.

Usage:
    python experiments/run_all.py            # CPU surrogate (E0) — always runs
    python experiments/run_all.py --audio    # + real-speech E1-E4 (needs .[audio] + GPU)

The CPU surrogate (E0) validates both theorems and is what the test suite gates. The audio
stages instantiate the resynthesis channel with real neural vocoders/codecs/voice-conversion
and require pretrained models; see experiments/README.md for the GPU workflow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", action="store_true",
                        help="also run real-speech experiments E1-E4 (GPU)")
    args = parser.parse_args()

    import e0_surrogate
    print("=" * 70)
    print("E0 — surrogate demonstration of Theorems 1 and 2 (CPU)")
    print("=" * 70)
    e0_surrogate.main()

    if args.audio:
        print("=" * 70)
        print("The real-speech pipeline (E1/E2/E3) is now a separate one-command runner")
        print("with its own arguments (dataset root, device, seeds):")
        print("    python experiments/run_audio_all.py --root <LibriSpeech parent> "
              "--device cuda")
        print("=" * 70)
    print("\nDone. Figures in paper/figures/, numbers in results/.")


if __name__ == "__main__":
    main()
