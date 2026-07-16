"""One-command reproduction of every audio result in the paper.

    python experiments/run_audio_all.py --root <LibriSpeech parent> --device cuda

Steps (all strict — a missing attacker/baseline/model ABORTS the run):
  1. build the seeded manifests (3 seeds),
  2. E1  deployed-watermark survival (calibrated operating points),
  3. E2  held-out predictor validation,
  4. E3  multi-bit proof of concept,
  5. regenerate paper macros/tables/figures from the result JSONs.

Outputs land in results/ (JSON + per-clip score arrays) and paper/ (macros, tables,
figures). Seeds/keys are fixed inside each script; nothing depends on wall clock.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        sys.exit(f"FAILED ({proc.returncode}): {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir containing LibriSpeech/")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-test", type=int, default=1000)
    args = ap.parse_args()

    py = sys.executable
    seeds = args.seeds.split(",")
    for s in seeds:
        run([py, "scripts/make_manifest.py", "--root", args.root, "--seed", s,
             "--out", f"data/manifest_seed{s}.json"])
    # calibration-only manifests (speaker-disjoint from test-clean) so the E1
    # calibration pool clears the >=5000-negative spec in a single pass
    extra = []
    for part in ("dev-other", "test-other"):
        out = f"data/manifest_calib_{part}.json"
        run([py, "scripts/make_manifest.py", "--root", args.root, "--seed", "0",
             "--calib-extra-from", part, "--n-calib", "4000", "--out", out])
        extra.append(out)

    # pin/verify the model set (P0-4): capture models.lock.json once if absent, then
    # every experiment verifies checkpoint hashes against it at startup.
    if not (ROOT / "models.lock.json").exists():
        run([py, "scripts/capture_model_lock.py", "--device", args.device])

    m0 = f"data/manifest_seed{seeds[0]}.json"
    run([py, "-m", "experiments.audio.e1_survival", "--manifest", m0,
         "--device", args.device, "--n-test", str(args.n_test),
         "--extra-calib", ",".join(extra), "--strict"])
    # EXACT parameters that generate the paper's E2 numbers (P0-3): 108 directions x
    # 32 utts/dir, PESQ-matched budget, determinism seed 0. Changing any of these
    # changes the reported statistics, so they are pinned here, not left to defaults.
    run([py, "-m", "experiments.audio.e2_predictor", "--manifest", m0,
         "--device", args.device, "--n-dirs", "108", "--utts-per-dir", "32",
         "--max-utts", "400", "--seed", seeds[0], "--strict"])
    run([py, "-m", "experiments.audio.e3_payload_poc", "--manifest", m0,
         "--device", args.device, "--strict"])
    # seed robustness for the headline predictor result (independent manifests +
    # RNG offset via --seed; smaller 60-direction sweeps to bound cost)
    for s in seeds[1:]:
        run([py, "-m", "experiments.audio.e2_predictor",
             "--manifest", f"data/manifest_seed{s}.json", "--device", args.device,
             "--n-dirs", "60", "--utts-per-dir", "32", "--max-utts", "400",
             "--seed", s, "--strict",
             "--out", f"results/e2_predictor_seed{s}.json"])
    run([py, "scripts/make_paper_macros.py"])
    print("ALL DONE")


if __name__ == "__main__":
    main()
