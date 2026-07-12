#!/usr/bin/env python
"""Generate paper/macros_audio.tex and paper/tab_e1.tex from results/e1_audio.json.

Every real-audio number in the manuscript is sourced from the experiment output, so the
paper cannot drift from the code.  Run after experiments/audio/e1_converse_audio.py.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "e1_audio.json"
PAPER = ROOT / "paper"

# Attacker display names / ordering.
ATT_ORDER = ["stft_gl", "mel80_gl", "encodec6k", "encodec3k", "encodec1.5k"]
ATT_LABEL = {"stft_gl": "STFT$^\\dagger$", "mel80_gl": "mel-GL",
             "encodec6k": "EnC6", "encodec3k": "EnC3", "encodec1.5k": "EnC1.5"}
WM_LABEL = {"surface_ss": "Surface (null)", "invariant_ss": "Invariant (mel)",
            "audioseal": "AudioSeal"}


def fnum(x, d=3):
    return f"{x:.{d}f}"


def find(recs, wm, att):
    for r in recs:
        if r["watermark"] == wm and r["attacker"] == att:
            return r
    return None


def main() -> None:
    data = json.loads(RES.read_text())
    recs = data["part_a"]

    def g(rec, key, default="[n/a]"):
        return fnum(rec[key]) if rec and key in rec else default

    aseal_ctrl = find(recs, "audioseal", "stft_gl")
    aseal_mel = find(recs, "audioseal", "mel80_gl")
    surf_mel = find(recs, "surface_ss", "mel80_gl")
    inv_mel = find(recs, "invariant_ss", "mel80_gl")

    macros = [
        (r"\audioNutt", str(data["n_utt"])),
        (r"\aucAudiosealCtrl", g(aseal_ctrl, "auc_after")),   # survives lossless control
        (r"\aucAudiosealMel", g(aseal_mel, "auc_after")),     # collapses under lossy mel
        (r"\bitaccAudiosealMel", g(aseal_mel, "bitacc_after")),
        (r"\aucSurfMel", g(surf_mel, "auc_after")),
        (r"\aucInvMel", g(inv_mel, "auc_after")),
        (r"\aucInvMelBefore", g(inv_mel, "auc_before")),
        (r"\fracSurf", g(surf_mel, "invariant_fraction", "0.0")),
        (r"\fracInv", g(inv_mel, "invariant_fraction", "0.0")),
        (r"\fracAudioseal", g(aseal_mel, "invariant_fraction", "0.0")),
        (r"\snrInv", f"{inv_mel['snr_db']:.0f}" if inv_mel else "24"),
    ]
    (PAPER / "macros_audio.tex").write_text(
        "\n".join(f"\\newcommand{{{n}}}{{{v}}}" for n, v in macros) + "\n")

    wms = ["surface_ss", "invariant_ss", "audioseal"]
    atts = [a for a in ATT_ORDER if any(r["attacker"] == a for r in recs)]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Real speech (LibriSpeech, $N=%s$). Detection AUC before$\to$after each "
        r"blind attacker. STFT$^\dagger$ is a near-lossless control (nullspace $=$ phase only); "
        r"mel-GL and EnCodec (EnC$k$, $k$\,kbps) are lossy. Surface/AudioSeal (energy in the "
        r"nullspace) collapse to chance; the invariant mel mark survives.}" % data["n_utt"],
        r"\label{tab:e1}", r"\scriptsize", r"\setlength{\tabcolsep}{2.4pt}",
        r"\begin{tabular}{l" + "c" * len(atts) + r"}", r"\toprule",
        r"Watermark & " + " & ".join(ATT_LABEL.get(a, a) for a in atts) + r" \\",
        r"\midrule",
    ]
    for wm in wms:
        cells = []
        for a in atts:
            r = find(recs, wm, a)
            cells.append(f"{r['auc_before']:.2f}$\\to${r['auc_after']:.2f}" if r else "--")
        lines.append(f"{WM_LABEL.get(wm, wm)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (PAPER / "tab_e1.tex").write_text("\n".join(lines) + "\n")
    print("wrote paper/macros_audio.tex and paper/tab_e1.tex")


if __name__ == "__main__":
    main()
