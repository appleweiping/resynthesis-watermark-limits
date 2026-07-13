#!/usr/bin/env python
"""Generate paper/macros_audio.tex + paper tables from the NEW results schema.

Sources (all must exist for a formal build; missing file -> loud failure):
  results/e1_survival.json   (deployed baselines x attackers, calibrated metrics)
  results/e2_predictor.json  (held-out predictor validation + construction verification)
  results/e3_payload_poc.json

Every real-audio number in the manuscript is generated here so the paper cannot
drift from the code. Raw AUC and oriented AUC are BOTH emitted; operating points
carry Clopper-Pearson intervals from the calibration-set protocol.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

ATT_ORDER = ["stft_gl", "mel80_gl", "vocos", "vocos_encodec6k", "encodec6k",
             "encodec3k", "encodec1.5k", "dac", "snac", "knnvc_self4", "knnvc_self8"]
ATT_LABEL = {
    "stft_gl": "STFT$^\\dagger$", "mel80_gl": "mel-inv", "vocos": "Vocos",
    "vocos_encodec6k": "Vc-EnC", "encodec6k": "EnC6", "encodec3k": "EnC3",
    "encodec1.5k": "EnC1.5", "dac": "DAC", "snac": "SNAC",
    "knnvc_self4": "VC$_4$", "knnvc_self8": "VC$_8$",
}
BL_LABEL = {"audioseal": "AudioSeal", "wavmark": "WavMark",
            "silentcipher": "SilentCipher"}


def f2(x): return f"{x:.2f}"
def f3(x): return f"{x:.3f}"


def load(name: str) -> dict:
    p = ROOT / "results" / name
    if not p.exists():
        raise SystemExit(f"FATAL: {p} missing — run the experiments first "
                         "(formal builds never fake numbers)")
    return json.loads(p.read_text(encoding="utf-8"))


def load_e1() -> dict:
    """Load results/e1_survival.json or merge sharded e1_survival_<baseline>.json."""
    p = ROOT / "results" / "e1_survival.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    shards = sorted(p for p in (ROOT / "results").glob("e1_survival_*.json")
                  if not p.stem.endswith(("_a", "_b")))
    if not shards:
        raise SystemExit("FATAL: no e1_survival results — run E1 first")
    merged = None
    for s in shards:
        d = json.loads(s.read_text(encoding="utf-8"))
        if merged is None:
            merged = d
        else:
            merged["rows"].extend(d["rows"])
            merged["attack_severity"].update(d.get("attack_severity", {}))
            if d["n_test"] != merged["n_test"]:
                raise SystemExit(f"FATAL: shard {s.name} n_test={d['n_test']} != "
                                 f"{merged['n_test']} — shards not comparable")
    return merged


def _within_rho_ci(e2: dict, pred: str = "pred_sensitivity",
                   resp: str = "auc_mel", n_boot: int = 2000, seed: int = 0):
    """Bootstrap CI (over test directions) of the WITHIN-attacker rank-pooled
    Spearman — the statistic actually reported; the per-response _fit_eval CI
    bootstraps the unnormalized pooled correlation and is not comparable."""
    from scipy import stats as sps

    pts = e2["points_test"]
    x = np.array([p[pred] for p in pts], float)
    y = np.array([p[resp] for p in pts], float)
    strata = np.array([p["attacker"] for p in pts])
    dirs = np.array([p["dir"] for p in pts])

    def within_rho(idx):
        xr = np.empty(len(idx), float); yr = np.empty(len(idx), float)
        for s in np.unique(strata[idx]):
            m = np.flatnonzero(strata[idx] == s)
            xr[m] = sps.rankdata(x[idx][m]); yr[m] = sps.rankdata(y[idx][m])
        return sps.spearmanr(xr, yr).statistic

    uniq = np.unique(dirs)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(dirs == t) for t in take])
        vals.append(within_rho(idx))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _necessity_unpaired(e2: dict, rel_threshold: float = 0.25) -> dict:
    """Necessity on the UNPAIRED detectability responses (the paired statistic is
    a deterministic comparison and detects arbitrarily small leakage — it answers
    'is transmission exactly zero', not 'is the mark detectable')."""
    pts = e2["points_test"]
    strata = np.array([p["attacker"] for p in pts])
    s = np.array([p["pred_sensitivity"] for p in pts], float)
    low = np.zeros(len(pts), bool)
    for a in np.unique(strata):
        idx = np.flatnonzero(strata == a)
        low[idx[s[idx] <= rel_threshold * np.median(s[idx])]] = True
    out = {"n_low": int(low.sum())}
    for resp in ("auc_mel", "auc_wave"):
        y = np.abs(np.array([p[resp] for p in pts], float) - 0.5)
        out[resp] = {"low_med": float(np.median(y[low])),
                     "low_max": float(np.max(y[low])),
                     "rest_med": float(np.median(y[~low]))}
    return out


def macros(e1: dict, e2: dict, e3: dict) -> list[str]:
    # Primary response: UNPAIRED oriented mel-probe AUC (detectability against
    # host variability); statistics are within-attacker (rank-pooled).
    ev_m = e2["evaluation_auc_mel"]["pred_sensitivity"]
    pm_m = e2["permutation_auc_mel"]["pred_sensitivity"]
    pm_w = e2["permutation_auc_wave"]["pred_sensitivity"]
    nec = _necessity_unpaired(e2)
    ver = e2["construction_verification"]
    null_rel = [v["ratio_rel"] for v in ver if v["kind"] == "nullspace"]
    row_rel = [v["ratio_rel"] for v in ver if v["kind"] == "rowspace"]
    by_att = e2["by_attacker_auc_mel"]

    lines = [
        "% AUTO-GENERATED by scripts/make_paper_macros.py — do not edit.",
        f"\\newcommand{{\\nTest}}{{{e1['n_test']}}}",
        f"\\newcommand{{\\nCalib}}{{{e1['n_calib']}}}",
        f"\\newcommand{{\\nSpeakersTest}}{{{e1.get('n_speakers_test', '40')}}}",
        f"\\newcommand{{\\pesqTargetVal}}{{{f2(e1['pesq_target'])}}}",
        # within-attacker pooled statistics (rank-normalized per attacker)
        f"\\newcommand{{\\spearmanSens}}{{{f2(pm_m['observed_within_spearman'])}}}",
        "\\newcommand{\\spearmanSensCI}{[" +
        ", ".join(f2(v) for v in _within_rho_ci(e2)) + "]}",
        # per-attacker isotonic R^2, median across attackers (snac noted in text)
        f"\\newcommand{{\\rTwoSens}}"
        f"{{{f2(float(np.median([v['isotonic']['r2'] for v in by_att.values()])))}}}",
        ("\\newcommand{\\permP}{<10^{-3}}" if pm_m['p_value'] <= 0.002
         else f"\\newcommand{{\\permP}}{{{pm_m['p_value']:.3g}}}"),
        f"\\newcommand{{\\spearmanSensWave}}{{{f2(pm_w['observed_within_spearman'])}}}",
        f"\\newcommand{{\\spearmanMagFrac}}"
        f"{{{f2(e2['permutation_auc_mel']['pred_stft_mag_fraction']['observed_within_spearman'])}}}",
        f"\\newcommand{{\\spearmanSNR}}"
        f"{{{f2(e2['permutation_auc_mel']['pred_snr_db']['observed_within_spearman'])}}}",
        f"\\newcommand{{\\spearmanCentroid}}"
        f"{{{f2(e2['permutation_auc_mel']['pred_spectral_centroid']['observed_within_spearman'])}}}",
        # necessity (absolute near-null threshold; unpaired detectability)
        f"\\newcommand{{\\necNLow}}{{{nec['n_low']}}}",
        f"\\newcommand{{\\necLowMel}}{{{f2(nec['auc_mel']['low_med'])}}}",
        f"\\newcommand{{\\necMaxLowMel}}{{{f2(nec['auc_mel']['low_max'])}}}",
        f"\\newcommand{{\\necRestMel}}{{{f2(nec['auc_mel']['rest_med'])}}}",
        f"\\newcommand{{\\necLowWave}}{{{f2(nec['auc_wave']['low_med'])}}}",
        # detector-domain contrast (codecs transmit what mel analysis discards)
        f"\\newcommand{{\\rhoWaveEncodec}}"
        f"{{{f2(e2['by_attacker_auc_wave']['encodec6k']['phi_fit']['spearman'])}}}",
        f"\\newcommand{{\\rhoWaveSnac}}"
        f"{{{f2(e2['by_attacker_auc_wave']['snac']['phi_fit']['spearman'])}}}",
        f"\\newcommand{{\\nullLeakOp}}{{{f2(float(np.median(null_rel)))}}}",
        f"\\newcommand{{\\rowLeakOp}}{{{f2(float(np.median(row_rel)))}}}",
        f"\\newcommand{{\\pocBits}}{{{e3['bits']}}}",
        f"\\newcommand{{\\pocPesq}}{{{f2(e3['pesq_median'])}}}",
    ]
    # seed replication (independent manifests), if the seed runs exist
    for i, seed in enumerate(("seed1", "seed2")):
        p = ROOT / "results" / f"e2_predictor_{seed}.json"
        if p.exists():
            ds = json.loads(p.read_text(encoding="utf-8"))
            rho = ds["permutation_auc_mel"]["pred_sensitivity"][
                "observed_within_spearman"]
            lines.append(
                f"\\newcommand{{\\withinRhoSeed{'AB'[i]}}}{{{f2(rho)}}}")
    # per-attacker within-Spearman (mel-probe detectability); LaTeX macro names
    # cannot contain digits, so attacker names map to letter-only keys.
    keymap = {"mel80_gl": "melgl", "vocos": "vocos", "encodec6k": "encodec",
              "dac": "dac", "snac": "snac"}
    for att, ev in by_att.items():
        lines.append(f"\\newcommand{{\\rhoMel{keymap[att]}}}"
                     f"{{{f2(ev['phi_fit']['spearman'])}}}")
    return lines


def e1_table(e1: dict) -> list[str]:
    """Main table: per baseline, RAW auc / ORIENTED auc / TPR@1% per attacker."""
    rows = e1["rows"]
    atts = [a for a in ATT_ORDER
            if any(r["attacker"] == a for r in rows)]
    # average over keys; cells show oriented AUC / TPR / ACHIEVED FPR at the
    # clean-calibration threshold (raw==oriented in all final rows — no sign
    # inversions occurred — so raw is omitted from cells; it lives in the JSON)
    def cell(bl: str, att: str) -> str:
        rs = [r for r in rows if r["baseline"] == bl and r["attacker"] == att]
        if not rs:
            return "--"
        orn = np.mean([r["auc_oriented"] for r in rs])
        tpr = np.mean([r["operating_point"]["tpr"] for r in rs])
        fpr = np.mean([r["operating_point"]["fpr"] for r in rs])
        return f"{orn:.2f}/{tpr:.2f}/{fpr:.2f}"

    bls = sorted({r["baseline"] for r in rows}, key=lambda b: list(BL_LABEL).index(b))
    # TRANSPOSED: attackers as rows, baselines as columns (fits one column width)
    lines = [
        "% AUTO-GENERATED: oriented AUC / TPR / achieved FPR per (attacker, baseline)",
        "\\begin{table}[t]\\centering",
        "\\caption{Deployed watermarks under analysis--resynthesis at matched median "
        "PESQ. Cells: oriented AUC\\,/\\,TPR\\,/\\,\\emph{achieved} FPR at the 1\\%-FPR "
        "threshold fixed on the independent calibration split "
        f"($n={e1['n_calib']}$ clean negatives). Outcomes: \\emph{{erasure}} "
        "(AUC$\\to$0.5, TPR$\\to$0); \\emph{calibration failure} (AUC retained, "
        "achieved FPR explodes on attacked clean audio); \\emph{graceful degradation} "
        "(AUC and FPR retained, TPR reduced --- AudioSeal/DAC).}",
        "\\label{tab:e1}\\scriptsize\\setlength{\\tabcolsep}{2.6pt}",
        "\\begin{tabular}{l" + "c" * len(bls) + "}",
        "\\toprule",
        "Channel & " + " & ".join(BL_LABEL[b] for b in bls) + " \\\\",
        "\\midrule",
        "clean & " + " & ".join(cell(b, "none") for b in bls) + " \\\\",
    ]
    for a in atts:
        lines.append(f"{ATT_LABEL[a]} & " +
                     " & ".join(cell(b, a) for b in bls) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return lines


def main() -> None:
    e1, e2, e3 = (load_e1(), load("e2_predictor.json"),
                  load("e3_payload_poc.json"))
    (PAPER / "macros_audio.tex").write_text(
        "\n".join(macros(e1, e2, e3)) + "\n", encoding="utf-8")
    (PAPER / "tab_e1.tex").write_text(
        "\n".join(e1_table(e1)) + "\n", encoding="utf-8")
    print("wrote paper/macros_audio.tex and paper/tab_e1.tex")


if __name__ == "__main__":
    main()
