"""Shared plotting style and paths for experiments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "paper" / "figures"
RESDIR = ROOT / "results"
FIGDIR.mkdir(parents=True, exist_ok=True)
RESDIR.mkdir(parents=True, exist_ok=True)


def use_paper_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "lines.linewidth": 1.6,
            "figure.autolayout": True,
            "savefig.bbox": "tight",
        }
    )


def save_fig(fig: plt.Figure, name: str) -> Path:
    path = FIGDIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


def save_json(obj: dict, name: str) -> Path:
    path = RESDIR / name
    path.write_text(json.dumps(obj, indent=2))
    return path
