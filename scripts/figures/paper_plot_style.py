"""Shared, print-safe plotting style for CBMJev empirical figures."""
from __future__ import annotations

from pathlib import Path


COLORS = {
    "fixed": "#0072B2",
    "random": "#999999",
    "static": "#D55E00",
    "value": "#009E73",
    "value_singleton": "#56B4E9",
    "static_value": "#E69F00",
    "all": "#CC79A7",
}


def apply_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.size": 9,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",
    })


def save_figure(fig, output: str | Path) -> None:
    import matplotlib.pyplot as plt

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "CBMJev reproducible figure pipeline",
                "CreationDate": None, "ModDate": None}
    fig.savefig(output, metadata=metadata)
    plt.close(fig)
