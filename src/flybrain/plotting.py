"""One visual system for every journal figure (validated reference palette, light surface)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from flybrain.paths import FIGURES  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed order: blue, orange, aqua, yellow
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
NEUTRAL = "#f0efec"
DIVERGING_LOW = "#2a78d6"
DIVERGING_HIGH = "#e34948"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "text.color": TEXT_PRIMARY,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "font.size": 10,
        "legend.frameon": False,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def diverging_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("flybrain_div", [DIVERGING_LOW, NEUTRAL, DIVERGING_HIGH])


def save_figure(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    return path
