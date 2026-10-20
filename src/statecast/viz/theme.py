"""Shared chart styling: colors by role, and matplotlib defaults.

Colors come from a validated reference palette. Marks carry series colors;
all text uses ink tokens, never a series color.
"""

from typing import Final

from matplotlib.typing import RcKeyType

SURFACE: Final = "#fcfcfb"
INK_PRIMARY: Final = "#0b0b0b"
INK_SECONDARY: Final = "#52514e"
INK_MUTED: Final = "#898781"
GRIDLINE: Final = "#e1e0d9"
AXIS_LINE: Final = "#c3c2b7"
SERIES_BLUE: Final = "#2a78d6"
SERIES_ORANGE: Final = "#eb6834"
SERIES_AQUA: Final = "#1baf7a"

FONT_FAMILY: Final = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

RC_PARAMS: Final[dict[RcKeyType, object]] = {
    "font.family": "sans-serif",
    "font.sans-serif": FONT_FAMILY,
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS_LINE,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "legend.frameon": False,
}
