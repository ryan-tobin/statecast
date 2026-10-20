"""Event-study chart: alert scores before IL stints, relative to healthy games."""

from itertools import pairwise
from typing import Final, NamedTuple

import matplotlib as mpl
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from statecast.viz import theme

FIGURE_WIDTH_IN: Final = 10.0
FIGURE_HEIGHT_IN: Final = 5.8
LABEL_GAP_FRACTION: Final = 0.07
DETECTION_EDGE: Final = -1.5
GROUP_STYLES: Final[dict[str, tuple[str, str]]] = {
    "arm": (theme.SERIES_ORANGE, "Arm"),
    "trunk": (theme.SERIES_AQUA, "Trunk"),
    "lower_body": (theme.SERIES_BLUE, "Lower body"),
}
FOOTNOTE: Final = (
    "Alert score: real-time talent minus the pitcher's own level, from games "
    "already pitched, scaled to be N(0, 1) when nothing is wrong.\n"
    "Shown as the difference from healthy games (no IL stint within 60 days). "
    "Bands are 95% intervals; counts are IL placements."
)


class LineLabel(NamedTuple):
    """A direct label for the right end of a line.

    Attributes:
        height: Where the line ends, in data units.
        text: Label text.
        emphasized: Bold primary ink for injury groups, plain for the baseline.
    """

    height: float
    text: str
    emphasized: bool


def spread_labels(positions: list[float], gap: float) -> list[float]:
    """Nudge label heights apart so neighbours are at least ``gap`` apart.

    Labels keep their order; each one moves up only as far as needed.

    Args:
        positions: Desired label heights, in data units.
        gap: Minimum separation, in data units.

    Returns:
        Adjusted heights, in the input order.

    Example:
        >>> spread_labels([0.0, 0.02, -1.0], gap=0.1)
        [0.0, 0.1, -1.0]
    """
    placed = list(positions)
    order = np.argsort(positions, kind="stable")
    for below, above in pairwise(order):
        placed[above] = max(placed[above], placed[below] + gap)
    return placed


def _placements(line: pl.DataFrame) -> int:
    """Placements behind a group's line: the count at its final game."""
    return int(line.filter(pl.col("starts_before") == -1)["placements"].sum())


def _draw_group(axes: Axes, line: pl.DataFrame, color: str, label: str) -> LineLabel:
    """Band, line and dots for one injury group."""
    x = line["starts_before"].to_numpy()
    shift = line["shift"].to_numpy()
    axes.fill_between(
        x,
        line["lower_95"].to_numpy(),
        line["upper_95"].to_numpy(),
        color=color,
        alpha=0.13,
        linewidth=0,
    )
    axes.plot(x, shift, color=color, linewidth=2.0, zorder=3)
    axes.scatter(
        x,
        shift,
        s=30,
        color=color,
        edgecolors=theme.SURFACE,
        linewidths=1.5,
        zorder=4,
    )
    return LineLabel(float(shift[-1]), f"{label} ({_placements(line):,})", True)


def _label_line_ends(axes: Axes, labels: list[LineLabel]) -> None:
    """Direct labels in the right margin, spread so they never overlap."""
    low, high = axes.get_ylim()
    heights = spread_labels(
        [label.height for label in labels], LABEL_GAP_FRACTION * (high - low)
    )
    right = axes.get_xlim()[1]
    for label, height in zip(labels, heights, strict=True):
        axes.annotate(
            label.text,
            (right, height),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            fontweight="bold" if label.emphasized else "normal",
            color=theme.INK_PRIMARY if label.emphasized else theme.INK_SECONDARY,
            annotation_clip=False,
        )


def _mark_phases(axes: Axes, first: int) -> None:
    """Shade the final game and name the warning and detection phases."""
    axes.axvspan(DETECTION_EDGE, -0.6, color=theme.GRIDLINE, alpha=0.5, linewidth=0)
    for x, text in (((first + DETECTION_EDGE) / 2, "Warning games"), (-1, "Final")):
        axes.annotate(
            text,
            (x, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=theme.INK_SECONDARY,
        )


def _style_axes(axes: Axes, first: int) -> None:
    """Ticks counting back to the final game, labels and gridlines."""
    axes.set_xticks(range(first, 0))
    axes.set_xticklabels(["Final" if k == -1 else str(k) for k in range(first, 0)])
    axes.set_xlim(first - 0.4, -0.6)
    axes.set_xlabel("Appearances before the IL placement")
    axes.set_ylabel("Alert score minus healthy games (sd)")
    axes.grid(axis="y")
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)


def plot_event_study(relative: pl.DataFrame, title: str, subtitle: str) -> Figure:
    """Draw each injury group's alert score relative to healthy games.

    Args:
        relative: Output of ``relative_to_healthy``.
        title: Headline.
        subtitle: Which run and seasons.

    Returns:
        The figure.

    Raises:
        ValueError: If no charted injury group has any appearances.
    """
    charted = relative.filter(pl.col("group").is_in(list(GROUP_STYLES)))
    if charted.is_empty():
        raise ValueError("Event study has no appearances before a placement")
    first = int(charted["starts_before"].min())  # type: ignore[arg-type]
    with mpl.rc_context(theme.RC_PARAMS):
        figure = Figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=200)
        axes = figure.subplots()
        axes.axhline(
            0.0, color=theme.INK_MUTED, linewidth=1.25, linestyle=(0, (4, 2)), zorder=2
        )
        labels = [LineLabel(0.0, "Healthy games", False)]
        for group, (color, label) in GROUP_STYLES.items():
            line = charted.filter(pl.col("group") == group).sort("starts_before")
            if not line.is_empty():
                labels.append(_draw_group(axes, line, color, label))
        _style_axes(axes, first)
        _mark_phases(axes, first)
        _label_line_ends(axes, labels)
        figure.text(
            0.015,
            0.965,
            title,
            fontsize=15,
            fontweight="bold",
            color=theme.INK_PRIMARY,
            va="top",
        )
        figure.text(
            0.015, 0.9, subtitle, fontsize=10.5, color=theme.INK_SECONDARY, va="top"
        )
        figure.text(
            0.015,
            0.018,
            FOOTNOTE,
            fontsize=8.5,
            color=theme.INK_MUTED,
            va="bottom",
            linespacing=1.4,
        )
        figure.subplots_adjust(left=0.08, right=0.8, top=0.78, bottom=0.21)
    return figure
