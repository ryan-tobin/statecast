"""Event-study chart: real-time alert scores in the games before an IL stint."""

from itertools import pairwise
from typing import Final, NamedTuple

import matplotlib as mpl
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from statecast.viz import theme

FIGURE_WIDTH_IN: Final = 10.0
FIGURE_HEIGHT_IN: Final = 5.6
LABEL_GAP_FRACTION: Final = 0.07
GROUP_STYLES: Final[dict[str, tuple[str, str]]] = {
    "arm": (theme.SERIES_ORANGE, "Arm injury"),
    "other": (theme.SERIES_BLUE, "Other injury"),
}
FOOTNOTE: Final = (
    "Alert score: real-time talent minus the pitcher's own level, from games "
    "already pitched, scaled to be N(0, 1) when nothing is wrong.\n"
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
    """Placements behind a group's line: the count at its last appearance."""
    return int(line.filter(pl.col("starts_before") == -1)["appearances"].sum())


def _draw_baseline(axes: Axes, healthy: pl.DataFrame) -> LineLabel | None:
    """Dashed line and band for appearances with no IL stint ahead."""
    if healthy.is_empty():
        return None
    baseline = healthy.row(0, named=True)
    axes.axhspan(
        baseline["lower_95"],
        baseline["upper_95"],
        color=theme.INK_MUTED,
        alpha=0.15,
        linewidth=0,
    )
    axes.axhline(
        baseline["mean_z"],
        color=theme.INK_MUTED,
        linewidth=1.25,
        linestyle=(0, (4, 2)),
        zorder=2,
    )
    return LineLabel(float(baseline["mean_z"]), "No IL stint within 60 days", False)


def _draw_group(axes: Axes, line: pl.DataFrame, color: str, label: str) -> LineLabel:
    """Band, line and dots for one injury group."""
    x = line["starts_before"].to_numpy()
    mean = line["mean_z"].to_numpy()
    axes.fill_between(
        x,
        line["lower_95"].to_numpy(),
        line["upper_95"].to_numpy(),
        color=color,
        alpha=0.15,
        linewidth=0,
    )
    axes.plot(x, mean, color=color, linewidth=2.0, zorder=3)
    axes.scatter(
        x,
        mean,
        s=30,
        color=color,
        edgecolors=theme.SURFACE,
        linewidths=1.5,
        zorder=4,
    )
    return LineLabel(float(mean[-1]), f"{label} ({_placements(line):,})", True)


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


def _style_axes(axes: Axes, first: int) -> None:
    """Ticks counting back to the last game, labels and gridlines."""
    axes.set_xticks(range(first, 0))
    axes.set_xticklabels(["Last game" if k == -1 else str(k) for k in range(first, 0)])
    axes.set_xlim(first - 0.4, -0.6)
    axes.set_xlabel("Appearances before the IL placement")
    axes.set_ylabel("Alert score (sd from own level)")
    axes.grid(axis="y")
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)


def plot_event_study(study: pl.DataFrame, title: str, subtitle: str) -> Figure:
    """Draw mean alert score by appearances before IL placement, with intervals.

    Args:
        study: Output of ``event_study``.
        title: Headline.
        subtitle: What the score is and which run it came from.

    Returns:
        The figure.

    Raises:
        ValueError: If the study has no pre-placement appearances.
    """
    groups = study.filter(pl.col("starts_before").is_not_null())
    if groups.is_empty():
        raise ValueError("Event study has no appearances before a placement")
    first = int(groups["starts_before"].min())  # type: ignore[arg-type]
    with mpl.rc_context(theme.RC_PARAMS):
        figure = Figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=200)
        axes = figure.subplots()
        axes.axhline(0.0, color=theme.AXIS_LINE, linewidth=1.0, zorder=1)
        labels: list[LineLabel] = []
        baseline = _draw_baseline(axes, study.filter(pl.col("group") == "healthy"))
        if baseline is not None:
            labels.append(baseline)
        for group, (color, label) in GROUP_STYLES.items():
            line = groups.filter(pl.col("group") == group).sort("starts_before")
            if not line.is_empty():
                labels.append(_draw_group(axes, line, color, label))
        _style_axes(axes, first)
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
        figure.subplots_adjust(left=0.08, right=0.76, top=0.8, bottom=0.2)
    return figure
