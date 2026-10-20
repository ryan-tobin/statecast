"""Forest chart: does any metric warn of arm injuries more than the placebo?"""

from typing import Final

import matplotlib as mpl
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from statecast.evaluation.injury_alerts import PHASES
from statecast.viz import theme
from statecast.viz.trajectories import METRIC_LABELS

FIGURE_WIDTH_IN: Final = 10.0
BASE_HEIGHT_IN: Final = 2.4
ROW_HEIGHT_IN: Final = 0.5
X_MARGIN: Final = 1.35
DEFAULT_REACH: Final = 0.1
SIGNIFICANCE: Final = 0.05
PHASE_TITLES: Final = {"warning": "Warning games", "detection": "Final game"}
FOOTNOTE: Final = (
    "Point: arm-injury AUC minus lower-body AUC, each against healthy games; "
    "above 0 means the metric warns of arm injuries specifically.\n"
    "Lines: 95% pitcher-bootstrap intervals. Filled: Holm-adjusted p < 0.05 "
    "across every metric and phase in the scan."
)


def metric_label(metric: str, direction: str) -> str:
    """Readable row label for a metric and its scored direction.

    Args:
        metric: Metric column.
        direction: ``drop``, ``rise`` or ``either``.

    Returns:
        A label such as ``Velocity (drop)``.

    Example:
        >>> metric_label("arm_angle", "either")
        'Arm angle (either)'
    """
    name = METRIC_LABELS.get(metric, (metric.replace("_", " "), ""))[0]
    return f"{name.capitalize()} ({direction})"


def _draw_phase(axes: Axes, rows: pl.DataFrame, positions: np.ndarray) -> None:
    """Intervals, points and value labels for one phase's panel."""
    difference = rows["difference"].to_numpy()
    lower = rows["lower_95"].to_numpy()
    upper = rows["upper_95"].to_numpy()
    significant = rows["p_holm"].fill_null(1.0).to_numpy() < SIGNIFICANCE
    axes.axvline(
        0.0, color=theme.INK_MUTED, linewidth=1.25, linestyle=(0, (4, 2)), zorder=1
    )
    axes.hlines(positions, lower, upper, color=theme.SERIES_BLUE, linewidth=2.0)
    axes.scatter(
        difference,
        positions,
        s=60,
        color=np.where(significant, theme.SERIES_BLUE, theme.SURFACE),
        edgecolors=theme.SERIES_BLUE,
        linewidths=2.0,
        zorder=3,
    )
    for y, value, high in zip(positions, difference, upper, strict=True):
        axes.annotate(
            f"{value:+.3f}".replace("-", "\N{MINUS SIGN}"),
            (high, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=theme.INK_SECONDARY,
        )
    axes.grid(axis="x")
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)


def _x_limits(scan: pl.DataFrame) -> tuple[float, float]:
    """Symmetric x-range around zero covering every finite interval."""
    bounds = scan.select("difference", "lower_95", "upper_95").to_numpy().ravel()
    finite = np.abs(bounds[np.isfinite(bounds)])
    reach = float(finite.max()) * X_MARGIN if finite.size else DEFAULT_REACH
    reach = reach or DEFAULT_REACH
    return -reach, reach


def plot_metric_scan(scan: pl.DataFrame, title: str, subtitle: str) -> Figure:
    """Draw each metric's arm-minus-placebo AUC difference by phase.

    Args:
        scan: Output of ``metric_scan``.
        title: Headline.
        subtitle: Pitch type, seasons and horizon.

    Returns:
        The figure, one panel per phase and one row per metric.

    Raises:
        ValueError: If the scan is empty.
    """
    if scan.is_empty():
        raise ValueError("Scan has no metrics to plot")
    metrics = scan.select("metric", "direction").unique(maintain_order=True)
    labels = [metric_label(*row) for row in metrics.iter_rows()]
    positions = np.arange(len(labels))[::-1].astype(float)
    height = BASE_HEIGHT_IN + ROW_HEIGHT_IN * len(labels)
    with mpl.rc_context(theme.RC_PARAMS):
        figure = Figure(figsize=(FIGURE_WIDTH_IN, height), dpi=200)
        panels = figure.subplots(1, len(PHASES), sharex=True, sharey=True)
        for axes, phase in zip(panels, PHASES, strict=True):
            rows = metrics.join(
                scan.filter(pl.col("phase") == phase),
                on=["metric", "direction"],
                how="left",
                maintain_order="left",
            )
            _draw_phase(axes, rows, positions)
            axes.set_title(PHASE_TITLES[phase], loc="left", fontsize=10)
            axes.set_xlabel("AUC, arm minus lower body")
        panels[0].set_xlim(*_x_limits(scan))
        panels[0].set_yticks(positions, labels)
        panels[0].set_ylim(positions.min() - 0.6, positions.max() + 0.6)
        figure.text(
            0.015,
            1 - 0.3 / height,
            title,
            fontsize=15,
            fontweight="bold",
            color=theme.INK_PRIMARY,
            va="top",
        )
        figure.text(
            0.015,
            1 - 0.62 / height,
            subtitle,
            fontsize=10.5,
            color=theme.INK_SECONDARY,
            va="top",
        )
        figure.text(
            0.015,
            0.12 / height,
            FOOTNOTE,
            fontsize=8.5,
            color=theme.INK_MUTED,
            va="bottom",
            linespacing=1.4,
        )
        figure.subplots_adjust(
            left=0.2,
            right=0.97,
            top=1 - 1.15 / height,
            bottom=1.05 / height,
            wspace=0.12,
        )
    return figure
