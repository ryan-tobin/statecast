"""Forest chart: how well each alert score warns of IL placements."""

from collections.abc import Mapping
from typing import Final

import matplotlib as mpl
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from statecast.evaluation.injury_alerts import PHASES
from statecast.evaluation.score_comparison import ANY_GROUP, CONTRAST
from statecast.viz import theme

FIGURE_WIDTH_IN: Final = 10.0
FIGURE_HEIGHT_IN: Final = 6.4
X_MARGIN: Final = 1.25
MAX_TICKS: Final = 5
MIN_REACH: Final = 0.05
PHASE_TITLES: Final = {"warning": "Warning games", "detection": "Final game"}
ROWS: Final = (
    (ANY_GROUP, "AUC, any IL placement", 0.5),
    (CONTRAST, "AUC, arm minus lower body", 0.0),
)
FOOTNOTE: Final = (
    "Top: chance a game before a placement scores higher than a healthy game "
    "(0.5 is no signal). Bottom: arm-injury AUC minus lower-body AUC;\n"
    "above 0 means the score warns of arm injuries specifically. Lines: 95% "
    "pitcher-bootstrap intervals. The first score, in grey, is the baseline."
)


def _x_limits(rows: pl.DataFrame, center: float) -> tuple[float, float]:
    """Symmetric range around the null value covering every interval."""
    bounds = rows.select("estimate", "lower_95", "upper_95").to_numpy().ravel()
    finite = np.abs(bounds[np.isfinite(bounds)] - center)
    reach = max(float(finite.max()) * X_MARGIN if finite.size else 0.0, MIN_REACH)
    return center - reach, center + reach


def _draw_panel(
    axes: Axes, rows: pl.DataFrame, positions: np.ndarray, center: float
) -> None:
    """Intervals, points and value labels for one statistic and phase."""
    colors = [theme.INK_MUTED, *([theme.SERIES_BLUE] * (rows.height - 1))]
    axes.axvline(
        center, color=theme.INK_MUTED, linewidth=1.25, linestyle=(0, (4, 2)), zorder=1
    )
    axes.hlines(
        positions,
        rows["lower_95"].to_numpy(),
        rows["upper_95"].to_numpy(),
        colors=colors,
        linewidth=2.0,
    )
    axes.scatter(
        rows["estimate"].to_numpy(),
        positions,
        s=60,
        color=colors,
        edgecolors=theme.SURFACE,
        linewidths=1.5,
        zorder=3,
    )
    for y, value, high in zip(
        positions, rows["estimate"].to_numpy(), rows["upper_95"].to_numpy(), strict=True
    ):
        axes.annotate(
            f"{value:.3f}".replace("-", "\N{MINUS SIGN}"),
            (high, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=theme.INK_SECONDARY,
        )
    axes.xaxis.set_major_locator(MaxNLocator(nbins=MAX_TICKS, steps=[1, 2, 5, 10]))
    axes.grid(axis="x")
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)


def plot_score_comparison(
    aucs: pl.DataFrame, labels: Mapping[str, str], title: str, subtitle: str
) -> Figure:
    """Draw each score's any-injury AUC and arm-specific contrast by phase.

    Args:
        aucs: The ``aucs`` table of ``compare_scores``.
        labels: Display name for each score, in the order to draw them; the
            first is the baseline.
        title: Headline.
        subtitle: Pitch type, seasons and horizon.

    Returns:
        The figure, one row per statistic and one column per phase.

    Raises:
        ValueError: If a score in ``labels`` has no rows.
    """
    scores = list(labels)
    missing = set(scores) - set(aucs["score"].unique().to_list())
    if missing:
        raise ValueError(f"No AUC rows for {sorted(missing)}")
    positions = np.arange(len(scores))[::-1].astype(float)
    order = pl.DataFrame({"score": scores})
    with mpl.rc_context(theme.RC_PARAMS):
        figure = Figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=200)
        panels = figure.subplots(len(ROWS), len(PHASES), sharey=True, squeeze=False)
        for row, (statistic, axis_label, center) in enumerate(ROWS):
            statistic_rows = aucs.filter(pl.col("statistic") == statistic)
            limits = _x_limits(statistic_rows, center)
            for column, phase in enumerate(PHASES):
                axes = panels[row, column]
                rows = order.join(
                    statistic_rows.filter(pl.col("phase") == phase),
                    on="score",
                    how="left",
                    maintain_order="left",
                )
                _draw_panel(axes, rows, positions, center)
                axes.set_xlim(*limits)
                axes.set_xlabel(axis_label)
                if row == 0:
                    axes.set_title(PHASE_TITLES[phase], loc="left", fontsize=10)
            panels[row, 0].set_yticks(positions, [labels[score] for score in scores])
        panels[0, 0].set_ylim(positions.min() - 0.6, positions.max() + 0.6)
        figure.text(
            0.015,
            0.97,
            title,
            fontsize=15,
            fontweight="bold",
            color=theme.INK_PRIMARY,
            va="top",
        )
        figure.text(
            0.015, 0.915, subtitle, fontsize=10.5, color=theme.INK_SECONDARY, va="top"
        )
        figure.text(
            0.015,
            0.015,
            FOOTNOTE,
            fontsize=8.5,
            color=theme.INK_MUTED,
            va="bottom",
            linespacing=1.4,
        )
        figure.subplots_adjust(
            left=0.22, right=0.95, top=0.83, bottom=0.17, wspace=0.22, hspace=0.55
        )
    return figure
