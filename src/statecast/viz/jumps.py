"""Jump chart: real-time talent from the jump model, with jump probabilities."""

from dataclasses import replace
from itertools import pairwise
from typing import Final

import matplotlib as mpl
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, MonthLocator
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from statecast.viz import theme
from statecast.viz.trajectories import (
    MIN_PANEL_DAYS,
    Z_50,
    Z_90,
    ChartLabels,
    chart_labels,
    season_span,
)

FIGURE_WIDTH_IN: Final = 12.0
FIGURE_HEIGHT_IN: Final = 7.2
LIKELY_JUMP: Final = 0.5
BAR_WIDTH_DAYS: Final = 2.0
BREAK_GAP_DAYS: Final = 30
Y_PADDING_FRACTION: Final = 0.06
FOOTNOTE: Final = (
    "Talent and jump probabilities use only games played so far. A bar is the "
    "probability that talent jumped just before that game.\n"
    "Seasonal curve removed from all values; game averages also carry one-day "
    "noise the intervals exclude. Lines break across absences over "
    f"{BREAK_GAP_DAYS} days."
)


def jump_chart_labels(
    player_name: str, metric: str, pitch_name: str | None, seasons: list[int]
) -> ChartLabels:
    """Chart words for a jump run: the fan chart's, with its own subtitle.

    Args:
        player_name: Player's full name.
        metric: Metric column.
        pitch_name: Pitch name for pitcher runs, ``None`` for hitters.
        seasons: Seasons shown.

    Returns:
        Title, subtitle, unit and sample label.

    Example:
        >>> labels = jump_chart_labels("Zack Wheeler", "release_speed",
        ...                            "4-Seam Fastball", [2026])
        >>> labels.subtitle.split(",")[0]
        'Real-time talent from the jump model'
    """
    labels = chart_labels(player_name, metric, pitch_name, seasons)
    subtitle = (
        f"Real-time talent from the jump model, season-adjusted to July 1, "
        f"{season_span(seasons)} regular season ({labels.unit})"
    )
    return replace(labels, subtitle=subtitle)


def _adjusted_game_means(frame: pl.DataFrame) -> np.ndarray:
    """Game averages with the league seasonal shift removed."""
    adjusted: np.ndarray = (
        frame["appearance_mean"] - frame["seasonal_effect"]
    ).to_numpy()
    return adjusted


def _y_limits(trajectory: pl.DataFrame) -> tuple[float, float]:
    """Talent-panel range covering the 90% band and every adjusted game mean."""
    center = trajectory["talent_mean"].to_numpy()
    spread = trajectory["talent_sd"].to_numpy()
    means = _adjusted_game_means(trajectory)
    low = min(float((center - Z_90 * spread).min()), float(means.min()))
    high = max(float((center + Z_90 * spread).max()), float(means.max()))
    pad = (high - low) * Y_PADDING_FRACTION
    return low - pad, high + pad


def _panel_days(season: pl.DataFrame) -> int:
    """Days a season's games span, floored so short seasons stay legible."""
    dates = season["game_date"].to_numpy()
    span = int((dates.max() - dates.min()) // np.timedelta64(1, "D"))
    return max(span, MIN_PANEL_DAYS)


def segments(dates: np.ndarray) -> list[slice]:
    """Runs of games with no absence longer than ``BREAK_GAP_DAYS``.

    Args:
        dates: Sorted game dates (``datetime64``).

    Returns:
        One slice per run, in order.

    Example:
        >>> days = np.array(["2026-04-01", "2026-04-06", "2026-06-20"],
        ...                 dtype="datetime64[D]")
        >>> segments(days)
        [slice(0, 2, None), slice(2, 3, None)]
    """
    gaps = np.diff(dates) / np.timedelta64(1, "D")
    breaks = (np.flatnonzero(gaps > BREAK_GAP_DAYS) + 1).tolist()
    edges = [0, *breaks, len(dates)]
    return [slice(start, end) for start, end in pairwise(edges)]


def _draw_talent(axes: Axes, season: pl.DataFrame) -> None:
    """Bands, game dots, the talent line and likely jumps for one season."""
    dates = season["game_date"].to_numpy()
    center = season["talent_mean"].to_numpy()
    spread = season["talent_sd"].to_numpy()
    runs = segments(dates)
    for run in runs:
        for z_score, alpha in ((Z_90, 0.10), (Z_50, 0.20)):
            axes.fill_between(
                dates[run],
                center[run] - z_score * spread[run],
                center[run] + z_score * spread[run],
                color=theme.SERIES_BLUE,
                alpha=alpha,
                linewidth=0,
            )
    axes.scatter(
        dates,
        _adjusted_game_means(season),
        s=14,
        color=theme.INK_MUTED,
        alpha=0.55,
        edgecolors=theme.SURFACE,
        linewidths=0.8,
        zorder=2,
    )
    for run in runs:
        axes.plot(
            dates[run],
            center[run],
            color=theme.SERIES_BLUE,
            linewidth=2.0,
            marker="o" if run.stop - run.start == 1 else "",
            markersize=3,
            zorder=3,
        )
    likely = season["jump_prob"].to_numpy() >= LIKELY_JUMP
    axes.scatter(
        dates[likely],
        center[likely],
        s=46,
        color=theme.SERIES_ORANGE,
        edgecolors=theme.SURFACE,
        linewidths=1.5,
        zorder=4,
    )
    axes.set_title(str(season["game_date"][0].year), loc="left", fontsize=10)
    axes.grid(axis="y")
    axes.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        axes.spines[side].set_visible(False)


def _draw_probabilities(axes: Axes, season: pl.DataFrame) -> None:
    """Jump probability bars for one season, likely jumps in orange."""
    dates = season["game_date"].to_numpy()
    probability = season["jump_prob"].to_numpy()
    colors = np.where(probability >= LIKELY_JUMP, theme.SERIES_ORANGE, theme.INK_MUTED)
    axes.bar(dates, probability, width=BAR_WIDTH_DAYS, color=colors, linewidth=0)
    axes.axhline(
        LIKELY_JUMP,
        color=theme.INK_MUTED,
        linewidth=1.0,
        linestyle=(0, (4, 2)),
        zorder=1,
    )
    axes.set_ylim(0.0, 1.0)
    axes.set_yticks([0.0, 0.5, 1.0])
    axes.xaxis.set_major_locator(MonthLocator())  # type: ignore[no-untyped-call]
    axes.xaxis.set_major_formatter(DateFormatter("%b"))  # type: ignore[no-untyped-call]
    axes.grid(axis="y")
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)


def _label_last_value(axes: Axes, trajectory: pl.DataFrame, unit: str) -> None:
    """End-dot and direct label for the latest real-time talent estimate."""
    last = trajectory.row(-1, named=True)
    axes.scatter(
        [last["game_date"]],
        [last["talent_mean"]],
        s=40,
        color=theme.SERIES_BLUE,
        edgecolors=theme.SURFACE,
        linewidths=2.0,
        zorder=5,
    )
    axes.annotate(
        f"{last['talent_mean']:.1f} {unit}",
        (last["game_date"], last["talent_mean"]),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=theme.INK_PRIMARY,
        annotation_clip=False,
    )


def _legend_handles() -> list[Line2D | Patch]:
    """Keys for every mark type on the chart."""
    return [
        Line2D([], [], color=theme.SERIES_BLUE, linewidth=2.0, label="Talent"),
        Patch(
            facecolor=theme.SERIES_BLUE,
            alpha=0.25,
            linewidth=0,
            label="50% and 90% intervals",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=5,
            markerfacecolor=theme.INK_MUTED,
            markeredgecolor=theme.SURFACE,
            alpha=0.7,
            label="Adjusted game average",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=7,
            markerfacecolor=theme.SERIES_ORANGE,
            markeredgecolor=theme.SURFACE,
            label=f"Likely jump (probability {LIKELY_JUMP:.0%} or more)",
        ),
    ]


def plot_jump_chart(trajectory: pl.DataFrame, labels: ChartLabels) -> Figure:
    """Draw a player's real-time talent and jump probabilities by season.

    The top row is the particle filter's real-time talent with 50% and 90%
    intervals over season-adjusted game averages; games that most likely
    followed a jump are marked in orange. The bottom row shows the jump
    probability for every game.

    Args:
        trajectory: Output of ``load_jump_trajectory`` for one player.
        labels: Chart words, usually from ``jump_chart_labels``.

    Returns:
        The figure; save it with ``save_figure``.

    Raises:
        ValueError: If the trajectory is empty.
    """
    if trajectory.is_empty():
        raise ValueError("Trajectory has no appearances to plot")
    seasons = trajectory.with_columns(
        season=pl.col("game_date").dt.year()
    ).partition_by("season", maintain_order=True, include_key=False)
    widths = [_panel_days(season) for season in seasons]
    with mpl.rc_context(theme.RC_PARAMS):
        figure = Figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=200)
        panels = figure.subplots(
            2,
            len(seasons),
            sharex="col",
            sharey="row",
            squeeze=False,
            width_ratios=widths,
            height_ratios=[3, 1],
        )
        for column, season in enumerate(seasons):
            _draw_talent(panels[0, column], season)
            _draw_probabilities(panels[1, column], season)
        panels[0, 0].set_ylim(*_y_limits(trajectory))
        panels[0, 0].set_ylabel(labels.unit)
        panels[1, 0].set_ylabel("Jump probability")
        _label_last_value(panels[0, -1], trajectory, labels.unit)
        figure.text(
            0.012,
            0.97,
            labels.title,
            fontsize=15,
            fontweight="bold",
            color=theme.INK_PRIMARY,
            va="top",
        )
        figure.text(
            0.012,
            0.922,
            labels.subtitle,
            fontsize=10.5,
            color=theme.INK_SECONDARY,
            va="top",
        )
        figure.legend(
            handles=_legend_handles(),
            loc="upper left",
            bbox_to_anchor=(0.006, 0.893),
            ncols=4,
            fontsize=9.5,
            labelcolor=theme.INK_SECONDARY,
            handlelength=2.2,
            columnspacing=1.8,
        )
        figure.text(
            0.012,
            0.015,
            FOOTNOTE,
            fontsize=8.5,
            color=theme.INK_MUTED,
            va="bottom",
            linespacing=1.4,
        )
        figure.subplots_adjust(
            left=0.07, right=0.93, top=0.8, bottom=0.12, wspace=0.07, hspace=0.12
        )
    return figure
