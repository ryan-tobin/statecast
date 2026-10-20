"""Talent fan chart: estimated true talent with credible bands over game data."""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
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

Z_50: Final = 0.6744897501960817
Z_90: Final = 1.6448536269514722
MIN_PANEL_DAYS: Final = 45
FIGURE_WIDTH_IN: Final = 12.0
FIGURE_HEIGHT_IN: Final = 5.6
DOT_AREA_BASE: Final = 8.0
DOT_AREA_PER_PITCH: Final = 0.9
DOT_AREA_MAX: Final = 90.0
Y_PADDING_FRACTION: Final = 0.06

METRIC_LABELS: Final[dict[str, tuple[str, str]]] = {
    "release_speed": ("velocity", "mph"),
    "spin_rate": ("spin rate", "rpm"),
    "release_extension": ("extension", "ft"),
    "arm_angle": ("arm angle", "degrees"),
    "bat_speed": ("bat speed", "mph"),
    "swing_length": ("swing length", "ft"),
    "attack_angle": ("attack angle", "degrees"),
    "swing_path_tilt": ("swing path tilt", "degrees"),
}


@dataclass(frozen=True, slots=True)
class ChartLabels:
    """Words on the chart.

    Attributes:
        title: Headline, such as ``Zack Wheeler, 4-seam fastball velocity``.
        subtitle: What is plotted and over which seasons.
        unit: Unit of the metric, such as ``mph``.
        sample_label: What one observation is, such as ``pitches``.
    """

    title: str
    subtitle: str
    unit: str
    sample_label: str


def chart_labels(
    player_name: str, metric: str, pitch_name: str | None, seasons: list[int]
) -> ChartLabels:
    """Build the chart's words from a run's metadata.

    Args:
        player_name: Player's full name.
        metric: Metric column.
        pitch_name: Pitch name for pitcher runs, ``None`` for hitters.
        seasons: Seasons shown.

    Returns:
        Title, subtitle, unit and sample label.

    Example:
        >>> chart_labels("Zack Wheeler", "release_speed", "4-Seam Fastball",
        ...              [2024, 2026]).title
        'Zack Wheeler, 4-seam fastball velocity'
    """
    metric_name, unit = METRIC_LABELS.get(metric, (metric.replace("_", " "), ""))
    subject = f"{pitch_name.lower()} {metric_name}" if pitch_name else metric_name
    span = season_span(seasons)
    return ChartLabels(
        title=f"{player_name}, {subject}",
        subtitle=(
            f"Estimated true talent, season-adjusted to July 1, "
            f"{span} regular season ({unit})"
        ),
        unit=unit,
        sample_label="pitches" if pitch_name else "swings",
    )


def season_span(seasons: list[int]) -> str:
    """Format a set of seasons as one year or a range.

    Args:
        seasons: Season years.

    Returns:
        ``2026`` for one season, ``2024-2026`` for several.

    Example:
        >>> season_span([2026, 2024, 2025])
        '2024-2026'
    """
    first, last = min(seasons), max(seasons)
    return str(first) if first == last else f"{first}-{last}"


def _panel_days(season: pl.DataFrame) -> int:
    """Days a season's games span, floored so short seasons stay legible."""
    dates = season["game_date"].to_numpy()
    span = int((dates.max() - dates.min()) // np.timedelta64(1, "D"))
    return max(span, MIN_PANEL_DAYS)


def _dot_areas(sample_size: np.ndarray) -> np.ndarray:
    """Marker areas that grow with the evidence in each game, capped."""
    return np.minimum(DOT_AREA_BASE + DOT_AREA_PER_PITCH * sample_size, DOT_AREA_MAX)


def _adjusted_game_means(frame: pl.DataFrame) -> np.ndarray:
    """Game averages with the league seasonal shift removed."""
    adjusted: np.ndarray = (
        frame["appearance_mean"] - frame["seasonal_effect"]
    ).to_numpy()
    return adjusted


def _y_limits(trajectory: pl.DataFrame) -> tuple[float, float]:
    """Shared y-range covering the 90% band and every adjusted game mean."""
    center = trajectory["smoothed_mean"].to_numpy()
    spread = trajectory["smoothed_sd"].to_numpy()
    means = _adjusted_game_means(trajectory)
    low = min(float((center - Z_90 * spread).min()), float(means.min()))
    high = max(float((center + Z_90 * spread).max()), float(means.max()))
    pad = (high - low) * Y_PADDING_FRACTION
    return low - pad, high + pad


def _draw_season(axes: Axes, season: pl.DataFrame) -> None:
    """Draw bands, talent lines and game dots for one season's panel."""
    dates = season["game_date"].to_numpy()
    center = season["smoothed_mean"].to_numpy()
    spread = season["smoothed_sd"].to_numpy()
    for z_score, alpha in ((Z_90, 0.10), (Z_50, 0.20)):
        axes.fill_between(
            dates,
            center - z_score * spread,
            center + z_score * spread,
            color=theme.SERIES_BLUE,
            alpha=alpha,
            linewidth=0,
        )
    axes.scatter(
        dates,
        _adjusted_game_means(season),
        s=_dot_areas(season["sample_size"].to_numpy()),
        color=theme.INK_MUTED,
        alpha=0.55,
        edgecolors=theme.SURFACE,
        linewidths=1.0,
        zorder=2,
    )
    axes.plot(
        dates,
        season["filtered_mean"].to_numpy(),
        color=theme.SERIES_ORANGE,
        linewidth=1.25,
        linestyle=(0, (4, 2)),
        zorder=3,
    )
    axes.plot(dates, center, color=theme.SERIES_BLUE, linewidth=2.0, zorder=4)
    axes.set_title(str(season["game_date"][0].year), loc="left", fontsize=10)
    axes.xaxis.set_major_locator(MonthLocator())  # type: ignore[no-untyped-call]
    axes.xaxis.set_major_formatter(DateFormatter("%b"))  # type: ignore[no-untyped-call]
    axes.grid(axis="y")
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)


def _label_last_value(axes: Axes, trajectory: pl.DataFrame, unit: str) -> None:
    """End-dot and direct label for the latest hindsight talent estimate."""
    last = trajectory.row(-1, named=True)
    axes.scatter(
        [last["game_date"]],
        [last["smoothed_mean"]],
        s=40,
        color=theme.SERIES_BLUE,
        edgecolors=theme.SURFACE,
        linewidths=2.0,
        zorder=5,
    )
    axes.annotate(
        f"{last['smoothed_mean']:.1f} {unit}",
        (last["game_date"], last["smoothed_mean"]),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=theme.INK_PRIMARY,
        annotation_clip=False,
    )


def _legend_handles(sample_label: str) -> list[Line2D | Patch]:
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
            color=theme.SERIES_ORANGE,
            linewidth=1.25,
            linestyle=(0, (4, 2)),
            label="Real-time estimate",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=6,
            markerfacecolor=theme.INK_MUTED,
            markeredgecolor=theme.SURFACE,
            alpha=0.7,
            label=f"Adjusted game average (size = {sample_label})",
        ),
    ]


def plot_talent_fan(trajectory: pl.DataFrame, labels: ChartLabels) -> Figure:
    """Draw a player's talent trajectory, one panel per season on a shared axis.

    Everything is season-adjusted to July 1. The blue line is the smoothed
    talent (every game used, past and future); the dashed orange line is the
    real-time filter estimate (only games played so far). Dots are single-game
    averages minus the league seasonal shift; their scatter also includes the
    one-game day effect that the talent bands exclude.

    Args:
        trajectory: Output of ``load_trajectory`` for one player.
        labels: Chart words.

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
            1, len(seasons), sharey=True, squeeze=False, width_ratios=widths
        )[0]
        for axes, season in zip(panels, seasons, strict=True):
            _draw_season(axes, season)
        panels[0].set_ylim(*_y_limits(trajectory))
        panels[0].set_ylabel(labels.unit)
        _label_last_value(panels[-1], trajectory, labels.unit)
        figure.text(
            0.012,
            0.965,
            labels.title,
            fontsize=15,
            fontweight="bold",
            color=theme.INK_PRIMARY,
            va="top",
        )
        figure.text(
            0.012,
            0.905,
            labels.subtitle,
            fontsize=10.5,
            color=theme.INK_SECONDARY,
            va="top",
        )
        figure.legend(
            handles=_legend_handles(labels.sample_label),
            loc="upper left",
            bbox_to_anchor=(0.006, 0.865),
            ncols=4,
            fontsize=9.5,
            labelcolor=theme.INK_SECONDARY,
            handlelength=2.2,
            columnspacing=1.8,
        )
        figure.text(
            0.012,
            0.02,
            "Hindsight talent uses every game; the real-time estimate uses only "
            "games played so far. The league seasonal curve is removed from all "
            "values; game averages also carry one-day noise the intervals exclude.",
            fontsize=8.5,
            color=theme.INK_MUTED,
            va="bottom",
        )
        figure.subplots_adjust(
            left=0.055, right=0.93, top=0.74, bottom=0.13, wspace=0.04
        )
    return figure


def save_figure(figure: Figure, path: Path) -> None:
    """Write a figure to disk, creating the folder if needed.

    Args:
        figure: Figure to save.
        path: Destination, such as ``outputs/figures/zack_wheeler.png``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)


def figure_filename(player_name: str, metric: str, pitch_type: str | None) -> str:
    """Build a filesystem-safe PNG name for a player's chart.

    Args:
        player_name: Player's full name (accents are transliterated).
        metric: Metric column.
        pitch_type: Pitch-type code, or ``None`` for hitters.

    Returns:
        A name such as ``cristopher_sanchez_ff_release_speed.png``.

    Example:
        >>> figure_filename("Cristopher Sánchez", "release_speed", "FF")
        'cristopher_sanchez_ff_release_speed.png'
    """
    ascii_name = (
        unicodedata.normalize("NFKD", player_name).encode("ascii", "ignore").decode()
    )
    parts = [ascii_name, pitch_type or "", metric]
    stem = "_".join(
        re.sub(r"[^a-z0-9]+", "_", part.lower()).strip("_") for part in parts if part
    )
    return f"{stem}.png"
