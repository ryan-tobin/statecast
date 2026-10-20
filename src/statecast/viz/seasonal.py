"""League seasonal curve chart: the shared shift by time of year."""

from datetime import date, timedelta
from typing import Final

import matplotlib as mpl
import numpy as np
import polars as pl
from matplotlib.dates import DateFormatter
from matplotlib.figure import Figure

from statecast.models.seasonal import ANCHOR_INDEX, KNOT_DAYS, KNOT_NAMES
from statecast.viz import theme

REFERENCE_MARCH_FIRST: Final = date(2026, 3, 1)
FIGURE_WIDTH_IN: Final = 8.0
FIGURE_HEIGHT_IN: Final = 4.6


def seasonal_knots(summary: pl.DataFrame) -> pl.DataFrame:
    """Pull every knot's estimate and 95% interval out of a fit summary.

    The July 1 anchor is added back as an exact zero.

    Args:
        summary: Output of ``summarize_fit``.

    Returns:
        One row per knot, in calendar order: ``knot``, ``knot_date``,
        ``estimate``, ``lower_95``, ``upper_95``.
    """
    estimates = summary.filter(pl.col("parameter").str.starts_with("season_"))
    by_name = {
        row["parameter"].removeprefix("season_"): row
        for row in estimates.iter_rows(named=True)
    }
    rows = [
        (
            name,
            REFERENCE_MARCH_FIRST + timedelta(days=int(day)),
            *(
                (0.0, 0.0, 0.0)
                if index == ANCHOR_INDEX
                else (
                    by_name[name]["estimate"],
                    by_name[name]["lower_95"],
                    by_name[name]["upper_95"],
                )
            ),
        )
        for index, (name, day) in enumerate(
            zip(KNOT_NAMES, KNOT_DAYS.tolist(), strict=True)
        )
    ]
    return pl.DataFrame(
        rows,
        schema=["knot", "knot_date", "estimate", "lower_95", "upper_95"],
        orient="row",
    )


def plot_seasonal_curve(summary: pl.DataFrame, title: str, subtitle: str) -> Figure:
    """Draw the fitted league seasonal curve with 95% intervals at each knot.

    Args:
        summary: Output of ``summarize_fit``.
        title: Headline, such as ``League seasonal curve, 4-seam velocity``.
        subtitle: Units and seasons.

    Returns:
        The figure.
    """
    knots = seasonal_knots(summary)
    dates = knots["knot_date"].to_numpy()
    estimate = knots["estimate"].to_numpy()
    with mpl.rc_context(theme.RC_PARAMS):
        figure = Figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=200)
        axes = figure.subplots()
        axes.axhline(0.0, color=theme.AXIS_LINE, linewidth=1.0, zorder=1)
        axes.fill_between(
            dates,
            knots["lower_95"].to_numpy(),
            knots["upper_95"].to_numpy(),
            color=theme.SERIES_BLUE,
            alpha=0.12,
            linewidth=0,
        )
        axes.plot(dates, estimate, color=theme.SERIES_BLUE, linewidth=2.0, zorder=3)
        axes.scatter(
            dates,
            estimate,
            s=40,
            color=theme.SERIES_BLUE,
            edgecolors=theme.SURFACE,
            linewidths=2.0,
            zorder=4,
        )
        axes.annotate(
            "July 1 anchor",
            (dates[ANCHOR_INDEX], 0.0),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=theme.INK_SECONDARY,
        )
        axes.annotate(
            f"{estimate[0]:+.2f}",
            (dates[0], estimate[0]),
            xytext=(-10, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=theme.INK_PRIMARY,
        )
        axes.set_xticks(dates)
        axes.xaxis.set_major_formatter(DateFormatter("%b 1"))  # type: ignore[no-untyped-call]
        axes.set_xlim(
            dates[0] - np.timedelta64(12, "D"), dates[-1] + np.timedelta64(5, "D")
        )
        axes.grid(axis="y")
        axes.set_axisbelow(True)
        for side in ("top", "right", "left"):
            axes.spines[side].set_visible(False)
        figure.text(
            0.02,
            0.96,
            title,
            fontsize=14,
            fontweight="bold",
            color=theme.INK_PRIMARY,
            va="top",
        )
        figure.text(
            0.02, 0.885, subtitle, fontsize=10, color=theme.INK_SECONDARY, va="top"
        )
        figure.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.1)
    return figure
