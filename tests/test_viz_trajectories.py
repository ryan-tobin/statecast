from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statecast.viz.seasonal import plot_seasonal_curve, seasonal_knots
from statecast.viz.trajectories import (
    chart_labels,
    figure_filename,
    plot_talent_fan,
    save_figure,
    season_span,
)


def two_season_trajectory() -> pl.DataFrame:
    dates = [
        date(season, 4, 1) + timedelta(days=5 * start)
        for season in (2025, 2026)
        for start in range(30)
    ]
    talent = 95.0 + 0.3 * np.sin(np.arange(len(dates)) / 6.0)
    return pl.DataFrame(
        {
            "game_date": dates,
            "sample_size": [25] * len(dates),
            "appearance_mean": talent + 0.5 * np.cos(np.arange(len(dates))),
            "seasonal_effect": [-0.1] * len(dates),
            "filtered_mean": talent + 0.1,
            "filtered_sd": [0.4] * len(dates),
            "smoothed_mean": talent,
            "smoothed_sd": [0.3] * len(dates),
        }
    )


def test_chart_labels_for_pitcher_and_hitter() -> None:
    pitcher = chart_labels(
        "Zack Wheeler", "release_speed", "4-Seam Fastball", [2024, 2026]
    )
    hitter = chart_labels("Bryce Harper", "bat_speed", None, [2026])

    assert pitcher.title == "Zack Wheeler, 4-seam fastball velocity"
    assert pitcher.subtitle == (
        "Estimated true talent, season-adjusted to July 1, "
        "2024-2026 regular season (mph)"
    )
    assert pitcher.sample_label == "pitches"
    assert hitter.title == "Bryce Harper, bat speed"
    assert "2026 regular season" in hitter.subtitle
    assert hitter.sample_label == "swings"


def test_figure_filename_is_ascii_and_descriptive() -> None:
    assert figure_filename("Cristopher Sánchez", "release_speed", "FF") == (
        "cristopher_sanchez_ff_release_speed.png"
    )
    assert figure_filename("Bryce Harper", "bat_speed", None) == (
        "bryce_harper_bat_speed.png"
    )


def test_plot_talent_fan_draws_one_panel_per_season(output_dir: Path) -> None:
    labels = chart_labels("Test Pitcher", "release_speed", "4-Seam Fastball", [2025])
    path = output_dir / "fan.png"

    figure = plot_talent_fan(two_season_trajectory(), labels)
    save_figure(figure, path)

    assert len(figure.axes) == 2
    assert [axes.get_title(loc="left") for axes in figure.axes] == ["2025", "2026"]
    assert path.stat().st_size > 10_000


def test_plot_talent_fan_rejects_empty_trajectory() -> None:
    labels = chart_labels("Test Pitcher", "release_speed", "4-Seam Fastball", [2025])

    with pytest.raises(ValueError, match="no appearances"):
        plot_talent_fan(two_season_trajectory().clear(), labels)


def fit_summary() -> pl.DataFrame:
    knots = ["apr", "may", "jun", "aug", "sep", "oct"]
    estimates = [-0.2, -0.1, -0.03, 0.04, 0.03, 0.0]
    return pl.DataFrame(
        {
            "parameter": ["mean_reversion", *(f"season_{name}" for name in knots)],
            "estimate": [0.01, *estimates],
            "lower_95": [0.009, *(value - 0.03 for value in estimates)],
            "upper_95": [0.011, *(value + 0.03 for value in estimates)],
        }
    )


def test_seasonal_knots_restore_the_july_anchor() -> None:
    knots = seasonal_knots(fit_summary())

    assert knots["knot"].to_list() == ["apr", "may", "jun", "jul", "aug", "sep", "oct"]
    assert knots["knot_date"][0] == date(2026, 4, 1)
    assert knots.row(3) == ("jul", date(2026, 7, 1), 0.0, 0.0, 0.0)


def test_plot_seasonal_curve_saves(output_dir: Path) -> None:
    path = output_dir / "seasonal.png"

    figure = plot_seasonal_curve(fit_summary(), "League seasonal curve", "mph")
    save_figure(figure, path)

    assert len(figure.axes) == 1
    assert path.stat().st_size > 10_000


def test_season_span() -> None:
    assert season_span([2026]) == "2026"
    assert season_span([2026, 2024]) == "2024-2026"
