from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from statecast.viz.jumps import (
    BREAK_GAP_DAYS,
    LIKELY_JUMP,
    jump_chart_labels,
    plot_jump_chart,
    segments,
)
from statecast.viz.trajectories import save_figure


def assert_text_inside(figure: Figure) -> None:
    """Every piece of text, including annotations, lies within the figure."""
    figure.draw_without_rendering()
    bounds = figure.bbox
    texts = [*figure.texts, *(text for axes in figure.axes for text in axes.texts)]
    for text in texts:
        extent = text.get_window_extent()
        label = text.get_text()
        assert extent.x0 >= bounds.x0, label
        assert extent.x1 <= bounds.x1, label
        assert extent.y0 >= bounds.y0, label
        assert extent.y1 <= bounds.y1, label


def two_season_jump_trajectory() -> pl.DataFrame:
    dates = [
        date(season, 4, 1) + timedelta(days=5 * start)
        for season in (2025, 2026)
        for start in range(30)
    ]
    count = len(dates)
    talent = np.where(np.arange(count) < 45, 95.0, 93.2)
    jump_prob = np.full(count, 0.02)
    jump_prob[45] = 0.9
    return pl.DataFrame(
        {
            "game_date": dates,
            "sample_size": [90] * count,
            "appearance_mean": talent - 0.2 + 0.4 * np.sin(np.arange(count)),
            "seasonal_effect": [-0.2] * count,
            "talent_mean": talent,
            "talent_sd": [0.35] * count,
            "prior_jump_prob": [0.007] * count,
            "jump_prob": jump_prob,
            "jump_evidence": np.log(jump_prob / 0.007),
            "recent_jump_prob": np.minimum(1.0, jump_prob * 1.1),
            "recent_drop_prob": jump_prob,
        }
    )


def test_plot_jump_chart_draws_talent_and_probability_rows(output_dir: Path) -> None:
    trajectory = two_season_jump_trajectory()
    labels = jump_chart_labels(
        "Zack Wheeler", "release_speed", "4-Seam Fastball", [2025, 2026]
    )

    figure = plot_jump_chart(trajectory, labels)

    assert len(figure.axes) == 4
    talent_2026, probability_2026 = figure.axes[1], figure.axes[3]
    assert talent_2026.get_title(loc="left") == "2026"
    assert probability_2026.get_ylim() == (0.0, 1.0)
    heights = [
        bar.get_height()
        for bar in probability_2026.patches
        if isinstance(bar, Rectangle)
    ]
    assert max(heights) == 0.9
    assert sum(height >= LIKELY_JUMP for height in heights) == 1
    assert_text_inside(figure)
    save_figure(figure, output_dir / "jump_chart.png")
    assert (output_dir / "jump_chart.png").exists()


def test_jump_chart_labels_keep_title_and_change_subtitle() -> None:
    labels = jump_chart_labels("Zack Wheeler", "spin_rate", "Sinker", [2026])

    assert labels.title == "Zack Wheeler, sinker spin rate"
    assert labels.subtitle.startswith("Real-time talent from the jump model")
    assert labels.subtitle.endswith("2026 regular season (rpm)")


def test_plot_jump_chart_rejects_empty_trajectory() -> None:
    labels = jump_chart_labels("Nobody", "release_speed", None, [2026])

    with pytest.raises(ValueError, match="no appearances"):
        plot_jump_chart(two_season_jump_trajectory().head(0), labels)


def test_segments_split_at_long_absences() -> None:
    days = np.array(
        ["2025-04-01", "2025-04-06", "2025-04-11", "2025-08-01", "2025-08-06"],
        dtype="datetime64[D]",
    )

    runs = segments(days)

    assert runs == [slice(0, 3), slice(3, 5)]
    assert segments(days[:3]) == [slice(0, 3)]
    assert BREAK_GAP_DAYS == 30


def test_plot_jump_chart_breaks_the_line_across_an_absence() -> None:
    trajectory = two_season_jump_trajectory()
    absent = (pl.col("game_date") > date(2025, 5, 15)) & (
        pl.col("game_date") < date(2025, 8, 1)
    )
    with_gap = trajectory.filter(~absent)
    labels = jump_chart_labels(
        "Zack Wheeler", "release_speed", "4-Seam Fastball", [2025, 2026]
    )

    figure = plot_jump_chart(with_gap, labels)

    talent_2025, talent_2026 = figure.axes[0], figure.axes[1]
    assert len(talent_2025.get_lines()) == 2
    assert len(talent_2026.get_lines()) == 1
