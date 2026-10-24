from pathlib import Path

import polars as pl
import pytest
from matplotlib.figure import Figure

from statecast.evaluation.score_comparison import (
    COMPARISON_SCHEMA,
    STATISTICS,
)
from statecast.viz.score_comparison import plot_score_comparison
from statecast.viz.trajectories import save_figure

LABELS = {
    "smooth": "Smooth alert (\N{MINUS SIGN}z)",
    "drop_prob": "Recent drop probability",
    "drop_evidence": "Recent drop evidence",
}


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


def aucs() -> pl.DataFrame:
    rows = []
    for offset, score in enumerate(LABELS):
        for statistic in STATISTICS:
            for phase in ("warning", "detection"):
                center = 0.0 if statistic == "arm - lower_body" else 0.55
                estimate = center + 0.02 * offset
                low, high = estimate - 0.04, estimate + 0.04
                rows.append((score, statistic, phase, estimate, low, high, 0.2))
    return pl.DataFrame(rows, orient="row", schema=COMPARISON_SCHEMA).with_columns(
        p_holm=pl.lit(0.5)
    )


def test_plot_score_comparison_draws_two_statistics_by_phase(
    output_dir: Path,
) -> None:
    figure = plot_score_comparison(aucs(), LABELS, "Title", "Subtitle")

    assert len(figure.axes) == 4
    top_left, bottom_left = figure.axes[0], figure.axes[2]
    assert top_left.get_title(loc="left") == "Warning games"
    ticks = {tick.get_text() for tick in top_left.get_yticklabels()}
    assert ticks == set(LABELS.values())
    low, high = top_left.get_xlim()
    assert low < 0.5 < high
    low, high = bottom_left.get_xlim()
    assert low < 0.0 < high
    assert_text_inside(figure)
    save_figure(figure, output_dir / "score_comparison.png")
    assert (output_dir / "score_comparison.png").exists()


def test_plot_score_comparison_rejects_unknown_scores() -> None:
    with pytest.raises(ValueError, match="No AUC rows"):
        plot_score_comparison(
            aucs(), {**LABELS, "mystery": "Mystery"}, "Title", "Subtitle"
        )
