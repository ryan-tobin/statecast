from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statecast.evaluation.injury_alerts import AUC_TABLE_SCHEMA, Direction
from statecast.evaluation.metric_scan import holm_adjust, metric_scan, scan_rows
from statecast.viz.metric_scan import metric_label, plot_metric_scan
from statecast.viz.trajectories import save_figure


def auc_rows(shift: float, p_warning: float, p_detection: float) -> pl.DataFrame:
    """An ``auc_table``-shaped frame with chosen arm and placebo AUCs."""
    rows = []
    for phase, p_value in (("warning", p_warning), ("detection", p_detection)):
        rows += [
            ("arm", phase, 100, 0.55 + shift, 0.5, 0.6, 0.01),
            ("lower_body", phase, 40, 0.55, 0.5, 0.6, 0.02),
            (
                "arm - lower_body",
                phase,
                None,
                shift,
                shift - 0.05,
                shift + 0.05,
                p_value,
            ),
        ]
    return pl.DataFrame(rows, orient="row", schema=AUC_TABLE_SCHEMA)


def test_holm_adjust_matches_the_step_down_rule() -> None:
    p_values = np.array([0.04, 0.001, 0.03, np.nan, 0.2])

    adjusted = holm_adjust(p_values)

    expected = [max(0.004, 0.09, 0.08), 0.004, max(0.004, 0.09), np.nan, 0.2]
    np.testing.assert_allclose(adjusted, expected)


def test_holm_adjust_caps_at_one() -> None:
    assert holm_adjust(np.array([0.6, 0.7])).tolist() == [1.0, 1.0]


def test_scan_rows_pull_target_placebo_and_difference() -> None:
    rows = scan_rows("spin_rate", Direction.DROP, auc_rows(0.04, 0.2, 0.01))

    assert rows["phase"].to_list() == ["warning", "detection"]
    first = rows.row(0, named=True)
    assert first["metric"] == "spin_rate"
    assert first["direction"] == "drop"
    assert first["arm_auc"] == pytest.approx(0.59)
    assert first["placebo_auc"] == pytest.approx(0.55)
    assert first["difference"] == pytest.approx(0.04)
    assert first["p_value"] == pytest.approx(0.2)


def test_metric_scan_adjusts_across_every_metric_and_phase() -> None:
    scan = metric_scan(
        {
            ("release_speed", Direction.DROP): auc_rows(-0.03, 0.3, 0.8),
            ("arm_angle", Direction.EITHER): auc_rows(0.07, 0.01, 0.02),
        }
    )

    assert scan["metric"].to_list() == ["release_speed"] * 2 + ["arm_angle"] * 2
    np.testing.assert_allclose(scan["p_holm"], [0.6, 0.8, 0.04, 0.06])


def test_metric_scan_of_nothing_is_empty() -> None:
    scan = metric_scan({})

    assert scan.is_empty()
    assert "p_holm" in scan.columns


def test_metric_label_reads_naturally() -> None:
    assert metric_label("release_speed", "drop") == "Velocity (drop)"
    assert metric_label("release_extension", "drop") == "Extension (drop)"


def test_plot_metric_scan_draws_one_row_per_metric(output_dir: Path) -> None:
    scan = metric_scan(
        {
            ("release_speed", Direction.DROP): auc_rows(-0.03, 0.3, 0.8),
            ("spin_rate", Direction.DROP): auc_rows(0.05, 0.001, 0.4),
            ("arm_angle", Direction.EITHER): auc_rows(0.07, 0.01, 0.02),
        }
    )
    path = output_dir / "scan.png"

    figure = plot_metric_scan(scan, "Which metrics warn of arm injuries?", "FF")
    save_figure(figure, path)

    ticks = figure.axes[0].get_yticklabels()
    top_to_bottom = [
        tick.get_text() for tick in sorted(ticks, key=lambda t: -t.get_position()[1])
    ]
    assert top_to_bottom == [
        "Velocity (drop)",
        "Spin rate (drop)",
        "Arm angle (either)",
    ]
    assert len(figure.axes) == 2
    assert path.stat().st_size > 10_000


def test_plot_metric_scan_rejects_an_empty_scan() -> None:
    with pytest.raises(ValueError, match="no metrics"):
        plot_metric_scan(metric_scan({}), "Title", "Subtitle")


def test_plot_metric_scan_survives_missing_intervals(output_dir: Path) -> None:
    rows = auc_rows(0.0, 0.5, 0.5).with_columns(
        lower_95=pl.lit(float("nan")), upper_95=pl.lit(float("nan"))
    )
    scan = metric_scan({("release_speed", Direction.DROP): rows})

    figure = plot_metric_scan(scan, "Title", "Subtitle")

    assert figure.axes[0].get_xlim() == (-0.1, 0.1)
