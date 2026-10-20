from datetime import date, timedelta

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from statecast.evaluation.injury_alerts import (
    Direction,
    alert_score,
    auc_table,
    bootstrap_summary,
    current_alerts,
    event_study,
    group_contrast,
    label_appearances,
    phase_sample,
    relative_to_healthy,
    threshold_table,
    weighted_auc,
)
from statecast.evaluation.simulate import Schedule, simulate_population
from statecast.models.kalman import PopulationParams
from statecast.models.trajectory import smooth_population, trajectory_frame


def states(rows: list[tuple[int, date, float]]) -> pl.DataFrame:
    """Appearances with a chosen alert z, scored as a drop."""
    return pl.DataFrame(
        {
            "player_id": [row[0] for row in rows],
            "full_name": [f"Pitcher {row[0]}" for row in rows],
            "game_date": [row[1] for row in rows],
            "filtered_mean": [95.0] * len(rows),
            "filtered_gap_mean": [0.5 * row[2] for row in rows],
            "alert_z": [row[2] for row in rows],
        }
    ).with_columns(alert_score(Direction.DROP))


def placements(rows: list[tuple[int, date, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": [row[0] for row in rows],
            "il_start": [row[1] for row in rows],
            "injury_group": [row[2] for row in rows],
            "injury": [f"{row[2]} injury" for row in rows],
        }
    )


GAMES = states(
    [
        (1, date(2025, 5, 1), 0.2),
        (1, date(2025, 5, 6), -1.2),
        (1, date(2025, 5, 11), -2.6),
        (2, date(2025, 5, 2), 0.1),
        (2, date(2025, 5, 7), -0.4),
        (2, date(2025, 8, 20), -2.2),
        (3, date(2025, 5, 3), -2.1),
        (3, date(2025, 5, 8), 0.3),
    ]
)
STINTS = placements(
    [
        (1, date(2025, 5, 14), "arm"),
        (2, date(2025, 5, 20), "lower_body"),
        (2, date(2025, 5, 7), "arm"),
    ]
)


def test_alert_score_signs_each_direction() -> None:
    frame = pl.DataFrame({"alert_z": [-2.0, 0.5]})

    scores = {
        direction: frame.select(alert_score(direction))["score"].to_list()
        for direction in Direction
    }

    assert scores == {
        Direction.DROP: [2.0, -0.5],
        Direction.RISE: [-2.0, 0.5],
        Direction.EITHER: [2.0, 0.5],
    }


def test_bootstrap_summary_interval_and_two_sided_p_value() -> None:
    draws = np.concatenate([np.linspace(-0.1, 0.3, 95), [np.nan] * 5])

    low, high, p_value = bootstrap_summary(draws, 0.0)

    below = (np.sum(draws[:95] <= 0.0) + 1) / 96
    assert low == pytest.approx(np.percentile(draws[:95], 2.5))
    assert high == pytest.approx(np.percentile(draws[:95], 97.5))
    assert p_value == pytest.approx(2 * below)
    assert all(np.isnan(bootstrap_summary(np.array([np.nan]), 0.0)))


def test_label_appearances_finds_next_placement_and_final_game() -> None:
    labeled = label_appearances(GAMES, STINTS)

    rows = labeled.select("player_id", "days_to_il", "injury_group", "is_final")
    assert rows.rows() == [
        (1, 13, "arm", False),
        (1, 8, "arm", False),
        (1, 3, "arm", True),
        (2, 5, "arm", True),
        (2, 13, "lower_body", True),
        (2, None, None, False),
        (3, None, None, False),
        (3, None, None, False),
    ]


def test_phase_sample_keeps_one_phase_and_healthy_games() -> None:
    labeled = label_appearances(GAMES, STINTS)

    warning = phase_sample(labeled, 30, "arm", "warning")
    detection = phase_sample(labeled, 30, "arm", "detection")

    assert warning.filter(pl.col("positive"))["alert_z"].to_list() == [0.2, -1.2]
    assert detection.filter(pl.col("positive"))["alert_z"].to_list() == [-2.6, 0.1]
    assert warning.filter(~pl.col("positive"))["alert_z"].to_list() == [
        -2.2,
        -2.1,
        0.3,
    ]


def test_weighted_auc_counts_ties_as_half_and_uses_weights() -> None:
    scores = np.array([2.0, 1.0, 1.0, 0.0])
    positive = np.array([True, True, False, False])

    unweighted = weighted_auc(scores, positive, np.ones(4))
    doubled_tie = weighted_auc(scores, positive, np.array([1.0, 2.0, 1.0, 1.0]))

    assert unweighted == pytest.approx((2 + 1.5) / 4)
    assert doubled_tie == pytest.approx((2 + 2 * 1.5) / 6)


def test_auc_table_point_estimates_and_target_minus_placebo() -> None:
    labeled = label_appearances(GAMES, STINTS)

    table = auc_table(labeled, 30, reps=50)

    rows = {(row["group"], row["phase"]): row for row in table.iter_rows(named=True)}
    assert rows["arm", "warning"]["auc"] == pytest.approx(2 / 6)
    assert rows["arm", "warning"]["placements"] == 1
    assert rows["arm", "detection"]["auc"] == pytest.approx(4 / 6)
    assert rows["arm", "detection"]["placements"] == 2
    assert rows["lower_body", "detection"]["auc"] == pytest.approx(1 / 3)
    assert np.isnan(rows["lower_body", "warning"]["auc"])
    difference = rows["arm - lower_body", "detection"]
    assert difference["auc"] == pytest.approx(4 / 6 - 1 / 3)
    assert difference["placements"] is None
    assert 0.0 < difference["p_value"] <= 1.0


def test_threshold_table_scores_warning_games_against_healthy_ones() -> None:
    labeled = label_appearances(GAMES, STINTS)

    table = threshold_table(labeled, [1.0, 2.0], horizon_days=30)

    one, two = table.rows(named=True)
    assert one["alert_rate"] == pytest.approx(3 / 5)
    assert one["precision"] == pytest.approx(1 / 3)
    assert one["lift"] == pytest.approx((1 / 3) / (2 / 5))
    assert (one["events_caught"], one["events"]) == (1, 1)
    assert one["median_lead_days"] == 8.0
    assert two["alert_rate"] == pytest.approx(2 / 5)
    assert two["events_caught"] == 0


def test_event_study_relative_shift_and_contrast() -> None:
    labeled = label_appearances(GAMES, STINTS)
    study = event_study(labeled, max_starts=2)

    relative = relative_to_healthy(study)
    contrast = group_contrast(study, "arm", "lower_body")

    healthy_mean = (2.2 + 2.1 - 0.3) / 3
    arm = relative.filter(pl.col("group") == "arm").sort("starts_before")
    np.testing.assert_allclose(
        arm["shift"], [1.2 - healthy_mean, (2.6 - 0.1) / 2 - healthy_mean]
    )
    last = arm.row(-1, named=True)
    healthy_se = np.std([-2.2, -2.1, 0.3], ddof=1) / np.sqrt(3)
    arm_se = np.std([-2.6, 0.1], ddof=1) / np.sqrt(2)
    half_width = 1.959963984540054 * np.hypot(healthy_se, arm_se)
    assert last["upper_95"] - last["shift"] == pytest.approx(half_width)
    assert contrast["starts_before"].to_list() == [-1]
    assert contrast["difference"][0] == pytest.approx(1.25 - 0.4)


def test_relative_to_healthy_requires_a_baseline() -> None:
    study = event_study(label_appearances(GAMES, STINTS), max_starts=2)

    with pytest.raises(ValueError, match="no healthy baseline"):
        relative_to_healthy(study.filter(pl.col("group") != "healthy"))


def test_current_alerts_report_what_happened_next() -> None:
    alerts = current_alerts(
        GAMES, placements([(2, date(2025, 9, 1), "arm")]), threshold=2.0
    )

    assert alerts["full_name"].to_list() == ["Pitcher 1", "Pitcher 2"]
    assert alerts["alert_z"].to_list() == [-2.6, -2.2]
    assert alerts["next_il"].to_list() == [None, date(2025, 9, 1)]
    assert alerts["injury"].to_list() == [None, "arm injury"]


SIM_PARAMS = PopulationParams(
    mean_reversion=jnp.asarray(0.007),
    diffusion=jnp.asarray(0.09),
    observation_sd=jnp.asarray(0.9),
    population_mean=jnp.asarray(94.0),
    population_sd=jnp.asarray(2.0),
    day_effect_sd=jnp.asarray(0.5),
    seasonal_knots=jnp.zeros(6),
)
SIM_STARTS = 30
SIM_PLAYERS = 600
INJURED = 120
DROP_STARTS = 5
DROP_MPH = 1.5


def simulated_injuries() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Pitchers whose velocity fades over 5 starts before an arm IL stint.

    The first ``INJURED`` pitchers lose up to 1.5 mph before an arm placement;
    the next ``INJURED`` get a lower-body placement with no velocity change.
    """
    days = 30.0 + 5.0 * jnp.arange(SIM_STARTS)
    schedule = Schedule(jnp.concatenate([jnp.zeros(1), jnp.diff(days)]), days)
    batch, _ = simulate_population(
        jax.random.key(11), SIM_PARAMS, SIM_PLAYERS, schedule, 25
    )
    ramp = np.zeros((SIM_PLAYERS, SIM_STARTS))
    ramp[:INJURED, -DROP_STARTS:] = -DROP_MPH * np.linspace(0.2, 1.0, DROP_STARTS)
    batch = batch._replace(mean=batch.mean + jnp.asarray(ramp))
    filtered, smoothed = smooth_population(SIM_PARAMS, batch)
    first_game = date(2025, 3, 31)
    dates = [first_game + timedelta(days=5 * start) for start in range(SIM_STARTS)]
    appearances = pl.DataFrame(
        {
            "player_id": np.repeat(np.arange(SIM_PLAYERS), SIM_STARTS),
            "game_date": dates * SIM_PLAYERS,
            "sample_size": np.full(SIM_PLAYERS * SIM_STARTS, 25),
            "mean": np.asarray(batch.mean).ravel(),
            "within_ss": np.asarray(batch.within_ss).ravel(),
        }
    )
    paths = trajectory_frame(
        appearances, batch, SIM_PARAMS, filtered, smoothed
    ).with_columns(full_name=pl.lit("Simulated"))
    stint_start = dates[-1] + timedelta(days=3)
    stints = placements(
        [(player, stint_start, "arm") for player in range(INJURED)]
        + [
            (player, stint_start, "lower_body")
            for player in range(INJURED, 2 * INJURED)
        ]
    )
    return paths, stints


@pytest.fixture(scope="module")
def simulated_labels() -> pl.DataFrame:
    paths, stints = simulated_injuries()
    return label_appearances(paths.with_columns(alert_score(Direction.DROP)), stints)


def test_simulated_fade_raises_arm_scores_above_healthy_only(
    simulated_labels: pl.DataFrame,
) -> None:
    study = event_study(simulated_labels, max_starts=8)

    relative = relative_to_healthy(study)
    contrast = group_contrast(study, "arm", "lower_body")

    arm = relative.filter(pl.col("group") == "arm").sort("starts_before")
    placebo = relative.filter(pl.col("group") == "lower_body").sort("starts_before")
    assert float(arm["shift"][0]) < 0.3
    assert float(arm["lower_95"][-1]) > 1.0
    assert float(placebo["lower_95"][-1]) < 0.0 < float(placebo["upper_95"][-1])
    assert float(contrast["lower_95"][-1]) > 0.0


def test_simulated_auc_is_higher_at_detection_than_warning(
    simulated_labels: pl.DataFrame,
) -> None:
    table = auc_table(simulated_labels, 30, reps=200)

    rows = {(row["group"], row["phase"]): row for row in table.iter_rows(named=True)}
    warning = rows["arm", "warning"]["auc"]
    detection = rows["arm", "detection"]["auc"]
    assert detection > warning > 0.55
    assert abs(rows["lower_body", "warning"]["auc"] - 0.5) < 0.1
    assert rows["arm - lower_body", "warning"]["auc"] > 0.0
    assert rows["arm - lower_body", "detection"]["lower_95"] > 0.2


def test_simulated_alert_rate_matches_the_normal_tail(
    simulated_labels: pl.DataFrame,
) -> None:
    table = threshold_table(simulated_labels, [2.0], horizon_days=30)

    assert 0.015 < float(table["alert_rate"][0]) < 0.03
    assert float(table["lift"][0]) > 2.0
