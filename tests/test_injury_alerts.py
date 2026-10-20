from datetime import date, timedelta

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from statecast.evaluation.injury_alerts import (
    current_alerts,
    event_study,
    label_appearances,
    roc_auc,
    threshold_table,
)
from statecast.evaluation.simulate import Schedule, simulate_population
from statecast.models.kalman import PopulationParams
from statecast.models.trajectory import smooth_population, trajectory_frame


def states(rows: list[tuple[int, date, float]]) -> pl.DataFrame:
    """Appearances with a chosen alert score."""
    return pl.DataFrame(
        {
            "player_id": [row[0] for row in rows],
            "full_name": [f"Pitcher {row[0]}" for row in rows],
            "game_date": [row[1] for row in rows],
            "filtered_mean": [95.0] * len(rows),
            "filtered_gap_mean": [0.5 * row[2] for row in rows],
            "alert_z": [row[2] for row in rows],
        }
    )


def placements(rows: list[tuple[int, date, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": [row[0] for row in rows],
            "il_start": [row[1] for row in rows],
            "injury_group": [row[2] for row in rows],
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
        (2, date(2025, 5, 20), "other"),
        (2, date(2025, 5, 7), "arm"),
    ]
)


def test_label_appearances_finds_the_next_placement_after_each_game() -> None:
    labeled = label_appearances(GAMES, STINTS)

    rows = labeled.select("player_id", "days_to_il", "injury_group").rows()
    assert rows == [
        (1, 13, "arm"),
        (1, 8, "arm"),
        (1, 3, "arm"),
        (2, 5, "arm"),
        (2, 13, "other"),
        (2, None, None),
        (3, None, None),
        (3, None, None),
    ]


def test_event_study_numbers_games_back_from_the_placement() -> None:
    labeled = label_appearances(GAMES, STINTS)

    study = event_study(labeled, max_starts=2)

    arm = study.filter(pl.col("group") == "arm")
    assert arm["starts_before"].to_list() == [-2, -1]
    np.testing.assert_allclose(arm["mean_z"], [-1.2, (-2.6 + 0.1) / 2])
    healthy = study.filter(pl.col("group") == "healthy").row(0, named=True)
    assert healthy["starts_before"] is None
    assert healthy["appearances"] == 3
    np.testing.assert_allclose(healthy["mean_z"], (-2.2 - 2.1 + 0.3) / 3)


def test_roc_auc_ranks_pre_injury_games_low() -> None:
    labeled = label_appearances(GAMES, STINTS)

    arm = roc_auc(labeled, 30, "arm")
    nothing = roc_auc(labeled, 30, "elbow-only")

    positives = [0.2, -1.2, -2.6, 0.1]
    negatives = [-0.4, -2.2, -2.1, 0.3]
    expected = np.mean([[p < n for n in negatives] for p in positives])
    np.testing.assert_allclose(arm, expected)
    assert np.isnan(nothing)


def test_threshold_table_counts_alerts_precision_and_lead_time() -> None:
    labeled = label_appearances(GAMES, STINTS)

    table = threshold_table(labeled, [1.0, 2.0], horizon_days=30)

    one, two = table.rows(named=True)
    assert one["alert_rate"] == pytest.approx(4 / 8)
    assert one["precision"] == pytest.approx(2 / 4)
    assert one["lift"] == pytest.approx((2 / 4) / (4 / 8))
    assert (one["events_caught"], one["events"]) == (1, 2)
    assert one["median_lead_days"] == 8.0
    assert two["alert_rate"] == pytest.approx(3 / 8)
    assert two["median_lead_days"] == 3.0


def test_current_alerts_use_each_pitchers_latest_game() -> None:
    alerts = current_alerts(GAMES, threshold=2.0)

    assert alerts["full_name"].to_list() == ["Pitcher 1", "Pitcher 2"]
    assert alerts["alert_z"].to_list() == [-2.6, -2.2]


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
    the next ``INJURED`` get an "other" placement with no velocity change.
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
        + [(player, stint_start, "other") for player in range(INJURED, 2 * INJURED)]
    )
    return paths, stints


@pytest.fixture(scope="module")
def simulated_labels() -> pl.DataFrame:
    paths, stints = simulated_injuries()
    return label_appearances(paths, stints)


def test_simulated_velocity_fade_warns_of_arm_injuries_only(
    simulated_labels: pl.DataFrame,
) -> None:
    labeled = simulated_labels

    study = event_study(labeled, max_starts=8)

    arm = study.filter(pl.col("group") == "arm").sort("starts_before")
    other = study.filter(pl.col("group") == "other").sort("starts_before")
    healthy = study.filter(pl.col("group") == "healthy")
    assert float(arm["mean_z"][0]) > -0.3
    assert float(arm["mean_z"][-1]) < -1.0
    assert bool((arm["mean_z"].diff().drop_nulls() < 0.05).all())
    assert abs(float(other["mean_z"][-1])) < 0.4
    assert abs(float(healthy["mean_z"][0])) < 0.1
    assert roc_auc(labeled, 30, "arm") > 0.6
    assert abs(roc_auc(labeled, 30, "other") - 0.5) < 0.1


def test_simulated_alert_rate_matches_the_normal_tail(
    simulated_labels: pl.DataFrame,
) -> None:
    table = threshold_table(simulated_labels, [2.0], horizon_days=30)

    assert 0.015 < float(table["alert_rate"][0]) < 0.03
    assert float(table["lift"][0]) > 2.0
