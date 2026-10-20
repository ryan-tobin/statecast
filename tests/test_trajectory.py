from datetime import date

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from statecast.evaluation.simulate import Schedule, simulate_population
from statecast.models.kalman import (
    PopulationParams,
    adjusted_means,
    batch_from_frame,
    kalman_filter,
    rts_smoother,
)
from statecast.models.seasonal import seasonal_effect
from statecast.models.trajectory import (
    gap_scores,
    smooth_population,
    trajectory_frame,
)

PARAMS = PopulationParams(
    mean_reversion=jnp.asarray(0.01),
    diffusion=jnp.asarray(0.1),
    observation_sd=jnp.asarray(0.9),
    population_mean=jnp.asarray(94.5),
    population_sd=jnp.asarray(2.0),
    day_effect_sd=jnp.asarray(0.5),
    seasonal_knots=jnp.array([-0.3, -0.2, -0.1, 0.1, 0.05, 0.0]),
)
FRAME = pl.DataFrame(
    {
        "player_id": [9, 4, 4, 4, 9],
        "game_date": [
            date(2026, 4, 3),
            date(2026, 4, 10),
            date(2026, 4, 1),
            date(2026, 4, 5),
            date(2026, 4, 8),
        ],
        "sample_size": [20, 30, 25, 28, 22],
        "mean": [97.1, 93.0, 93.4, 93.2, 96.8],
        "within_ss": [15.0, 25.0, 20.0, 22.0, 18.0],
    }
)


def test_smooth_population_matches_one_player_at_a_time() -> None:
    batch, _ = batch_from_frame(FRAME)

    filtered, smoothed = smooth_population(PARAMS, batch)

    single = kalman_filter(
        PARAMS,
        batch.gap_days[1],
        adjusted_means(PARAMS, batch)[1],
        batch.sample_size[1],
        batch.within_ss[1],
        batch.mask[1],
    )
    np.testing.assert_allclose(filtered.filtered_means[1], single.filtered_means)
    np.testing.assert_allclose(smoothed.means[1], rts_smoother(single).means)


def test_trajectory_frame_aligns_rows_with_sorted_appearances() -> None:
    batch, _ = batch_from_frame(FRAME)
    filtered, smoothed = smooth_population(PARAMS, batch)

    paths = trajectory_frame(FRAME, batch, PARAMS, filtered, smoothed)

    assert paths.select("player_id", "game_date").rows() == [
        (4, date(2026, 4, 1)),
        (4, date(2026, 4, 5)),
        (4, date(2026, 4, 10)),
        (9, date(2026, 4, 3)),
        (9, date(2026, 4, 8)),
    ]
    np.testing.assert_allclose(paths["appearance_mean"], [93.4, 93.2, 93.0, 97.1, 96.8])
    np.testing.assert_allclose(
        paths["smoothed_mean"][3:], np.asarray(smoothed.means[1, :2, 0])
    )
    assert bool((paths["smoothed_sd"] <= paths["filtered_sd"] + 1e-12).all())
    expected_shift = seasonal_effect(
        PARAMS.seasonal_knots, jnp.array([31.0, 35.0, 40.0, 33.0, 38.0])
    )
    np.testing.assert_allclose(paths["seasonal_effect"], expected_shift)


def test_trajectory_frame_reports_real_time_gap_to_own_level() -> None:
    batch, _ = batch_from_frame(FRAME)
    filtered, smoothed = smooth_population(PARAMS, batch)

    paths = trajectory_frame(FRAME, batch, PARAMS, filtered, smoothed)

    means = np.asarray(filtered.filtered_means[0, :3])
    covs = np.asarray(filtered.filtered_covs[0, :3])
    gap_var = covs[:, 0, 0] + covs[:, 1, 1] - 2.0 * covs[:, 0, 1]
    prior_var = 0.1**2 / (2.0 * 0.01)
    gap_mean = means[:, 0] - means[:, 1]
    np.testing.assert_allclose(paths["filtered_gap_mean"][:3], gap_mean)
    np.testing.assert_allclose(paths["filtered_gap_sd"][:3], np.sqrt(gap_var))
    np.testing.assert_allclose(
        paths["alert_z"][:3], gap_mean / np.sqrt(prior_var - gap_var)
    )


def test_alert_score_is_standard_normal_when_talent_follows_the_model() -> None:
    days = 30.0 + 5.0 * jnp.arange(20)
    schedule = Schedule(jnp.concatenate([jnp.zeros(1), jnp.diff(days)]), days)
    batch, _ = simulate_population(jax.random.key(7), PARAMS, 20_000, schedule, 20)
    filtered, _ = smooth_population(PARAMS, batch)

    scores = np.asarray(gap_scores(PARAMS, filtered).alert_z)

    np.testing.assert_allclose(scores[:, [1, 9, 19]].mean(axis=0), 0.0, atol=0.03)
    np.testing.assert_allclose(scores[:, [1, 9, 19]].std(axis=0), 1.0, rtol=0.03)
