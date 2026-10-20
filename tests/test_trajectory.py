from datetime import date

import jax.numpy as jnp
import numpy as np
import polars as pl

from statecast.models.kalman import (
    PopulationParams,
    adjusted_means,
    batch_from_frame,
    kalman_filter,
    rts_smoother,
)
from statecast.models.seasonal import seasonal_effect
from statecast.models.trajectory import smooth_population, trajectory_frame

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

    paths = trajectory_frame(FRAME, batch, PARAMS.seasonal_knots, filtered, smoothed)

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
