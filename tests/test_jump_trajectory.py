from datetime import date

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from statecast.models.jump_trajectory import jump_trajectory_frame
from statecast.models.jumps import JumpModel, JumpParams
from statecast.models.kalman import PopulationParams, batch_from_frame
from statecast.models.rbpf import FilterSettings, filter_jump_population
from statecast.models.seasonal import season_day, seasonal_effect

MODEL = JumpModel(
    PopulationParams(
        mean_reversion=jnp.asarray(0.01),
        diffusion=jnp.asarray(0.1),
        observation_sd=jnp.asarray(0.9),
        population_mean=jnp.asarray(94.5),
        population_sd=jnp.asarray(2.0),
        day_effect_sd=jnp.asarray(0.5),
        seasonal_knots=jnp.array([-0.3, -0.2, -0.1, 0.1, 0.05, 0.0]),
    ),
    JumpParams(rate=jnp.asarray(0.01), mean=jnp.asarray(-1.0), sd=jnp.asarray(1.5)),
)


def appearances() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": [9, 7, 7, 7, 9],
            "game_date": [
                date(2026, 4, 3),
                date(2026, 4, 1),
                date(2026, 4, 6),
                date(2026, 4, 11),
                date(2026, 4, 9),
            ],
            "sample_size": [20, 90, 85, 88, 22],
            "mean": [92.0, 95.0, 94.8, 92.5, 92.3],
            "within_ss": [15.0, 70.0, 66.0, 71.0, 16.0],
        }
    )


def test_jump_trajectory_frame_lines_up_with_sorted_appearances() -> None:
    frame = appearances()
    batch, _ = batch_from_frame(frame)
    result = filter_jump_population(
        MODEL, batch, jax.random.key(0), FilterSettings(32, 2)
    )

    states = jump_trajectory_frame(frame, batch, MODEL, result)

    assert states["player_id"].to_list() == [7, 7, 7, 9, 9]
    assert states["appearance_mean"].to_list() == [95.0, 94.8, 92.5, 92.0, 92.3]
    np.testing.assert_allclose(
        states["talent_mean"].to_numpy(),
        np.asarray(result.talent_mean)[np.asarray(batch.mask)],
    )
    expected_shift = seasonal_effect(
        MODEL.population.seasonal_knots,
        jnp.asarray(states.select(season_day(pl.col("game_date")))["game_date"]),
    )
    np.testing.assert_allclose(states["seasonal_effect"].to_numpy(), expected_shift)
    assert states.schema["sample_size"] == pl.Int32
    assert states.null_count().sum_horizontal().item() == 0
    assert states["jump_prob"].to_list()[0] == 0.0
