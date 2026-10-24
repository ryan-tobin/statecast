"""Line the particle filter's output up with the appearances it came from."""

import numpy as np
import polars as pl
from jax import Array

from statecast.models.jumps import JumpModel
from statecast.models.kalman import ObservationBatch
from statecast.models.rbpf import ParticleFilterResult
from statecast.models.seasonal import seasonal_effect


def jump_trajectory_frame(
    appearances: pl.DataFrame,
    batch: ObservationBatch,
    model: JumpModel,
    result: ParticleFilterResult,
) -> pl.DataFrame:
    """One row per appearance with real-time talent and jump probabilities.

    ``batch_from_frame`` orders players by id and appearances by date, and
    pads each row at its end, so the real cells of the batch read in row-major
    order line up with ``appearances`` sorted by player and date.

    Args:
        appearances: The frame the batch was built from.
        batch: The padded batch.
        model: Jump-diffusion parameters the filter ran with.
        result: Output of ``filter_jump_population``.

    Returns:
        ``player_id``, ``game_date``, ``sample_size``, the raw
        ``appearance_mean`` and its ``seasonal_effect``, the real-time
        ``talent_mean`` and ``talent_sd`` (season-adjusted to July 1), the
        ``prior_jump_prob`` from the gap before the game alone, the posterior
        ``jump_prob``, the ``jump_evidence`` (log Bayes factor),
        ``recent_jump_prob``, ``recent_drop_prob`` and the filter's
        ``effective_size``.
    """
    mask = np.asarray(batch.mask)
    shift = seasonal_effect(model.population.seasonal_knots, batch.season_day)
    rate = float(model.jumps.rate)
    prior = -np.expm1(-rate * np.asarray(batch.gap_days))

    def real(values: Array) -> np.ndarray:
        real_cells: np.ndarray = np.asarray(values)[mask]
        return real_cells

    return (
        appearances.sort("player_id", "game_date")
        .select(
            "player_id",
            "game_date",
            pl.col("sample_size").cast(pl.Int32),
            pl.col("mean").alias("appearance_mean"),
        )
        .with_columns(
            seasonal_effect=real(shift),
            talent_mean=real(result.talent_mean),
            talent_sd=real(result.talent_sd),
            prior_jump_prob=prior[mask],
            jump_prob=real(result.jump_prob),
            jump_evidence=real(result.jump_evidence),
            recent_jump_prob=real(result.recent_jump_prob),
            recent_drop_prob=real(result.recent_drop_prob),
            effective_size=real(result.effective_size),
        )
    )
