"""Per-player talent trajectories from a fitted population model."""

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from jax import Array

from statecast.models.kalman import (
    FilterResult,
    ObservationBatch,
    PopulationParams,
    SmoothedResult,
    adjusted_means,
    kalman_filter,
    rts_smoother,
)
from statecast.models.seasonal import seasonal_effect


def smooth_population(
    params: PopulationParams, batch: ObservationBatch
) -> tuple[FilterResult, SmoothedResult]:
    """Filter and smooth every player in a batch at once.

    The filter sees season-adjusted means, so talent is the July 1 equivalent.

    Args:
        params: Fitted population parameters.
        batch: Padded appearances, shape ``(players, max_appearances)``.

    Returns:
        Filter and smoother outputs with a leading player axis.
    """

    def one_player(
        gap: Array, mean: Array, size: Array, ss: Array, mask: Array
    ) -> tuple[FilterResult, SmoothedResult]:
        filtered = kalman_filter(params, gap, mean, size, ss, mask)
        return filtered, rts_smoother(filtered)

    result: tuple[FilterResult, SmoothedResult] = jax.vmap(one_player)(
        batch.gap_days,
        adjusted_means(params, batch),
        batch.sample_size,
        batch.within_ss,
        batch.mask,
    )
    return result


def trajectory_frame(
    appearances: pl.DataFrame,
    batch: ObservationBatch,
    seasonal_knots: Array,
    filtered: FilterResult,
    smoothed: SmoothedResult,
) -> pl.DataFrame:
    """Align filtered and smoothed talent with the appearances they came from.

    ``batch_from_frame`` orders players by id and appearances by date, and
    pads each row at its end, so the real cells of the batch read in row-major
    order line up with ``appearances`` sorted by player and date.

    Args:
        appearances: The frame the batch was built from.
        batch: The padded batch.
        seasonal_knots: Fitted seasonal knots, to record each game's shift.
        filtered: Filter output with a leading player axis.
        smoothed: Smoother output with a leading player axis.

    Returns:
        One row per appearance: ``player_id``, ``game_date``, ``sample_size``,
        the raw ``appearance_mean``, its ``seasonal_effect``, and the filtered
        (real-time) and smoothed (hindsight) talent mean and standard
        deviation, all season-adjusted to July 1.
    """
    mask = np.asarray(batch.mask)

    def talent(values: Array) -> np.ndarray:
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
            seasonal_effect=talent(seasonal_effect(seasonal_knots, batch.season_day)),
            filtered_mean=talent(filtered.filtered_means[..., 0]),
            filtered_sd=np.sqrt(talent(filtered.filtered_covs[..., 0, 0])),
            smoothed_mean=talent(smoothed.means[..., 0]),
            smoothed_sd=np.sqrt(talent(jnp.clip(smoothed.covs[..., 0, 0], 0.0))),
        )
    )
