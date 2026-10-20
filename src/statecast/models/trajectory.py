"""Per-player talent trajectories from a fitted population model."""

from typing import NamedTuple

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

MIN_NULL_VARIANCE = 1e-12


class GapScores(NamedTuple):
    """Real-time gap between talent and the player's own long-run level.

    Attributes:
        mean: ``E[talent - level | games so far]`` (units).
        sd: ``sd[talent - level | games so far]`` (units).
        alert_z: ``mean`` scaled to be exactly ``N(0, 1)`` when the player's
            talent follows the model; strongly negative means unusually far
            below his own level.
    """

    mean: Array
    sd: Array
    alert_z: Array


def gap_scores(params: PopulationParams, filtered: FilterResult) -> GapScores:
    """Score how far each player's real-time talent sits below his own level.

    The gap ``g = talent - level`` is a stationary OU deviation with prior
    variance ``v = sigma^2 / (2 kappa)``. In a linear Gaussian model the
    posterior variance ``P = Var(g | y)`` does not depend on the data, so the
    posterior mean ``E[g | y]`` has marginal variance ``v - P``. Dividing by
    ``sqrt(v - P)`` makes the score exactly standard normal under the model,
    so thresholds read as tail probabilities. (Dividing by ``sqrt(P)`` instead
    would understate surprises: early in a career ``P`` is near ``v`` and the
    posterior mean barely moves.)

    Args:
        params: Population parameters.
        filtered: Filter output, any leading shape.

    Returns:
        Gap mean, gap sd and the calibrated alert score.
    """
    means = filtered.filtered_means
    covs = filtered.filtered_covs
    gap_mean = means[..., 0] - means[..., 1]
    gap_var = jnp.clip(covs[..., 0, 0] + covs[..., 1, 1] - 2.0 * covs[..., 0, 1], 0.0)
    prior_var = params.diffusion**2 / (2.0 * params.mean_reversion)
    null_sd = jnp.sqrt(jnp.clip(prior_var - gap_var, MIN_NULL_VARIANCE))
    return GapScores(mean=gap_mean, sd=jnp.sqrt(gap_var), alert_z=gap_mean / null_sd)


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
    params: PopulationParams,
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
        params: Fitted population parameters.
        filtered: Filter output with a leading player axis.
        smoothed: Smoother output with a leading player axis.

    Returns:
        One row per appearance: ``player_id``, ``game_date``, ``sample_size``,
        the raw ``appearance_mean``, its ``seasonal_effect``, the filtered
        (real-time) and smoothed (hindsight) talent mean and standard
        deviation, all season-adjusted to July 1, and the real-time gap to
        the player's own level (``filtered_gap_mean``, ``filtered_gap_sd``)
        with its calibrated ``alert_z``.
    """
    mask = np.asarray(batch.mask)
    gaps = gap_scores(params, filtered)
    shift = seasonal_effect(params.seasonal_knots, batch.season_day)

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
            filtered_mean=real(filtered.filtered_means[..., 0]),
            filtered_sd=np.sqrt(real(filtered.filtered_covs[..., 0, 0])),
            smoothed_mean=real(smoothed.means[..., 0]),
            smoothed_sd=np.sqrt(real(jnp.clip(smoothed.covs[..., 0, 0], 0.0))),
            filtered_gap_mean=real(gaps.mean),
            filtered_gap_sd=real(gaps.sd),
            alert_z=real(gaps.alert_z),
        )
    )
