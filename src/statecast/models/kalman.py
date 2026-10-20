"""Kalman filter, RTS smoother and population likelihood for appearance data.

State per player: ``x = [talent, level]``. Talent follows an OU process that
reverts to the player's own long-run level; the level is a constant random
effect drawn from the population, so the filter integrates it out exactly.

Each appearance adds a one-day shock ``delta ~ N(0, tau^2)`` shared by every
pitch in it (rest, weather, adrenaline) that does not persist to the next
appearance, plus a league-wide seasonal shift ``g(day)`` shared by every
player. Each appearance is summarized by sufficient statistics of its ``n``
pitches (or swings): the mean ``ybar`` and the within-appearance sum of
squares ``S``, so ``ybar ~ N(talent + g(day), tau^2 + r^2 / n)``. The filter
works on season-adjusted means ``ybar - g(day)``, so talent is the July 1
equivalent.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from jax import Array

from statecast.models.dynamics import STATE_DIM, talent_transition
from statecast.models.seasonal import FREE_KNOT_COUNT, season_day, seasonal_effect

LOG_TWO_PI = float(np.log(2.0 * np.pi))
OBSERVATION_ROW = jnp.array([1.0, 0.0])


class PopulationParams(NamedTuple):
    """Parameters shared by every player in one population.

    Attributes:
        mean_reversion: ``kappa``, rate talent returns to its level (per day).
        diffusion: ``sigma``, talent volatility (units per root day).
        observation_sd: ``r``, pitch-to-pitch noise within a game (units).
        population_mean: ``m``, mean long-run level across players (units).
        population_sd: ``s``, spread of long-run levels across players (units).
        day_effect_sd: ``tau``, one-game shock shared by its pitches (units).
        seasonal_knots: League seasonal shift at each monthly knot except
            July 1 (anchored at zero), shape ``(6,)`` (units).
    """

    mean_reversion: Array
    diffusion: Array
    observation_sd: Array
    population_mean: Array
    population_sd: Array
    day_effect_sd: Array
    seasonal_knots: Array


class ObservationBatch(NamedTuple):
    """Padded appearance sequences for many players.

    All arrays have shape ``(players, max_appearances)``; padding has
    ``mask == False``, ``gap_days == 0`` and ``sample_size == 1``.

    Attributes:
        gap_days: Days since the player's previous appearance (0 for the first).
        mean: Mean of the metric within the appearance.
        sample_size: Pitches or swings in the appearance.
        within_ss: Sum of squared deviations from the appearance mean.
        mask: True for real appearances.
        season_day: Days since March 1 of the appearance's year.
    """

    gap_days: Array
    mean: Array
    sample_size: Array
    within_ss: Array
    mask: Array
    season_day: Array


class FilterResult(NamedTuple):
    """Kalman filter output for one player (leading axis = appearances).

    Attributes:
        filtered_means: ``E[x_k | y_1..k]``, shape ``(T, 2)``.
        filtered_covs: ``Cov[x_k | y_1..k]``, shape ``(T, 2, 2)``.
        predicted_means: ``E[x_k | y_1..k-1]``, shape ``(T, 2)``.
        predicted_covs: ``Cov[x_k | y_1..k-1]``, shape ``(T, 2, 2)``.
        transitions: ``Phi_k`` from ``x_k-1`` to ``x_k``, shape ``(T, 2, 2)``.
        log_likelihood: ``log p(y_1..T)`` summed over real appearances.
    """

    filtered_means: Array
    filtered_covs: Array
    predicted_means: Array
    predicted_covs: Array
    transitions: Array
    log_likelihood: Array


class SmoothedResult(NamedTuple):
    """RTS smoother output for one player.

    Attributes:
        means: ``E[x_k | y_1..T]``, shape ``(T, 2)``.
        covs: ``Cov[x_k | y_1..T]``, shape ``(T, 2, 2)``.
    """

    means: Array
    covs: Array


def constrain(raw: Array) -> PopulationParams:
    """Map an unconstrained vector of length 12 to population parameters.

    Positive parameters are stored as logs; the population mean and the
    seasonal knots are unbounded.

    Args:
        raw: ``[log kappa, log sigma, log r, m, log s, log tau, knots...]``.

    Returns:
        The natural-scale parameters.
    """
    return PopulationParams(
        mean_reversion=jnp.exp(raw[0]),
        diffusion=jnp.exp(raw[1]),
        observation_sd=jnp.exp(raw[2]),
        population_mean=raw[3],
        population_sd=jnp.exp(raw[4]),
        day_effect_sd=jnp.exp(raw[5]),
        seasonal_knots=raw[6 : 6 + FREE_KNOT_COUNT],
    )


def unconstrain(params: PopulationParams) -> Array:
    """Inverse of ``constrain``.

    Args:
        params: Natural-scale parameters.

    Returns:
        ``[log kappa, log sigma, log r, m, log s, log tau, knots...]``.
    """
    scalars = jnp.array(
        [
            jnp.log(params.mean_reversion),
            jnp.log(params.diffusion),
            jnp.log(params.observation_sd),
            params.population_mean,
            jnp.log(params.population_sd),
            jnp.log(params.day_effect_sd),
        ]
    )
    return jnp.concatenate([scalars, params.seasonal_knots])


def adjusted_means(params: PopulationParams, batch: ObservationBatch) -> Array:
    """Appearance means with the league seasonal shift removed.

    Args:
        params: Population parameters.
        batch: Appearances, any leading shape.

    Returns:
        ``mean - g(season_day)``, same shape as ``batch.mean``.
    """
    return batch.mean - seasonal_effect(params.seasonal_knots, batch.season_day)


def initial_state(params: PopulationParams) -> tuple[Array, Array]:
    """Prior on ``[talent, level]`` at a player's first appearance.

    The level is ``N(m, s^2)``; talent is the level plus the OU stationary
    deviation ``N(0, sigma^2 / (2 kappa))``.

    Args:
        params: Population parameters.

    Returns:
        Prior mean (shape ``(2,)``) and covariance (shape ``(2, 2)``).
    """
    level_var = params.population_sd**2
    stationary_var = params.diffusion**2 / (2.0 * params.mean_reversion)
    mean = jnp.full(STATE_DIM, params.population_mean)
    cov = jnp.array([[level_var + stationary_var, level_var], [level_var, level_var]])
    return mean, cov


def residual_log_likelihood(
    sample_size: Array, within_ss: Array, obs_var: Array
) -> Array:
    """Log-density of the within-appearance spread, given the per-pitch variance.

    For ``n`` iid ``N(theta, r^2)`` draws, the joint density factors into the
    density of the mean ``N(theta, r^2 / n)`` and this term, which does not
    depend on ``theta``.

    Args:
        sample_size: Pitches in the appearance, ``n``.
        within_ss: Within-appearance sum of squares, ``S``.
        obs_var: Per-pitch variance, ``r^2``.

    Returns:
        ``-(n-1)/2 log(2 pi r^2) - S / (2 r^2) - log(n) / 2``.
    """
    return (
        -0.5 * (sample_size - 1.0) * (LOG_TWO_PI + jnp.log(obs_var))
        - 0.5 * within_ss / obs_var
        - 0.5 * jnp.log(sample_size)
    )


def kalman_filter(
    params: PopulationParams,
    gap_days: Array,
    mean: Array,
    sample_size: Array,
    within_ss: Array,
    mask: Array,
) -> FilterResult:
    """Filter one player's appearances and return the exact log-likelihood.

    The measurement noise on an appearance mean is ``tau^2 + r^2 / n``.
    Padded steps (``mask`` False) carry the state forward unchanged and add
    nothing to the likelihood. The covariance update uses the Joseph form.

    Args:
        params: Population parameters.
        gap_days: Days since the previous appearance, shape ``(T,)``.
        mean: Appearance means, shape ``(T,)``.
        sample_size: Pitches per appearance, shape ``(T,)``.
        within_ss: Within-appearance sums of squares, shape ``(T,)``.
        mask: True for real appearances, shape ``(T,)``.

    Returns:
        Filtered and predicted moments, transitions and the log-likelihood.
    """
    obs_var = params.observation_sd**2
    day_var = params.day_effect_sd**2
    identity = jnp.eye(STATE_DIM)

    def step(
        carry: tuple[Array, Array, Array], inputs: tuple[Array, ...]
    ) -> tuple[tuple[Array, Array, Array], tuple[Array, ...]]:
        state_mean, state_cov, log_likelihood = carry
        gap, appearance_mean, n, ss, observed = inputs
        transition, noise = talent_transition(
            params.mean_reversion, params.diffusion, gap
        )
        predicted_mean = transition @ state_mean
        predicted_cov = transition @ state_cov @ transition.T + noise
        measurement_var = day_var + obs_var / n
        innovation = appearance_mean - OBSERVATION_ROW @ predicted_mean
        innovation_var = (
            OBSERVATION_ROW @ predicted_cov @ OBSERVATION_ROW + measurement_var
        )
        gain = predicted_cov @ OBSERVATION_ROW / innovation_var
        correction = identity - jnp.outer(gain, OBSERVATION_ROW)
        updated_mean = predicted_mean + gain * innovation
        updated_cov = (
            correction @ predicted_cov @ correction.T
            + jnp.outer(gain, gain) * measurement_var
        )
        step_log_likelihood = (
            -0.5 * (LOG_TWO_PI + jnp.log(innovation_var))
            - 0.5 * innovation**2 / innovation_var
            + residual_log_likelihood(n, ss, obs_var)
        )
        new_mean = jnp.where(observed, updated_mean, predicted_mean)
        new_cov = jnp.where(observed, updated_cov, predicted_cov)
        new_log_likelihood = log_likelihood + jnp.where(
            observed, step_log_likelihood, 0.0
        )
        outputs = (new_mean, new_cov, predicted_mean, predicted_cov, transition)
        return (new_mean, new_cov, new_log_likelihood), outputs

    prior_mean, prior_cov = initial_state(params)
    (_, _, log_likelihood), outputs = jax.lax.scan(
        step,
        (prior_mean, prior_cov, jnp.zeros(())),
        (gap_days, mean, sample_size, within_ss, mask),
    )
    filtered_means, filtered_covs, predicted_means, predicted_covs, transitions = (
        outputs
    )
    return FilterResult(
        filtered_means=filtered_means,
        filtered_covs=filtered_covs,
        predicted_means=predicted_means,
        predicted_covs=predicted_covs,
        transitions=transitions,
        log_likelihood=log_likelihood,
    )


def rts_smoother(result: FilterResult) -> SmoothedResult:
    """Rauch-Tung-Striebel smoother: condition every state on all appearances.

    Runs backward from the last appearance with gain
    ``G_k = P_k Phi_k+1^T (P^-_k+1)^-1``.

    Args:
        result: Output of ``kalman_filter`` for one player.

    Returns:
        Smoothed means and covariances.
    """

    def step(
        carry: tuple[Array, Array], inputs: tuple[Array, ...]
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        next_mean, next_cov = carry
        filtered_mean, filtered_cov, next_pred_mean, next_pred_cov, next_trans = inputs
        gain = jnp.linalg.solve(next_pred_cov, next_trans @ filtered_cov).T
        mean = filtered_mean + gain @ (next_mean - next_pred_mean)
        cov = filtered_cov + gain @ (next_cov - next_pred_cov) @ gain.T
        cov = 0.5 * (cov + cov.T)
        return (mean, cov), (mean, cov)

    last_mean = result.filtered_means[-1]
    last_cov = result.filtered_covs[-1]
    _, (means, covs) = jax.lax.scan(
        step,
        (last_mean, last_cov),
        (
            result.filtered_means[:-1],
            result.filtered_covs[:-1],
            result.predicted_means[1:],
            result.predicted_covs[1:],
            result.transitions[1:],
        ),
        reverse=True,
    )
    return SmoothedResult(
        means=jnp.concatenate([means, last_mean[None]]),
        covs=jnp.concatenate([covs, last_cov[None]]),
    )


def population_log_likelihood(
    params: PopulationParams, batch: ObservationBatch
) -> Array:
    """Total log-likelihood of every player's appearances.

    Args:
        params: Population parameters.
        batch: Padded appearances for all players.

    Returns:
        Sum over players of ``log p(player's appearances)``.
    """
    per_player = jax.vmap(
        lambda gap, mean, size, ss, mask: (
            kalman_filter(params, gap, mean, size, ss, mask).log_likelihood
        )
    )(
        batch.gap_days,
        adjusted_means(params, batch),
        batch.sample_size,
        batch.within_ss,
        batch.mask,
    )
    return jnp.sum(per_player)


def batch_from_frame(appearances: pl.DataFrame) -> tuple[ObservationBatch, Array]:
    """Pad a long frame of appearances into a batch.

    Args:
        appearances: Columns ``player_id``, ``game_date``, ``sample_size``,
            ``mean``, ``within_ss``.

    Returns:
        The padded batch and the player id for each row of the batch.

    Example:
        >>> from datetime import date
        >>> frame = pl.DataFrame({
        ...     "player_id": [7, 7, 9], "game_date": [date(2026, 4, 1),
        ...     date(2026, 4, 6), date(2026, 4, 2)], "sample_size": [90, 85, 20],
        ...     "mean": [95.0, 94.5, 93.0], "within_ss": [60.0, 55.0, 12.0]})
        >>> batch, player_ids = batch_from_frame(frame)
        >>> batch.gap_days.tolist(), player_ids.tolist()
        ([[0.0, 5.0], [0.0, 0.0]], [7, 9])
        >>> batch.season_day.tolist()
        [[31.0, 36.0], [32.0, 0.0]]
    """
    ordered = appearances.sort("player_id", "game_date").with_columns(
        player_index=pl.col("player_id").rank("dense") - 1,
        position=pl.int_range(pl.len()).over("player_id"),
        gap_days=pl.col("game_date")
        .diff()
        .over("player_id")
        .dt.total_days()
        .fill_null(0)
        .cast(pl.Float64),
        season_day=season_day(pl.col("game_date")).cast(pl.Float64),
    )
    rows = ordered["player_index"].to_numpy()
    columns = ordered["position"].to_numpy()
    shape = (int(rows.max()) + 1, int(columns.max()) + 1)

    def scatter(column: str, fill: float) -> Array:
        values = np.full(shape, fill)
        values[rows, columns] = ordered[column].to_numpy()
        return jnp.asarray(values)

    mask = np.zeros(shape, dtype=bool)
    mask[rows, columns] = True
    batch = ObservationBatch(
        gap_days=scatter("gap_days", 0.0),
        mean=scatter("mean", 0.0),
        sample_size=scatter("sample_size", 1.0),
        within_ss=scatter("within_ss", 0.0),
        mask=jnp.asarray(mask),
        season_day=scatter("season_day", 0.0),
    )
    player_ids = ordered.unique("player_id", maintain_order=True)["player_id"]
    return batch, jnp.asarray(player_ids.to_numpy())
