"""Maximum-likelihood estimation of population parameters with L-BFGS."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import optax.tree_utils as otu
import polars as pl
from jax import Array

from statecast.models.kalman import (
    ObservationBatch,
    PopulationParams,
    constrain,
    population_log_likelihood,
    unconstrain,
)
from statecast.models.seasonal import FREE_KNOT_COUNT, FREE_KNOT_NAMES

PRIOR_HALF_LIFE_DAYS = 60.0
MIN_EXCESS_FRACTION = 0.01
Z_95 = 1.959963984540054
PARAMETER_NAMES = (
    "mean_reversion",
    "diffusion",
    "observation_sd",
    "population_mean",
    "population_sd",
    "day_effect_sd",
    *(f"season_{name}" for name in FREE_KNOT_NAMES),
)
LOG_SCALE = (True, True, True, False, True, True, *([False] * FREE_KNOT_COUNT))
MEAN_INDEX = PARAMETER_NAMES.index("population_mean")
KNOT_SLICE = slice(6, 6 + FREE_KNOT_COUNT)


class FitResult(NamedTuple):
    """Outcome of a maximum-likelihood fit.

    Attributes:
        params: Estimated natural-scale parameters.
        unconstrained: Estimate as ``[log kappa, log sigma, log r, m, log s,
            log tau, seasonal knots...]``.
        unconstrained_cov: Inverse observed information on that scale.
        log_likelihood: Maximized total log-likelihood.
        iterations: L-BFGS iterations used.
        gradient_norm: Final gradient norm of the per-pitch objective.
    """

    params: PopulationParams
    unconstrained: Array
    unconstrained_cov: Array
    log_likelihood: Array
    iterations: Array
    gradient_norm: Array


def initial_guess(batch: ObservationBatch) -> PopulationParams:
    """Method-of-moments starting values for the optimizer.

    Args:
        batch: Padded appearances.

    Returns:
        Starting parameters: the pooled mean and within-appearance variance,
        the spread of player means, and a 60-day half-life. The excess
        appearance-to-appearance variance within players (beyond the
        ``r^2 / n`` sampling noise) is split evenly between the day effect and
        stationary talent wander. The seasonal curve starts flat.
    """
    weights = jnp.where(batch.mask, batch.sample_size, 0.0)
    appearances = jnp.sum(batch.mask)
    population_mean = jnp.sum(weights * batch.mean) / jnp.sum(weights)
    observation_var = jnp.sum(jnp.where(batch.mask, batch.within_ss, 0.0)) / jnp.sum(
        jnp.where(batch.mask, batch.sample_size - 1.0, 0.0)
    )
    player_means = jnp.sum(weights * batch.mean, axis=1) / jnp.sum(weights, axis=1)
    population_sd = jnp.std(player_means)
    deviations = jnp.where(batch.mask, batch.mean - player_means[:, None], 0.0)
    sampling_var = jnp.where(batch.mask, observation_var / batch.sample_size, 0.0)
    excess_var = jnp.maximum(
        jnp.sum(deviations**2 - sampling_var) / appearances,
        MIN_EXCESS_FRACTION * observation_var,
    )
    mean_reversion = jnp.log(2.0) / PRIOR_HALF_LIFE_DAYS
    return PopulationParams(
        mean_reversion=jnp.asarray(mean_reversion),
        diffusion=jnp.sqrt(excess_var * mean_reversion),
        observation_sd=jnp.sqrt(observation_var),
        population_mean=population_mean,
        population_sd=population_sd,
        day_effect_sd=jnp.sqrt(0.5 * excess_var),
        seasonal_knots=jnp.zeros(FREE_KNOT_COUNT),
    )


def optimizer_coordinates(batch: ObservationBatch) -> tuple[Array, Array]:
    """Shift and scale that put the mean and seasonal knots in pooled-sd units.

    The optimizer works on ``(raw - shift) / scale``. Log-scale parameters are
    left alone, since a change of units only shifts them; the population mean
    and the knots are centered on the pooled mean and divided by the pooled
    per-pitch sd. The likelihood is unchanged, but L-BFGS then follows the
    same path whatever the units, so rpm and degrees converge as fast as mph.

    Args:
        batch: Padded appearances.

    Returns:
        ``(shift, scale)``, each the length of the unconstrained vector.
    """
    weights = jnp.where(batch.mask, batch.sample_size, 0.0)
    total = jnp.sum(weights)
    center = jnp.sum(weights * batch.mean) / total
    within = jnp.sum(jnp.where(batch.mask, batch.within_ss, 0.0))
    between = jnp.sum(weights * (batch.mean - center) ** 2)
    spread = jnp.sqrt((within + between) / total)
    size = len(PARAMETER_NAMES)
    shift = jnp.zeros(size).at[MEAN_INDEX].set(center)
    scale = jnp.ones(size).at[MEAN_INDEX].set(spread).at[KNOT_SLICE].set(spread)
    return shift, scale


def fit_population(
    batch: ObservationBatch,
    initial: PopulationParams,
    max_iterations: int = 200,
    tolerance: float = 1e-8,
) -> FitResult:
    """Maximize the population log-likelihood over the unconstrained parameters.

    The objective is the negative log-likelihood per pitch, so the gradient
    tolerance does not depend on how much data is in the batch. L-BFGS runs in
    the unit-free coordinates of ``optimizer_coordinates``. The covariance is
    the inverse Hessian of the total negative log-likelihood at the optimum,
    on the usual unconstrained scale.

    Args:
        batch: Padded appearances.
        initial: Starting parameters, usually from ``initial_guess``.
        max_iterations: L-BFGS iteration cap.
        tolerance: Stop when the gradient norm (in optimizer coordinates)
            falls below this.

    Returns:
        Estimates, their covariance and convergence diagnostics.
    """
    total_count = jnp.sum(jnp.where(batch.mask, batch.sample_size, 0.0))
    shift, scale = optimizer_coordinates(batch)

    def objective(raw: Array) -> Array:
        return -population_log_likelihood(constrain(raw), batch) / total_count

    def scaled_objective(point: Array) -> Array:
        return objective(shift + scale * point)

    solver = optax.lbfgs()
    value_and_grad = optax.value_and_grad_from_state(scaled_objective)

    def step(carry: tuple[Array, optax.OptState]) -> tuple[Array, optax.OptState]:
        point, state = carry
        value, grad = value_and_grad(point, state=state)
        updates, state = solver.update(
            grad, state, point, value=value, grad=grad, value_fn=scaled_objective
        )
        return optax.apply_updates(point, updates), state

    def keep_going(carry: tuple[Array, optax.OptState]) -> Array:
        _, state = carry
        iteration: Array = otu.tree_get(state, "count")
        gradient_norm: Array = otu.tree_norm(otu.tree_get(state, "grad"))
        return (iteration == 0) | (
            (iteration < max_iterations) & (gradient_norm >= tolerance)
        )

    start = (unconstrain(initial) - shift) / scale
    point, state = jax.lax.while_loop(keep_going, step, (start, solver.init(start)))
    raw = shift + scale * point
    hessian = jax.hessian(objective)(raw) * total_count
    return FitResult(
        params=constrain(raw),
        unconstrained=raw,
        unconstrained_cov=jnp.linalg.inv(hessian),
        log_likelihood=-objective(raw) * total_count,
        iterations=otu.tree_get(state, "count"),
        gradient_norm=otu.tree_norm(otu.tree_get(state, "grad")),
    )


def summarize_fit(fit: FitResult) -> pl.DataFrame:
    """Tabulate estimates with 95% intervals, plus two derived quantities.

    Seasonal knots (``season_apr`` and so on) are shifts from the July 1
    level, in the metric's units.

    Intervals are Wald intervals on the unconstrained scale, mapped back, so
    positive parameters get asymmetric intervals that stay positive. The talent
    half-life ``ln 2 / kappa`` and stationary spread ``sigma / sqrt(2 kappa)``
    follow from their own delta-method intervals on the log scale.

    Args:
        fit: Output of ``fit_population``.

    Returns:
        Columns ``parameter``, ``estimate``, ``lower_95``, ``upper_95``.
    """
    raw = np.asarray(fit.unconstrained)
    cov = np.asarray(fit.unconstrained_cov)
    log_half_life = np.log(np.log(2.0)) - raw[0]
    log_stationary_sd = raw[1] - 0.5 * (np.log(2.0) + raw[0])
    gradients = np.zeros((2, raw.size))
    gradients[0, 0] = -1.0
    gradients[1, :2] = (-0.5, 1.0)
    derived_se = np.sqrt(np.einsum("ij,jk,ik->i", gradients, cov, gradients))
    centers = np.concatenate([raw, [log_half_life, log_stationary_sd]])
    standard_errors = np.concatenate([np.sqrt(np.diag(cov)), derived_se])
    log_scale = np.array([*LOG_SCALE, True, True])
    bounds = centers[:, None] + np.array([0.0, -Z_95, Z_95]) * standard_errors[:, None]
    values = bounds.copy()
    values[log_scale] = np.exp(bounds[log_scale])
    return pl.DataFrame(
        {
            "parameter": [*PARAMETER_NAMES, "half_life_days", "stationary_sd"],
            "estimate": values[:, 0],
            "lower_95": values[:, 1],
            "upper_95": values[:, 2],
        }
    )
