"""Maximum-likelihood estimation of population parameters with L-BFGS."""

from collections.abc import Callable, Sequence
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


class OptimizerCoordinates(NamedTuple):
    """Affine map from the unconstrained vector to the optimizer's coordinates.

    The optimizer works on ``(raw - shift) / scale``.

    Attributes:
        shift: Subtracted from each unconstrained parameter.
        scale: Then divided into it.
    """

    shift: Array
    scale: Array


class Minimum(NamedTuple):
    """Where L-BFGS stopped.

    Attributes:
        raw: Minimizer on the unconstrained scale.
        iterations: Iterations used.
        gradient_norm: Final gradient norm in optimizer coordinates.
    """

    raw: Array
    iterations: Array
    gradient_norm: Array


class DerivedQuantity(NamedTuple):
    """A positive quantity whose log is linear in the unconstrained vector.

    ``log(value) = offset + sum(weight * raw[index])``, so its delta-method
    interval on the log scale is exact.

    Attributes:
        name: Row name in the parameter table.
        offset: Constant term of the log.
        weights: ``(index, weight)`` pairs of the linear term.
    """

    name: str
    offset: float
    weights: tuple[tuple[int, float], ...]


OU_DERIVED = (
    DerivedQuantity("half_life_days", float(np.log(np.log(2.0))), ((0, -1.0),)),
    DerivedQuantity("stationary_sd", float(-0.5 * np.log(2.0)), ((0, -0.5), (1, 1.0))),
)


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


def optimizer_coordinates(batch: ObservationBatch) -> OptimizerCoordinates:
    """Shift and scale that put the mean and seasonal knots in pooled-sd units.

    The optimizer works on ``(raw - shift) / scale``. Log-scale parameters are
    left alone, since a change of units only shifts them; the population mean
    and the knots are centered on the pooled mean and divided by the pooled
    per-pitch sd. The likelihood is unchanged, but L-BFGS then follows the
    same path whatever the units, so rpm and degrees converge as fast as mph.

    Args:
        batch: Padded appearances.

    Returns:
        ``shift`` and ``scale``, each the length of the unconstrained vector.
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
    return OptimizerCoordinates(shift=shift, scale=scale)


def minimize(
    objective: Callable[[Array], Array],
    start: Array,
    coordinates: OptimizerCoordinates,
    max_iterations: int,
    tolerance: float,
) -> Minimum:
    """Minimize a smooth objective with L-BFGS in rescaled coordinates.

    Args:
        objective: Function of the unconstrained vector, ideally of order one.
        start: Starting point on the unconstrained scale.
        coordinates: Shift and scale defining the optimizer's coordinates.
        max_iterations: Iteration cap.
        tolerance: Stop when the gradient norm (in optimizer coordinates)
            falls below this.

    Returns:
        The minimizer on the unconstrained scale and convergence diagnostics.

    Example:
        >>> coordinates = OptimizerCoordinates(jnp.zeros(2), jnp.ones(2))
        >>> result = minimize(lambda x: jnp.sum((x - 3.0) ** 2), jnp.zeros(2),
        ...                   coordinates, 50, 1e-10)
        >>> [round(float(value), 6) for value in result.raw]
        [3.0, 3.0]
    """
    shift, scale = coordinates

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

    initial = (start - shift) / scale
    point, state = jax.lax.while_loop(keep_going, step, (initial, solver.init(initial)))
    return Minimum(
        raw=shift + scale * point,
        iterations=otu.tree_get(state, "count"),
        gradient_norm=otu.tree_norm(otu.tree_get(state, "grad")),
    )


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

    def objective(raw: Array) -> Array:
        return -population_log_likelihood(constrain(raw), batch) / total_count

    found = minimize(
        objective,
        unconstrain(initial),
        optimizer_coordinates(batch),
        max_iterations,
        tolerance,
    )
    hessian = jax.hessian(objective)(found.raw) * total_count
    return FitResult(
        params=constrain(found.raw),
        unconstrained=found.raw,
        unconstrained_cov=jnp.linalg.inv(hessian),
        log_likelihood=-objective(found.raw) * total_count,
        iterations=found.iterations,
        gradient_norm=found.gradient_norm,
    )


def parameter_table(
    raw: Array,
    cov: Array,
    names: Sequence[str],
    log_scale: Sequence[bool],
    derived: Sequence[DerivedQuantity],
) -> pl.DataFrame:
    """Estimates with 95% Wald intervals, plus derived positive quantities.

    Intervals are computed on the unconstrained scale and mapped back, so
    positive parameters get asymmetric intervals that stay positive.

    Args:
        raw: Unconstrained estimate.
        cov: Its covariance.
        names: Parameter names, in the order of ``raw``.
        log_scale: True where ``raw`` holds the log of the parameter.
        derived: Extra rows, each a positive quantity log-linear in ``raw``.

    Returns:
        Columns ``parameter``, ``estimate``, ``lower_95``, ``upper_95``.
    """
    raw_values = np.asarray(raw)
    covariance = np.asarray(cov)
    gradients = np.zeros((len(derived), raw_values.size))
    offsets = np.zeros(len(derived))
    for row, quantity in enumerate(derived):
        offsets[row] = quantity.offset
        for index, weight in quantity.weights:
            gradients[row, index] = weight
    derived_se = np.sqrt(np.einsum("ij,jk,ik->i", gradients, covariance, gradients))
    centers = np.concatenate([raw_values, offsets + gradients @ raw_values])
    standard_errors = np.concatenate([np.sqrt(np.diag(covariance)), derived_se])
    is_log = np.array([*log_scale, *([True] * len(derived))])
    bounds = centers[:, None] + np.array([0.0, -Z_95, Z_95]) * standard_errors[:, None]
    values = bounds.copy()
    values[is_log] = np.exp(bounds[is_log])
    return pl.DataFrame(
        {
            "parameter": [*names, *(quantity.name for quantity in derived)],
            "estimate": values[:, 0],
            "lower_95": values[:, 1],
            "upper_95": values[:, 2],
        }
    )


def summarize_fit(fit: FitResult) -> pl.DataFrame:
    """Tabulate estimates with 95% intervals, plus two derived quantities.

    Seasonal knots (``season_apr`` and so on) are shifts from the July 1
    level, in the metric's units. The talent half-life ``ln 2 / kappa`` and
    stationary spread ``sigma / sqrt(2 kappa)`` get delta-method intervals on
    the log scale.

    Args:
        fit: Output of ``fit_population``.

    Returns:
        Columns ``parameter``, ``estimate``, ``lower_95``, ``upper_95``.
    """
    return parameter_table(
        fit.unconstrained, fit.unconstrained_cov, PARAMETER_NAMES, LOG_SCALE, OU_DERIVED
    )
