"""Jump-diffusion talent: the OU model plus sudden jumps in talent.

Talent follows the OU process of ``kalman.py`` and can also jump. Between two
appearances ``Delta`` days apart a jump happens with probability
``p = 1 - exp(-lambda Delta)`` (a Poisson process with rate ``lambda`` per day,
with at most one jump per gap), and it moves talent by
``eta ~ N(mu_J, sigma_J^2)`` just before the appearance. A jump then decays
back toward the player's level at the OU rate, like any other deviation.

Given which gaps had a jump the model is linear-Gaussian, so the exact filter
is a mixture with one Gaussian per jump history. This module provides the
pieces shared by both approximations used in the project:

* ``assumed_density_filter``: a deterministic one-step Gaussian-sum filter.
  Each step splits the state into a jump and a no-jump branch, updates both
  exactly, and collapses them back to one Gaussian by matching moments. Its
  likelihood is smooth in the parameters, so it is what the parameters are
  fitted to.
* ``models.rbpf``: a Rao-Blackwellized particle filter that keeps whole jump
  histories, used for the reported jump probabilities.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import polars as pl
from jax import Array

from statecast.models.dynamics import STATE_DIM, talent_transition
from statecast.models.estimation import (
    LOG_SCALE,
    OU_DERIVED,
    PARAMETER_NAMES,
    DerivedQuantity,
    OptimizerCoordinates,
    minimize,
    optimizer_coordinates,
    parameter_table,
)
from statecast.models.kalman import (
    LOG_TWO_PI,
    ObservationBatch,
    PopulationParams,
    adjusted_means,
    constrain,
    initial_state,
    residual_log_likelihood,
    unconstrain,
)

OU_SIZE = len(PARAMETER_NAMES)
JUMP_PARAMETER_NAMES = ("jump_rate", "jump_mean", "jump_sd")
JUMP_LOG_SCALE = (True, False, True)
JUMP_MEAN_INDEX = OU_SIZE + 1
JUMP_DERIVED = (
    *OU_DERIVED,
    DerivedQuantity("days_between_jumps", 0.0, ((OU_SIZE, -1.0),)),
)
DEFAULT_DAYS_BETWEEN_JUMPS = 180.0
DEFAULT_JUMP_SD_MULTIPLE = 2.0


class JumpParams(NamedTuple):
    """Jump component of the talent process.

    Attributes:
        rate: ``lambda``, expected jumps per day.
        mean: ``mu_J``, mean jump size (metric units).
        sd: ``sigma_J``, spread of jump sizes (metric units).
    """

    rate: Array
    mean: Array
    sd: Array


class JumpModel(NamedTuple):
    """Every parameter of the jump-diffusion model.

    Attributes:
        population: The OU, day-effect and seasonal parameters.
        jumps: The jump component.
    """

    population: PopulationParams
    jumps: JumpParams


class BranchUpdate(NamedTuple):
    """One Kalman measurement update.

    Attributes:
        mean: Updated state mean.
        cov: Updated state covariance.
        log_likelihood: Log predictive density of the observation.
    """

    mean: Array
    cov: Array
    log_likelihood: Array


class JumpFilterResult(NamedTuple):
    """Assumed-density filter output for one player.

    Attributes:
        filtered_means: Collapsed state means, shape ``(T, 2)``.
        filtered_covs: Collapsed state covariances, shape ``(T, 2, 2)``.
        jump_probs: ``P(jump just before appearance k | appearances 1..k)``.
        log_likelihood: Approximate total log-likelihood.
    """

    filtered_means: Array
    filtered_covs: Array
    jump_probs: Array
    log_likelihood: Array


class JumpFitResult(NamedTuple):
    """Outcome of a maximum-likelihood jump-diffusion fit.

    Attributes:
        model: Estimated natural-scale parameters.
        unconstrained: Estimate as the OU vector followed by
            ``[log lambda, mu_J, log sigma_J]``.
        unconstrained_cov: Inverse observed information on that scale.
        log_likelihood: Maximized (assumed-density) log-likelihood.
        iterations: L-BFGS iterations used.
        gradient_norm: Final gradient norm in optimizer coordinates.
    """

    model: JumpModel
    unconstrained: Array
    unconstrained_cov: Array
    log_likelihood: Array
    iterations: Array
    gradient_norm: Array


def log_jump_odds(rate: Array, gap_days: Array) -> tuple[Array, Array]:
    """Log prior probabilities of no jump and of a jump over a gap.

    ``log P(no jump) = -lambda Delta`` and ``log P(jump) = log(1 - e^(-lambda
    Delta))``. A gap of 0 (a player's first appearance, or padding) has no
    jump; the double ``where`` keeps the gradient finite there.

    Args:
        rate: Jump rate ``lambda`` per day.
        gap_days: Days since the previous appearance.

    Returns:
        ``(log P(no jump), log P(jump))``.

    Example:
        >>> no_jump, jump = log_jump_odds(jnp.asarray(0.01), jnp.asarray(10.0))
        >>> round(float(jnp.exp(jump)), 4)
        0.0952
    """
    has_gap = gap_days > 0.0
    safe_gap = jnp.where(has_gap, gap_days, 1.0)
    log_jump = jnp.log(-jnp.expm1(-rate * safe_gap))
    return -rate * gap_days, jnp.where(has_gap, log_jump, -jnp.inf)


def kalman_update(
    mean: Array, cov: Array, observation: Array, measurement_var: Array
) -> BranchUpdate:
    """Update a state whose first element is observed with noise.

    Uses the Joseph form, so the covariance stays symmetric and positive
    semi-definite.

    Args:
        mean: Predicted state mean, shape ``(d,)``.
        cov: Predicted state covariance, shape ``(d, d)``.
        observation: Observed value of the first state element plus noise.
        measurement_var: Noise variance.

    Returns:
        The updated mean and covariance and the observation's log density.
    """
    innovation = observation - mean[0]
    innovation_var = cov[0, 0] + measurement_var
    gain = cov[:, 0] / innovation_var
    selector = jnp.zeros(mean.shape[0]).at[0].set(1.0)
    correction = jnp.eye(mean.shape[0]) - jnp.outer(gain, selector)
    updated_cov = (
        correction @ cov @ correction.T + jnp.outer(gain, gain) * measurement_var
    )
    log_likelihood = -0.5 * (
        LOG_TWO_PI + jnp.log(innovation_var) + innovation**2 / innovation_var
    )
    return BranchUpdate(mean + gain * innovation, updated_cov, log_likelihood)


def assumed_density_filter(
    model: JumpModel,
    gap_days: Array,
    mean: Array,
    sample_size: Array,
    within_ss: Array,
    mask: Array,
) -> JumpFilterResult:
    """Filter one player with a jump and a no-jump branch at every step.

    Both branches are updated exactly, weighted by their posterior
    probabilities, and collapsed to one Gaussian with the same mean and
    covariance as the two-component mixture. The likelihood of each step is
    exact given the collapsed prior, so the only approximation is the
    collapse. ``mean`` must already be season-adjusted.

    Args:
        model: Jump-diffusion parameters.
        gap_days: Days since the previous appearance, shape ``(T,)``.
        mean: Season-adjusted appearance means, shape ``(T,)``.
        sample_size: Pitches per appearance, shape ``(T,)``.
        within_ss: Within-appearance sums of squares, shape ``(T,)``.
        mask: True for real appearances, shape ``(T,)``.

    Returns:
        Collapsed filtered moments, jump probabilities and the log-likelihood.
    """
    params, jumps = model
    obs_var = params.observation_sd**2
    day_var = params.day_effect_sd**2
    selector = jnp.zeros(STATE_DIM).at[0].set(1.0)

    def step(
        carry: tuple[Array, Array, Array], inputs: tuple[Array, ...]
    ) -> tuple[tuple[Array, Array, Array], tuple[Array, Array, Array]]:
        state_mean, state_cov, log_likelihood = carry
        gap, observation, n, ss, observed = inputs
        transition, noise = talent_transition(
            params.mean_reversion, params.diffusion, gap
        )
        predicted_mean = transition @ state_mean
        predicted_cov = transition @ state_cov @ transition.T + noise
        measurement_var = day_var + obs_var / n
        still = kalman_update(
            predicted_mean, predicted_cov, observation, measurement_var
        )
        jumped = kalman_update(
            predicted_mean + jumps.mean * selector,
            predicted_cov + jumps.sd**2 * jnp.outer(selector, selector),
            observation,
            measurement_var,
        )
        log_no_jump, log_jump = log_jump_odds(jumps.rate, gap)
        joint_still = log_no_jump + still.log_likelihood
        joint_jumped = log_jump + jumped.log_likelihood
        log_marginal = jnp.logaddexp(joint_still, joint_jumped)
        jump_prob = jnp.exp(joint_jumped - log_marginal)
        collapsed_mean = (1.0 - jump_prob) * still.mean + jump_prob * jumped.mean
        still_offset = still.mean - collapsed_mean
        jumped_offset = jumped.mean - collapsed_mean
        collapsed_cov = (1.0 - jump_prob) * (
            still.cov + jnp.outer(still_offset, still_offset)
        ) + jump_prob * (jumped.cov + jnp.outer(jumped_offset, jumped_offset))
        step_log_likelihood = log_marginal + residual_log_likelihood(n, ss, obs_var)
        new_mean = jnp.where(observed, collapsed_mean, predicted_mean)
        new_cov = jnp.where(observed, collapsed_cov, predicted_cov)
        new_log_likelihood = log_likelihood + jnp.where(
            observed, step_log_likelihood, 0.0
        )
        outputs = (new_mean, new_cov, jnp.where(observed, jump_prob, 0.0))
        return (new_mean, new_cov, new_log_likelihood), outputs

    prior_mean, prior_cov = initial_state(params)
    (_, _, log_likelihood), (means, covs, jump_probs) = jax.lax.scan(
        step,
        (prior_mean, prior_cov, jnp.zeros(())),
        (gap_days, mean, sample_size, within_ss, mask),
    )
    return JumpFilterResult(means, covs, jump_probs, log_likelihood)


def jump_population_log_likelihood(model: JumpModel, batch: ObservationBatch) -> Array:
    """Total assumed-density log-likelihood of every player's appearances.

    Args:
        model: Jump-diffusion parameters.
        batch: Padded appearances for all players.

    Returns:
        Sum over players of the approximate log-likelihood.
    """
    per_player = jax.vmap(
        lambda gap, mean, size, ss, mask: (
            assumed_density_filter(model, gap, mean, size, ss, mask).log_likelihood
        )
    )(
        batch.gap_days,
        adjusted_means(model.population, batch),
        batch.sample_size,
        batch.within_ss,
        batch.mask,
    )
    return jnp.sum(per_player)


def constrain_jump_model(raw: Array) -> JumpModel:
    """Map an unconstrained vector of length 15 to jump-diffusion parameters.

    Args:
        raw: The OU vector of ``kalman.constrain`` followed by
            ``[log lambda, mu_J, log sigma_J]``.

    Returns:
        The natural-scale parameters.
    """
    return JumpModel(
        population=constrain(raw[:OU_SIZE]),
        jumps=JumpParams(
            rate=jnp.exp(raw[OU_SIZE]),
            mean=raw[OU_SIZE + 1],
            sd=jnp.exp(raw[OU_SIZE + 2]),
        ),
    )


def unconstrain_jump_model(model: JumpModel) -> Array:
    """Inverse of ``constrain_jump_model``.

    Args:
        model: Natural-scale parameters.

    Returns:
        The unconstrained vector of length 15.
    """
    jump_part = jnp.array(
        [jnp.log(model.jumps.rate), model.jumps.mean, jnp.log(model.jumps.sd)]
    )
    return jnp.concatenate([unconstrain(model.population), jump_part])


def initial_jump_model(population: PopulationParams) -> JumpModel:
    """Starting values: an OU fit plus rare, symmetric, sizeable jumps.

    The jump rate starts at one jump per 180 days and the jump size spread at
    twice the typical appearance-to-appearance swing ``sqrt(tau^2 + v)``, so
    jumps start out distinguishable from ordinary noise.

    Args:
        population: Usually the OU maximum-likelihood estimate.

    Returns:
        A full jump-diffusion parameter set.
    """
    stationary_var = population.diffusion**2 / (2.0 * population.mean_reversion)
    swing = jnp.sqrt(population.day_effect_sd**2 + stationary_var)
    return JumpModel(
        population=population,
        jumps=JumpParams(
            rate=jnp.asarray(1.0 / DEFAULT_DAYS_BETWEEN_JUMPS),
            mean=jnp.asarray(0.0),
            sd=DEFAULT_JUMP_SD_MULTIPLE * swing,
        ),
    )


def jump_optimizer_coordinates(batch: ObservationBatch) -> OptimizerCoordinates:
    """Unit-free coordinates for the 15-parameter vector.

    The OU part is as in ``optimizer_coordinates``; the jump mean is divided by
    the pooled per-pitch sd, and the two log-scale jump parameters are left
    alone.

    Args:
        batch: Padded appearances.

    Returns:
        Shift and scale for the optimizer.
    """
    shift, scale = optimizer_coordinates(batch)
    spread = scale[OU_SIZE - 1]
    jump_scale = jnp.array([1.0, 1.0, 1.0]).at[1].set(spread)
    return OptimizerCoordinates(
        shift=jnp.concatenate([shift, jnp.zeros(3)]),
        scale=jnp.concatenate([scale, jump_scale]),
    )


def fit_jump_population(
    batch: ObservationBatch,
    initial: JumpModel,
    max_iterations: int = 300,
    tolerance: float = 1e-8,
) -> JumpFitResult:
    """Maximize the assumed-density likelihood over all 15 parameters.

    The objective is the negative log-likelihood per pitch; the covariance is
    the inverse Hessian of the total at the optimum, on the unconstrained
    scale.

    Args:
        batch: Padded appearances.
        initial: Starting parameters, usually from ``initial_jump_model``.
        max_iterations: L-BFGS iteration cap.
        tolerance: Stop when the gradient norm (in optimizer coordinates)
            falls below this.

    Returns:
        Estimates, their covariance and convergence diagnostics.
    """
    total_count = jnp.sum(jnp.where(batch.mask, batch.sample_size, 0.0))

    def objective(raw: Array) -> Array:
        model = constrain_jump_model(raw)
        return -jump_population_log_likelihood(model, batch) / total_count

    found = minimize(
        objective,
        unconstrain_jump_model(initial),
        jump_optimizer_coordinates(batch),
        max_iterations,
        tolerance,
    )
    hessian = jax.hessian(objective)(found.raw) * total_count
    return JumpFitResult(
        model=constrain_jump_model(found.raw),
        unconstrained=found.raw,
        unconstrained_cov=jnp.linalg.inv(hessian),
        log_likelihood=-objective(found.raw) * total_count,
        iterations=found.iterations,
        gradient_norm=found.gradient_norm,
    )


def summarize_jump_fit(fit: JumpFitResult) -> pl.DataFrame:
    """Tabulate jump-diffusion estimates with 95% intervals.

    Rows are the OU parameters and seasonal knots, the three jump parameters,
    the talent half-life and stationary spread, and ``days_between_jumps``
    (``1 / lambda``).

    Args:
        fit: Output of ``fit_jump_population``.

    Returns:
        Columns ``parameter``, ``estimate``, ``lower_95``, ``upper_95``.
    """
    return parameter_table(
        fit.unconstrained,
        fit.unconstrained_cov,
        (*PARAMETER_NAMES, *JUMP_PARAMETER_NAMES),
        (*LOG_SCALE, *JUMP_LOG_SCALE),
        JUMP_DERIVED,
    )
