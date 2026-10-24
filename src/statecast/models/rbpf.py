"""Rao-Blackwellized particle filter for jump-diffusion talent.

Given which gaps had a jump, the jump-diffusion model is linear-Gaussian. So
each particle carries one jump history (sampled) and an exact Kalman filter
for everything else (integrated out): the Rao-Blackwellization. The state is
``[talent, level, last jump size]``; the third element records the size of
the most recent jump, so the filter can say which way a jump went.

At every appearance each particle splits into a jump and a no-jump branch,
both updated exactly. The particle's weight is multiplied by the two
branches' total predictive density, and its branch is then drawn from their
posterior probabilities: the optimal proposal for this model, so weights
depend only on data already seen. Particles are resampled systematically when
the effective sample size falls below half the particle count.

Reported probabilities use each particle's exact branch probabilities for
the current appearance and its sampled history for earlier ones, which
lowers their Monte Carlo error.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm

from statecast.models.dynamics import talent_transition
from statecast.models.jumps import JumpModel, kalman_update, log_jump_odds
from statecast.models.kalman import (
    ObservationBatch,
    adjusted_means,
    residual_log_likelihood,
)

PARTICLE_STATE_DIM = 3
RESAMPLE_FRACTION = 0.5
KEEP_ON_JUMP = jnp.array([1.0, 1.0, 0.0])
JUMP_LOADING = jnp.array([1.0, 0.0, 1.0])


class FilterSettings(NamedTuple):
    """Particle filter settings.

    Attributes:
        particle_count: Particles per player.
        lag: Appearances in the "recent" window: the current one and the
            ``lag - 1`` before it.
    """

    particle_count: int = 512
    lag: int = 3


class PlayerSeries(NamedTuple):
    """One player's appearances.

    Attributes:
        gap_days: Days since the previous appearance, shape ``(T,)``.
        mean: Season-adjusted appearance means, shape ``(T,)``.
        sample_size: Pitches per appearance, shape ``(T,)``.
        within_ss: Within-appearance sums of squares, shape ``(T,)``.
        mask: True for real appearances, shape ``(T,)``.
    """

    gap_days: Array
    mean: Array
    sample_size: Array
    within_ss: Array
    mask: Array


class ParticleFilterResult(NamedTuple):
    """Real-time output after each appearance (NaN on padding).

    Attributes:
        talent_mean: ``E[talent | appearances 1..k]`` (July 1 equivalent).
        talent_sd: ``sd[talent | appearances 1..k]``.
        jump_prob: ``P(jump just before appearance k | 1..k)``.
        jump_evidence: Log Bayes factor of a jump just before appearance
            ``k``: ``log p(y_k | jump, 1..k-1) - log p(y_k | no jump,
            1..k-1)``. Unlike ``jump_prob`` it does not depend on the prior
            chance of a jump, so a long absence does not inflate it.
        recent_jump_prob: ``P(a jump before any of the last lag appearances
            | 1..k)``.
        recent_drop_prob: ``P(such a jump, and the latest one was downward
            | 1..k)``.
        effective_size: Effective sample size of the weights before
            resampling.
        log_likelihood: Estimated total log-likelihood, shape ``()``.
    """

    talent_mean: Array
    talent_sd: Array
    jump_prob: Array
    jump_evidence: Array
    recent_jump_prob: Array
    recent_drop_prob: Array
    effective_size: Array
    log_likelihood: Array


class _StepOutput(NamedTuple):
    """Per-appearance summaries emitted by the filter's scan."""

    talent_mean: Array
    talent_sd: Array
    jump_prob: Array
    jump_evidence: Array
    recent_jump_prob: Array
    recent_drop_prob: Array
    effective_size: Array


class _Particles(NamedTuple):
    """Filter state carried between appearances."""

    means: Array
    covs: Array
    log_weights: Array
    history: Array
    log_likelihood: Array
    key: Array


def systematic_resample(key: Array, weights: Array) -> Array:
    """Systematic resampling: one uniform draw, evenly spaced pointers.

    Each particle is copied ``floor(N w)`` or ``ceil(N w)`` times, which has
    lower variance than multinomial resampling.

    Args:
        key: JAX random key.
        weights: Normalized weights, shape ``(N,)``.

    Returns:
        Indices of the particles to keep, shape ``(N,)``.

    Example:
        >>> indices = systematic_resample(jax.random.key(0),
        ...                               jnp.array([0.5, 0.0, 0.25, 0.25]))
        >>> sorted(indices.tolist())
        [0, 0, 2, 3]
    """
    count = weights.shape[0]
    pointers = (jax.random.uniform(key) + jnp.arange(count)) / count
    indices = jnp.searchsorted(jnp.cumsum(weights), pointers)
    return jnp.clip(indices, 0, count - 1)


def _transition(model: JumpModel, gap: Array) -> tuple[Array, Array]:
    """OU transition for ``[talent, level, last jump size]``."""
    transition, noise = talent_transition(
        model.population.mean_reversion, model.population.diffusion, gap
    )
    full_transition = (
        jnp.zeros((PARTICLE_STATE_DIM, PARTICLE_STATE_DIM))
        .at[:2, :2]
        .set(transition)
        .at[2, 2]
        .set(1.0)
    )
    full_noise = (
        jnp.zeros((PARTICLE_STATE_DIM, PARTICLE_STATE_DIM)).at[0, 0].set(noise[0, 0])
    )
    return full_transition, full_noise


def _initial_particles(
    model: JumpModel, key: Array, settings: FilterSettings
) -> _Particles:
    """Every particle starts at the population prior, with no jump history."""
    params, jumps = model
    level_var = params.population_sd**2
    stationary_var = params.diffusion**2 / (2.0 * params.mean_reversion)
    mean = jnp.array([params.population_mean, params.population_mean, jumps.mean])
    cov = jnp.array(
        [
            [level_var + stationary_var, level_var, 0.0],
            [level_var, level_var, 0.0],
            [0.0, 0.0, jumps.sd**2],
        ]
    )
    count = settings.particle_count
    return _Particles(
        means=jnp.broadcast_to(mean, (count, PARTICLE_STATE_DIM)),
        covs=jnp.broadcast_to(cov, (count, PARTICLE_STATE_DIM, PARTICLE_STATE_DIM)),
        log_weights=jnp.full(count, -jnp.log(count)),
        history=jnp.zeros((count, settings.lag - 1), dtype=bool),
        log_likelihood=jnp.zeros(()),
        key=key,
    )


def particle_filter(
    model: JumpModel, series: PlayerSeries, key: Array, settings: FilterSettings
) -> ParticleFilterResult:
    """Run the Rao-Blackwellized particle filter over one player.

    With a jump rate of zero every particle follows the same Kalman filter,
    so the output matches ``kalman.kalman_filter`` exactly.

    Args:
        model: Jump-diffusion parameters.
        series: The player's appearances (season-adjusted means).
        key: JAX random key.
        settings: Particle count and recent-window length.

    Returns:
        Real-time talent, jump probabilities and diagnostics per appearance.
    """
    params, jumps = model
    obs_var = params.observation_sd**2
    day_var = params.day_effect_sd**2
    count = settings.particle_count
    update = jax.vmap(kalman_update, in_axes=(0, 0, None, None))

    def step(
        particles: _Particles, inputs: tuple[Array, ...]
    ) -> tuple[_Particles, _StepOutput]:
        gap, observation, n, ss, observed = inputs
        key, branch_key, resample_key = jax.random.split(particles.key, 3)
        transition, noise = _transition(model, gap)
        predicted_means = particles.means @ transition.T
        predicted_covs = transition @ particles.covs @ transition.T + noise
        measurement_var = day_var + obs_var / n
        still = update(predicted_means, predicted_covs, observation, measurement_var)
        jumped = update(
            predicted_means * KEEP_ON_JUMP + jumps.mean * JUMP_LOADING,
            predicted_covs * jnp.outer(KEEP_ON_JUMP, KEEP_ON_JUMP)
            + jumps.sd**2 * jnp.outer(JUMP_LOADING, JUMP_LOADING),
            observation,
            measurement_var,
        )
        log_no_jump, log_jump = log_jump_odds(jumps.rate, gap)
        joint_still = log_no_jump + still.log_likelihood
        joint_jumped = log_jump + jumped.log_likelihood
        log_marginal = jnp.logaddexp(joint_still, joint_jumped)
        branch_prob = jnp.exp(joint_jumped - log_marginal)
        unnormalized = particles.log_weights + log_marginal
        increment = logsumexp(unnormalized)
        log_weights = unnormalized - increment
        weights = jnp.exp(log_weights)

        talent_first = (1.0 - branch_prob) * still.mean[:, 0] + branch_prob * (
            jumped.mean[:, 0]
        )
        talent_second = (1.0 - branch_prob) * (
            still.cov[:, 0, 0] + still.mean[:, 0] ** 2
        ) + branch_prob * (jumped.cov[:, 0, 0] + jumped.mean[:, 0] ** 2)
        talent_mean = jnp.sum(weights * talent_first)
        talent_var = jnp.sum(weights * talent_second) - talent_mean**2
        earlier_jump = jnp.any(particles.history, axis=1)
        recent = branch_prob + (1.0 - branch_prob) * earlier_jump
        drop_if_jumped = norm.cdf(-jumped.mean[:, 2] / jnp.sqrt(jumped.cov[:, 2, 2]))
        drop_if_still = norm.cdf(-still.mean[:, 2] / jnp.sqrt(still.cov[:, 2, 2]))
        recent_drop = branch_prob * drop_if_jumped + (1.0 - branch_prob) * (
            earlier_jump * drop_if_still
        )
        effective_size = 1.0 / jnp.sum(weights**2)
        jump_evidence = logsumexp(particles.log_weights + jumped.log_likelihood) - (
            logsumexp(particles.log_weights + still.log_likelihood)
        )

        took_jump = jax.random.bernoulli(branch_key, branch_prob)
        means = jnp.where(took_jump[:, None], jumped.mean, still.mean)
        covs = jnp.where(took_jump[:, None, None], jumped.cov, still.cov)
        history = jnp.concatenate([took_jump[:, None], particles.history], axis=1)[
            :, : settings.lag - 1
        ]
        resample = effective_size < RESAMPLE_FRACTION * count
        indices = jnp.where(
            resample, systematic_resample(resample_key, weights), jnp.arange(count)
        )
        log_weights = jnp.where(resample, -jnp.log(count), log_weights)
        step_log_likelihood = increment + residual_log_likelihood(n, ss, obs_var)
        updated = _Particles(
            means=means[indices],
            covs=covs[indices],
            log_weights=log_weights,
            history=history[indices],
            log_likelihood=particles.log_likelihood + step_log_likelihood,
            key=key,
        )
        carried = jax.tree.map(
            lambda new, old: jnp.where(observed, new, old),
            updated,
            particles._replace(key=key),
        )
        summary = _StepOutput(
            talent_mean=talent_mean,
            talent_sd=jnp.sqrt(jnp.clip(talent_var, 0.0)),
            jump_prob=jnp.sum(weights * branch_prob),
            jump_evidence=jump_evidence,
            recent_jump_prob=jnp.sum(weights * recent),
            recent_drop_prob=jnp.sum(weights * recent_drop),
            effective_size=effective_size,
        )
        masked = jax.tree.map(
            lambda value: jnp.where(observed, value, jnp.nan), summary
        )
        return carried, masked

    final, outputs = jax.lax.scan(
        step,
        _initial_particles(model, key, settings),
        (
            series.gap_days,
            series.mean,
            series.sample_size,
            series.within_ss,
            series.mask,
        ),
    )
    return ParticleFilterResult(
        talent_mean=outputs.talent_mean,
        talent_sd=outputs.talent_sd,
        jump_prob=outputs.jump_prob,
        jump_evidence=outputs.jump_evidence,
        recent_jump_prob=outputs.recent_jump_prob,
        recent_drop_prob=outputs.recent_drop_prob,
        effective_size=outputs.effective_size,
        log_likelihood=final.log_likelihood,
    )


def filter_jump_population(
    model: JumpModel, batch: ObservationBatch, key: Array, settings: FilterSettings
) -> ParticleFilterResult:
    """Run the particle filter over every player at once.

    Args:
        model: Jump-diffusion parameters.
        batch: Padded appearances, shape ``(players, T)``.
        key: JAX random key, split into one per player.
        settings: Particle count and recent-window length.

    Returns:
        Filter output with a leading player axis.
    """
    keys = jax.random.split(key, batch.mean.shape[0])
    series = PlayerSeries(
        gap_days=batch.gap_days,
        mean=adjusted_means(model.population, batch),
        sample_size=batch.sample_size,
        within_ss=batch.within_ss,
        mask=batch.mask,
    )
    result: ParticleFilterResult = jax.vmap(
        lambda one, one_key: particle_filter(model, one, one_key, settings)
    )(series, keys)
    return result
