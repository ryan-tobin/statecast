"""Simulate synthetic players exactly from the generative model."""

import jax
import jax.numpy as jnp
from jax import Array

from statecast.models.dynamics import talent_transition
from statecast.models.kalman import ObservationBatch, PopulationParams


def simulate_population(
    key: Array,
    params: PopulationParams,
    player_count: int,
    gap_days: Array,
    pitches_per_appearance: int,
) -> tuple[ObservationBatch, Array]:
    """Draw players, their talent paths and every pitch they throw.

    Each player's level is ``N(m, s^2)``; talent starts at its stationary
    distribution around the level and moves by the exact OU transition over
    each gap. Each appearance draws a day effect ``N(0, tau^2)`` shared by its
    pitches, and pitches are ``N(talent + day effect, r^2)``. Pitches are
    reduced to sufficient statistics, so the simulator also exercises that
    reduction.

    Args:
        key: JAX random key.
        params: True population parameters.
        player_count: Players to simulate.
        gap_days: Shared schedule of gaps, shape ``(T,)`` (first entry 0).
        pitches_per_appearance: Pitches thrown in every appearance.

    Returns:
        The batch of appearances and the true talent, shape ``(players, T)``.
    """
    level_key, start_key, path_key, day_key, pitch_key = jax.random.split(key, 5)
    levels = params.population_mean + params.population_sd * jax.random.normal(
        level_key, (player_count,)
    )
    stationary_sd = params.diffusion / jnp.sqrt(2.0 * params.mean_reversion)
    start_talent = levels + stationary_sd * jax.random.normal(
        start_key, (player_count,)
    )
    shocks = jax.random.normal(path_key, (gap_days.shape[0], player_count))

    def step(talent: Array, inputs: tuple[Array, Array]) -> tuple[Array, Array]:
        gap, shock = inputs
        transition, noise = talent_transition(
            params.mean_reversion, params.diffusion, gap
        )
        next_talent = (
            transition[0, 0] * talent
            + transition[0, 1] * levels
            + jnp.sqrt(noise[0, 0]) * shock
        )
        return next_talent, next_talent

    _, talent_path = jax.lax.scan(step, start_talent, (gap_days, shocks))
    talent = talent_path.T
    day_effects = params.day_effect_sd * jax.random.normal(day_key, talent.shape)
    pitches = (talent + day_effects)[..., None] + params.observation_sd * (
        jax.random.normal(pitch_key, (*talent.shape, pitches_per_appearance))
    )
    means = pitches.mean(axis=-1)
    batch = ObservationBatch(
        gap_days=jnp.broadcast_to(gap_days, talent.shape),
        mean=means,
        sample_size=jnp.full(talent.shape, float(pitches_per_appearance)),
        within_ss=jnp.sum((pitches - means[..., None]) ** 2, axis=-1),
        mask=jnp.ones(talent.shape, dtype=bool),
    )
    return batch, talent
