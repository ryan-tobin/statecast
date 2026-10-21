"""Simulate synthetic players exactly from the generative models."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from statecast.models.dynamics import talent_transition
from statecast.models.jumps import JumpModel
from statecast.models.kalman import ObservationBatch, PopulationParams
from statecast.models.seasonal import seasonal_effect


class Schedule(NamedTuple):
    """When every simulated player appears.

    Attributes:
        gap_days: Days since the previous appearance, shape ``(T,)`` (first 0).
        season_days: Days since March 1 for each appearance, shape ``(T,)``.
    """

    gap_days: Array
    season_days: Array


class JumpSimulation(NamedTuple):
    """Simulated jump-diffusion players.

    Attributes:
        batch: Appearance sufficient statistics.
        talent: True (July 1 equivalent) talent, shape ``(players, T)``.
        jumped: True where a jump happened just before the appearance.
        jump_sizes: Size of that jump (0 where none), shape ``(players, T)``.
    """

    batch: ObservationBatch
    talent: Array
    jumped: Array
    jump_sizes: Array


def _observe(
    keys: tuple[Array, Array],
    params: PopulationParams,
    talent: Array,
    schedule: Schedule,
    pitches_per_appearance: int,
) -> ObservationBatch:
    """Throw every pitch around the talent path and reduce to statistics.

    Each appearance adds the league seasonal shift ``g(day)`` and a day effect
    ``N(0, tau^2)`` shared by its pitches; pitches are
    ``N(talent + g(day) + day effect, r^2)``.

    Args:
        keys: Keys for the day effects and for the pitches.
        params: True population parameters.
        talent: True talent, shape ``(players, T)``.
        schedule: Shared appearance schedule.
        pitches_per_appearance: Pitches thrown in every appearance.

    Returns:
        The batch of appearances.
    """
    day_key, pitch_key = keys
    shift = seasonal_effect(params.seasonal_knots, schedule.season_days)
    day_effects = params.day_effect_sd * jax.random.normal(day_key, talent.shape)
    pitches = (talent + shift + day_effects)[..., None] + params.observation_sd * (
        jax.random.normal(pitch_key, (*talent.shape, pitches_per_appearance))
    )
    means = pitches.mean(axis=-1)
    return ObservationBatch(
        gap_days=jnp.broadcast_to(schedule.gap_days, talent.shape),
        mean=means,
        sample_size=jnp.full(talent.shape, float(pitches_per_appearance)),
        within_ss=jnp.sum((pitches - means[..., None]) ** 2, axis=-1),
        mask=jnp.ones(talent.shape, dtype=bool),
        season_day=jnp.broadcast_to(schedule.season_days, talent.shape),
    )


def simulate_population(
    key: Array,
    params: PopulationParams,
    player_count: int,
    schedule: Schedule,
    pitches_per_appearance: int,
) -> tuple[ObservationBatch, Array]:
    """Draw players, their talent paths and every pitch they throw.

    Each player's level is ``N(m, s^2)``; talent starts at its stationary
    distribution around the level and moves by the exact OU transition over
    each gap. Pitches are reduced to sufficient statistics, so the simulator
    also exercises that reduction.

    Args:
        key: JAX random key.
        params: True population parameters.
        player_count: Players to simulate.
        schedule: Shared appearance schedule.
        pitches_per_appearance: Pitches thrown in every appearance.

    Returns:
        The batch of appearances and the true (July 1 equivalent) talent,
        shape ``(players, T)``.
    """
    level_key, start_key, path_key, day_key, pitch_key = jax.random.split(key, 5)
    levels = params.population_mean + params.population_sd * jax.random.normal(
        level_key, (player_count,)
    )
    stationary_sd = params.diffusion / jnp.sqrt(2.0 * params.mean_reversion)
    start_talent = levels + stationary_sd * jax.random.normal(
        start_key, (player_count,)
    )
    shocks = jax.random.normal(path_key, (schedule.gap_days.shape[0], player_count))

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

    _, talent_path = jax.lax.scan(step, start_talent, (schedule.gap_days, shocks))
    talent = talent_path.T
    batch = _observe(
        (day_key, pitch_key), params, talent, schedule, pitches_per_appearance
    )
    return batch, talent


def simulate_jump_population(
    key: Array,
    model: JumpModel,
    player_count: int,
    schedule: Schedule,
    pitches_per_appearance: int,
) -> JumpSimulation:
    """Draw jump-diffusion players exactly as the jump filters assume.

    Talent moves by the exact OU transition over each gap; then, with
    probability ``1 - exp(-lambda * gap)``, it jumps by
    ``N(mu_J, sigma_J^2)`` just before the appearance. A first appearance
    (gap 0) never has a jump.

    Args:
        key: JAX random key.
        model: True jump-diffusion parameters.
        player_count: Players to simulate.
        schedule: Shared appearance schedule.
        pitches_per_appearance: Pitches thrown in every appearance.

    Returns:
        The batch, true talent, where jumps happened and their sizes.
    """
    params, jumps = model
    keys = jax.random.split(key, 7)
    level_key, start_key, path_key, coin_key, size_key, day_key, pitch_key = keys
    levels = params.population_mean + params.population_sd * jax.random.normal(
        level_key, (player_count,)
    )
    stationary_sd = params.diffusion / jnp.sqrt(2.0 * params.mean_reversion)
    start_talent = levels + stationary_sd * jax.random.normal(
        start_key, (player_count,)
    )
    steps = schedule.gap_days.shape[0]
    shocks = jax.random.normal(path_key, (steps, player_count))
    coins = jax.random.uniform(coin_key, (steps, player_count))
    sizes = jumps.mean + jumps.sd * jax.random.normal(size_key, (steps, player_count))

    def step(
        talent: Array, inputs: tuple[Array, Array, Array, Array]
    ) -> tuple[Array, tuple[Array, Array, Array]]:
        gap, shock, coin, size = inputs
        transition, noise = talent_transition(
            params.mean_reversion, params.diffusion, gap
        )
        jumped = coin < -jnp.expm1(-jumps.rate * gap)
        jump_size = jnp.where(jumped, size, 0.0)
        next_talent = (
            transition[0, 0] * talent
            + transition[0, 1] * levels
            + jnp.sqrt(noise[0, 0]) * shock
            + jump_size
        )
        return next_talent, (next_talent, jumped, jump_size)

    _, (talent_path, jumped, jump_sizes) = jax.lax.scan(
        step, start_talent, (schedule.gap_days, shocks, coins, sizes)
    )
    talent = talent_path.T
    batch = _observe(
        (day_key, pitch_key), params, talent, schedule, pitches_per_appearance
    )
    return JumpSimulation(batch, talent, jumped.T, jump_sizes.T)
