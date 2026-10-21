import jax
import jax.numpy as jnp
import numpy as np

from statecast.evaluation.simulate import (
    Schedule,
    simulate_jump_population,
    simulate_population,
)
from statecast.models.jumps import JumpModel, JumpParams
from statecast.models.kalman import PopulationParams
from statecast.models.seasonal import seasonal_effect

PARAMS = PopulationParams(
    mean_reversion=jnp.asarray(0.02),
    diffusion=jnp.asarray(0.06),
    observation_sd=jnp.asarray(1.0),
    population_mean=jnp.asarray(94.0),
    population_sd=jnp.asarray(1.5),
    day_effect_sd=jnp.asarray(0.5),
    seasonal_knots=jnp.array([-0.3, -0.2, -0.1, 0.1, 0.05, 0.0]),
)


def test_simulate_population_shapes_and_sufficient_statistics() -> None:
    schedule = Schedule(jnp.array([0.0, 5.0, 5.0]), jnp.array([40.0, 45.0, 50.0]))

    batch, talent = simulate_population(jax.random.key(1), PARAMS, 4, schedule, 12)

    assert talent.shape == batch.mean.shape == (4, 3)
    assert bool(jnp.all(batch.mask))
    assert bool(jnp.all(batch.sample_size == 12.0))
    assert bool(jnp.all(batch.within_ss > 0.0))
    np.testing.assert_allclose(batch.gap_days[2], schedule.gap_days)
    np.testing.assert_allclose(batch.season_day[2], schedule.season_days)


def test_simulated_talent_has_stationary_spread() -> None:
    schedule = Schedule(jnp.array([0.0, 400.0]), jnp.array([122.0, 122.0]))
    stationary_var = 0.06**2 / (2.0 * 0.02)

    _, talent = simulate_population(jax.random.key(2), PARAMS, 20_000, schedule, 2)

    expected_sd = np.sqrt(1.5**2 + stationary_var)
    np.testing.assert_allclose(talent.mean(axis=0), [94.0, 94.0], atol=0.05)
    np.testing.assert_allclose(talent.std(axis=0), [expected_sd] * 2, rtol=0.03)


def test_day_effect_adds_independent_game_to_game_variance() -> None:
    schedule = Schedule(jnp.array([0.0, 0.0]), jnp.array([122.0, 122.0]))
    pitches = 4

    batch, talent = simulate_population(
        jax.random.key(4), PARAMS, 20_000, schedule, pitches
    )

    game_noise = (batch.mean - talent)[:, 1] - (batch.mean - talent)[:, 0]
    expected_var = 2.0 * (0.5**2 + 1.0**2 / pitches)
    np.testing.assert_allclose(game_noise.var(), expected_var, rtol=0.04)


def test_seasonal_shift_moves_every_player_the_same_way() -> None:
    schedule = Schedule(jnp.array([0.0, 0.0]), jnp.array([31.0, 122.0]))

    batch, talent = simulate_population(jax.random.key(5), PARAMS, 20_000, schedule, 25)

    gap = (batch.mean - talent).mean(axis=0)
    expected = seasonal_effect(PARAMS.seasonal_knots, schedule.season_days)
    np.testing.assert_allclose(gap, expected, atol=0.02)


JUMP_MODEL = JumpModel(
    population=PARAMS,
    jumps=JumpParams(
        rate=jnp.asarray(0.02), mean=jnp.asarray(-1.0), sd=jnp.asarray(0.5)
    ),
)


def test_jumps_arrive_at_the_poisson_rate_and_never_on_a_first_appearance() -> None:
    schedule = Schedule(jnp.array([0.0, 10.0, 30.0]), jnp.array([40.0, 50.0, 80.0]))

    simulation = simulate_jump_population(
        jax.random.key(6), JUMP_MODEL, 20_000, schedule, 4
    )

    frequency = np.asarray(simulation.jumped).mean(axis=0)
    expected = -np.expm1(-0.02 * np.array([0.0, 10.0, 30.0]))
    np.testing.assert_allclose(frequency, expected, atol=0.01)
    assert not bool(jnp.any(simulation.jump_sizes[~simulation.jumped]))


def test_jumps_move_talent_by_their_size() -> None:
    schedule = Schedule(jnp.array([0.0, 30.0]), jnp.array([40.0, 70.0]))

    simulation = simulate_jump_population(
        jax.random.key(7), JUMP_MODEL, 20_000, schedule, 4
    )

    jumped = np.asarray(simulation.jumped[:, 1])
    change = np.asarray(simulation.talent[:, 1] - simulation.talent[:, 0])
    sizes = np.asarray(simulation.jump_sizes[jumped, 1])
    np.testing.assert_allclose(sizes.mean(), -1.0, atol=0.05)
    np.testing.assert_allclose(sizes.std(), 0.5, atol=0.05)
    np.testing.assert_allclose(
        change[jumped].mean() - change[~jumped].mean(), -1.0, atol=0.06
    )
