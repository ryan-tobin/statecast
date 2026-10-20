import jax
import jax.numpy as jnp
import numpy as np

from statecast.evaluation.simulate import simulate_population
from statecast.models.kalman import PopulationParams

PARAMS = PopulationParams(
    mean_reversion=jnp.asarray(0.02),
    diffusion=jnp.asarray(0.06),
    observation_sd=jnp.asarray(1.0),
    population_mean=jnp.asarray(94.0),
    population_sd=jnp.asarray(1.5),
    day_effect_sd=jnp.asarray(0.5),
)


def test_simulate_population_shapes_and_sufficient_statistics() -> None:
    gaps = jnp.array([0.0, 5.0, 5.0])

    batch, talent = simulate_population(jax.random.key(1), PARAMS, 4, gaps, 12)

    assert talent.shape == batch.mean.shape == (4, 3)
    assert bool(jnp.all(batch.mask))
    assert bool(jnp.all(batch.sample_size == 12.0))
    assert bool(jnp.all(batch.within_ss > 0.0))
    np.testing.assert_allclose(batch.gap_days[2], gaps)


def test_simulated_talent_has_stationary_spread() -> None:
    gaps = jnp.array([0.0, 400.0])
    stationary_var = 0.06**2 / (2.0 * 0.02)

    _, talent = simulate_population(jax.random.key(2), PARAMS, 20_000, gaps, 2)

    expected_sd = np.sqrt(1.5**2 + stationary_var)
    np.testing.assert_allclose(talent.mean(axis=0), [94.0, 94.0], atol=0.05)
    np.testing.assert_allclose(talent.std(axis=0), [expected_sd] * 2, rtol=0.03)


def test_day_effect_adds_independent_game_to_game_variance() -> None:
    gaps = jnp.array([0.0, 0.0])
    pitches = 4

    batch, talent = simulate_population(
        jax.random.key(4), PARAMS, 20_000, gaps, pitches
    )

    game_noise = (batch.mean - talent)[:, 1] - (batch.mean - talent)[:, 0]
    expected_var = 2.0 * (0.5**2 + 1.0**2 / pitches)
    np.testing.assert_allclose(game_noise.var(), expected_var, rtol=0.04)
