import jax
import jax.numpy as jnp
import numpy as np
import pytest

from statecast.evaluation.simulate import (
    JumpSimulation,
    Schedule,
    simulate_jump_population,
)
from statecast.models.estimation import FitResult, fit_population, initial_guess
from statecast.models.jumps import (
    JUMP_MEAN_INDEX,
    OU_SIZE,
    JumpFitResult,
    JumpModel,
    JumpParams,
    assumed_density_filter,
    constrain_jump_model,
    fit_jump_population,
    initial_jump_model,
    jump_optimizer_coordinates,
    jump_population_log_likelihood,
    kalman_update,
    log_jump_odds,
    summarize_jump_fit,
    unconstrain_jump_model,
)
from statecast.models.kalman import (
    PopulationParams,
    initial_state,
    kalman_filter,
    population_log_likelihood,
)

POPULATION = PopulationParams(
    mean_reversion=jnp.asarray(0.02),
    diffusion=jnp.asarray(0.05),
    observation_sd=jnp.asarray(1.0),
    population_mean=jnp.asarray(94.0),
    population_sd=jnp.asarray(1.5),
    day_effect_sd=jnp.asarray(0.5),
    seasonal_knots=jnp.array([-0.25, -0.15, -0.05, 0.05, 0.05, 0.0]),
)
TRUE_MODEL = JumpModel(
    population=POPULATION,
    jumps=JumpParams(
        rate=jnp.asarray(0.01), mean=jnp.asarray(-1.0), sd=jnp.asarray(1.5)
    ),
)
STARTS_PER_SEASON = 36
ONE_SEASON = 28.0 + 5.0 * jnp.arange(STARTS_PER_SEASON)
CALENDAR_DAYS = jnp.concatenate([ONE_SEASON, ONE_SEASON + 365.0])
SCHEDULE = Schedule(
    gap_days=jnp.concatenate([jnp.zeros(1), jnp.diff(CALENDAR_DAYS)]),
    season_days=jnp.concatenate([ONE_SEASON, ONE_SEASON]),
)


def one_player(simulation: JumpSimulation, row: int = 0) -> tuple[jax.Array, ...]:
    """Inputs of the single-player filters for one simulated player."""
    batch = simulation.batch
    return (
        batch.gap_days[row],
        batch.mean[row],
        batch.sample_size[row],
        batch.within_ss[row],
        batch.mask[row],
    )


@pytest.fixture(scope="module")
def simulation() -> JumpSimulation:
    return simulate_jump_population(jax.random.key(2), TRUE_MODEL, 300, SCHEDULE, 15)


@pytest.fixture(scope="module")
def fits(simulation: JumpSimulation) -> tuple[FitResult, JumpFitResult]:
    batch = simulation.batch
    ou_fit: FitResult = jax.jit(fit_population)(batch, initial_guess(batch))
    jump_fit: JumpFitResult = jax.jit(fit_jump_population)(
        batch, initial_jump_model(ou_fit.params)
    )
    return ou_fit, jump_fit


def test_log_jump_odds_follow_a_poisson_process() -> None:
    gaps = jnp.array([0.0, 1.0, 5.0, 150.0])

    log_no_jump, log_jump = log_jump_odds(jnp.asarray(0.01), gaps)

    np.testing.assert_allclose(jnp.exp(log_no_jump), np.exp(-0.01 * gaps))
    np.testing.assert_allclose(jnp.exp(log_jump), -np.expm1(-0.01 * gaps))
    assert float(log_jump[0]) == -np.inf


def test_log_jump_odds_have_a_finite_gradient_at_a_first_appearance() -> None:
    def total(rate: jax.Array) -> jax.Array:
        log_no_jump, log_jump = log_jump_odds(rate, jnp.array([0.0, 5.0]))
        return jnp.sum(jnp.logaddexp(log_no_jump, log_jump + 1.0))

    gradient = jax.grad(total)(jnp.asarray(0.01))

    assert bool(jnp.isfinite(gradient))


def test_kalman_update_matches_the_scalar_formulas() -> None:
    mean = jnp.array([94.0, 93.0])
    cov = jnp.array([[2.0, 1.0], [1.0, 1.5]])

    result = kalman_update(mean, cov, jnp.asarray(95.0), jnp.asarray(0.5))

    innovation_var = 2.5
    np.testing.assert_allclose(result.mean, [94.0 + 2.0 / 2.5, 93.0 + 1.0 / 2.5])
    np.testing.assert_allclose(result.cov[0, 0], 2.0 - 4.0 / innovation_var)
    np.testing.assert_allclose(result.cov[1, 1], 1.5 - 1.0 / innovation_var)
    expected_log_likelihood = -0.5 * (
        np.log(2 * np.pi * innovation_var) + 1.0 / innovation_var
    )
    np.testing.assert_allclose(result.log_likelihood, expected_log_likelihood)


def test_zero_jump_rate_reduces_to_the_kalman_filter(
    simulation: JumpSimulation,
) -> None:
    no_jumps = TRUE_MODEL._replace(
        jumps=TRUE_MODEL.jumps._replace(rate=jnp.asarray(0.0))
    )
    inputs = one_player(simulation)

    jump_result = assumed_density_filter(no_jumps, *inputs)
    kalman_result = kalman_filter(POPULATION, *inputs)

    np.testing.assert_allclose(jump_result.log_likelihood, kalman_result.log_likelihood)
    np.testing.assert_allclose(jump_result.filtered_means, kalman_result.filtered_means)
    np.testing.assert_allclose(
        jump_result.filtered_covs, kalman_result.filtered_covs, atol=1e-12
    )
    np.testing.assert_allclose(jump_result.jump_probs, 0.0)
    np.testing.assert_allclose(
        jump_population_log_likelihood(no_jumps, simulation.batch),
        population_log_likelihood(POPULATION, simulation.batch),
    )


def test_second_appearance_jump_probability_is_bayes_rule(
    simulation: JumpSimulation,
) -> None:
    gap, mean, size, _, _ = one_player(simulation)
    measurement_var = 0.5**2 + 1.0 / size[0]
    prior_mean, prior_cov = initial_state(POPULATION)
    first = kalman_update(prior_mean, prior_cov, mean[0], measurement_var)
    decay = np.exp(-0.02 * float(gap[1]))
    transition = jnp.array([[decay, 1.0 - decay], [0.0, 1.0]])
    noise_var = 0.05**2 * (1.0 - decay**2) / (2.0 * 0.02)
    predicted_mean = transition @ first.mean
    predicted_var = float((transition @ first.cov @ transition.T)[0, 0]) + noise_var
    second_var = 0.5**2 + 1.0 / size[1]
    still_density = np.exp(
        -0.5 * (mean[1] - predicted_mean[0]) ** 2 / (predicted_var + second_var)
    ) / np.sqrt(predicted_var + second_var)
    jump_var = predicted_var + 1.5**2 + second_var
    jump_density = np.exp(
        -0.5 * (mean[1] - predicted_mean[0] + 1.0) ** 2 / jump_var
    ) / np.sqrt(jump_var)
    prior = -np.expm1(-0.01 * float(gap[1]))

    result = assumed_density_filter(TRUE_MODEL, *one_player(simulation))

    expected = (
        prior * jump_density / (prior * jump_density + (1 - prior) * still_density)
    )
    np.testing.assert_allclose(result.jump_probs[1], expected, rtol=1e-10)
    assert float(result.jump_probs[0]) == 0.0


def test_constrain_round_trips() -> None:
    raw = unconstrain_jump_model(TRUE_MODEL)

    restored = constrain_jump_model(raw)

    assert raw.shape == (OU_SIZE + 3,)
    np.testing.assert_allclose(restored.jumps.rate, 0.01)
    np.testing.assert_allclose(restored.jumps.mean, -1.0)
    np.testing.assert_allclose(restored.jumps.sd, 1.5)
    np.testing.assert_allclose(restored.population.diffusion, 0.05)


def test_initial_jump_model_starts_with_rare_sizeable_jumps() -> None:
    model = initial_jump_model(POPULATION)

    swing = np.sqrt(0.5**2 + 0.05**2 / (2 * 0.02))
    np.testing.assert_allclose(model.jumps.rate, 1 / 180)
    np.testing.assert_allclose(model.jumps.mean, 0.0)
    np.testing.assert_allclose(model.jumps.sd, 2 * swing)
    assert model.population is POPULATION


def test_jump_optimizer_coordinates_scale_the_jump_mean_only(
    simulation: JumpSimulation,
) -> None:
    shift, scale = (
        np.asarray(part) for part in jump_optimizer_coordinates(simulation.batch)
    )

    assert shift.shape == scale.shape == (OU_SIZE + 3,)
    np.testing.assert_allclose(shift[OU_SIZE:], 0.0)
    np.testing.assert_allclose(scale[JUMP_MEAN_INDEX], scale[3])
    np.testing.assert_allclose(scale[[OU_SIZE, OU_SIZE + 2]], 1.0)


def test_fit_recovers_jump_parameters_within_three_standard_errors(
    fits: tuple[FitResult, JumpFitResult],
) -> None:
    _, jump_fit = fits
    standard_errors = jnp.sqrt(jnp.diag(jump_fit.unconstrained_cov))
    z_scores = (
        jump_fit.unconstrained - unconstrain_jump_model(TRUE_MODEL)
    ) / standard_errors

    assert jump_fit.gradient_norm < 1e-6
    assert bool(jnp.all(jnp.abs(z_scores) < 3.0)), z_scores


def test_jumps_improve_the_likelihood_of_jump_data(
    fits: tuple[FitResult, JumpFitResult],
) -> None:
    ou_fit, jump_fit = fits

    assert float(jump_fit.log_likelihood - ou_fit.log_likelihood) > 100.0


def test_summarize_jump_fit_reports_days_between_jumps(
    fits: tuple[FitResult, JumpFitResult],
) -> None:
    _, jump_fit = fits

    summary = summarize_jump_fit(jump_fit)

    rows = {row["parameter"]: row for row in summary.iter_rows(named=True)}
    assert list(rows)[OU_SIZE : OU_SIZE + 3] == ["jump_rate", "jump_mean", "jump_sd"]
    np.testing.assert_allclose(
        rows["days_between_jumps"]["estimate"], 1.0 / rows["jump_rate"]["estimate"]
    )
    for name in ("jump_rate", "jump_sd", "days_between_jumps", "half_life_days"):
        assert rows[name]["lower_95"] < rows[name]["estimate"] < rows[name]["upper_95"]
