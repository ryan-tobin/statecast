import jax
import jax.numpy as jnp
import numpy as np
import pytest

from statecast.evaluation.simulate import Schedule, simulate_population
from statecast.models.estimation import (
    FitResult,
    fit_population,
    initial_guess,
    summarize_fit,
)
from statecast.models.kalman import PopulationParams, unconstrain

TRUE_PARAMS = PopulationParams(
    mean_reversion=jnp.asarray(0.02),
    diffusion=jnp.asarray(0.05),
    observation_sd=jnp.asarray(1.0),
    population_mean=jnp.asarray(94.0),
    population_sd=jnp.asarray(1.5),
    day_effect_sd=jnp.asarray(0.5),
    seasonal_knots=jnp.array([-0.25, -0.15, -0.05, 0.05, 0.05, 0.0]),
)
STARTS_PER_SEASON = 36
ONE_SEASON = 28.0 + 5.0 * jnp.arange(STARTS_PER_SEASON)
CALENDAR_DAYS = jnp.concatenate([ONE_SEASON, ONE_SEASON + 365.0])
SCHEDULE = Schedule(
    gap_days=jnp.concatenate([jnp.zeros(1), jnp.diff(CALENDAR_DAYS)]),
    season_days=jnp.concatenate([ONE_SEASON, ONE_SEASON]),
)


@pytest.fixture(scope="module")
def recovered() -> FitResult:
    batch, _ = simulate_population(jax.random.key(0), TRUE_PARAMS, 400, SCHEDULE, 15)
    result: FitResult = jax.jit(fit_population)(batch, initial_guess(batch))
    return result


def test_fit_recovers_true_parameters_within_three_standard_errors(
    recovered: FitResult,
) -> None:
    standard_errors = jnp.sqrt(jnp.diag(recovered.unconstrained_cov))
    z_scores = (recovered.unconstrained - unconstrain(TRUE_PARAMS)) / standard_errors

    assert recovered.gradient_norm < 1e-6
    assert bool(jnp.all(jnp.abs(z_scores) < 3.0)), z_scores


def test_initial_guess_starts_near_pooled_moments() -> None:
    batch, _ = simulate_population(jax.random.key(3), TRUE_PARAMS, 200, SCHEDULE, 15)

    guess = initial_guess(batch)

    np.testing.assert_allclose(guess.population_mean, 94.0, atol=0.3)
    np.testing.assert_allclose(guess.observation_sd, 1.0, rtol=0.02)
    assert 0.0 < float(guess.day_effect_sd) < 1.0
    np.testing.assert_allclose(guess.seasonal_knots, np.zeros(6))


def test_summarize_fit_reports_derived_quantities(recovered: FitResult) -> None:
    summary = summarize_fit(recovered)
    rows = {row["parameter"]: row for row in summary.iter_rows(named=True)}

    kappa = float(recovered.params.mean_reversion)
    sigma = float(recovered.params.diffusion)
    np.testing.assert_allclose(rows["half_life_days"]["estimate"], np.log(2) / kappa)
    np.testing.assert_allclose(
        rows["stationary_sd"]["estimate"], sigma / np.sqrt(2 * kappa)
    )
    for row in rows.values():
        assert row["lower_95"] < row["estimate"] < row["upper_95"]


def test_summarize_fit_lists_seasonal_knots(recovered: FitResult) -> None:
    names = summarize_fit(recovered)["parameter"].to_list()

    assert names[6:12] == [
        "season_apr",
        "season_may",
        "season_jun",
        "season_aug",
        "season_sep",
        "season_oct",
    ]
