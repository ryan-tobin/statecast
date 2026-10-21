import jax
import jax.numpy as jnp
import numpy as np
import pytest

from statecast.evaluation.injury_alerts import weighted_auc
from statecast.evaluation.simulate import (
    JumpSimulation,
    Schedule,
    simulate_jump_population,
)
from statecast.models.jumps import JumpModel, JumpParams, assumed_density_filter
from statecast.models.kalman import PopulationParams, adjusted_means, kalman_filter
from statecast.models.rbpf import (
    FilterSettings,
    ParticleFilterResult,
    PlayerSeries,
    filter_jump_population,
    particle_filter,
    systematic_resample,
)

POPULATION = PopulationParams(
    mean_reversion=jnp.asarray(0.007),
    diffusion=jnp.asarray(0.095),
    observation_sd=jnp.asarray(0.94),
    population_mean=jnp.asarray(94.6),
    population_sd=jnp.asarray(2.2),
    day_effect_sd=jnp.asarray(0.5),
    seasonal_knots=jnp.array([-0.4, -0.3, -0.1, -0.1, -0.1, -0.15]),
)
MODEL = JumpModel(
    population=POPULATION,
    jumps=JumpParams(
        rate=jnp.asarray(1 / 120), mean=jnp.asarray(-1.0), sd=jnp.asarray(1.5)
    ),
)
NO_JUMPS = MODEL._replace(jumps=MODEL.jumps._replace(rate=jnp.asarray(0.0)))
ONE_SEASON = 28.0 + 5.0 * jnp.arange(36)
CALENDAR_DAYS = jnp.concatenate([ONE_SEASON, ONE_SEASON + 365.0])
SCHEDULE = Schedule(
    gap_days=jnp.concatenate([jnp.zeros(1), jnp.diff(CALENDAR_DAYS)]),
    season_days=jnp.concatenate([ONE_SEASON, ONE_SEASON]),
)
SMALL = FilterSettings(particle_count=64, lag=3)
run_filter = jax.jit(particle_filter, static_argnames="settings")
run_population = jax.jit(filter_jump_population, static_argnames="settings")


def series_for(simulation: JumpSimulation, row: int = 0) -> PlayerSeries:
    """One simulated player's season-adjusted appearances."""
    batch = simulation.batch
    return PlayerSeries(
        gap_days=batch.gap_days[row],
        mean=adjusted_means(POPULATION, batch)[row],
        sample_size=batch.sample_size[row],
        within_ss=batch.within_ss[row],
        mask=batch.mask[row],
    )


def recent_truth(
    jumped: np.ndarray, sizes: np.ndarray, lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """Whether a jump hit in the last ``lag`` appearances, and its latest size."""
    recent = np.zeros_like(jumped)
    latest = np.zeros_like(sizes)
    for back in reversed(range(lag)):
        shifted_jump = np.zeros_like(jumped)
        shifted_size = np.zeros_like(sizes)
        shifted_jump[:, back:] = jumped[:, : jumped.shape[1] - back]
        shifted_size[:, back:] = sizes[:, : sizes.shape[1] - back]
        recent |= shifted_jump
        latest = np.where(shifted_jump, shifted_size, latest)
    return recent, latest


@pytest.fixture(scope="module")
def simulation() -> JumpSimulation:
    return simulate_jump_population(jax.random.key(1), MODEL, 300, SCHEDULE, 15)


@pytest.fixture(scope="module")
def population_result(simulation: JumpSimulation) -> ParticleFilterResult:
    result: ParticleFilterResult = run_population(
        MODEL, simulation.batch, jax.random.key(5), FilterSettings(512, 3)
    )
    return result


def test_systematic_resample_copies_each_particle_in_proportion() -> None:
    weights = jnp.array([0.5, 0.0, 0.3, 0.2])

    indices = systematic_resample(jax.random.key(3), weights)

    counts = np.bincount(np.asarray(indices), minlength=4)
    assert counts.sum() == 4
    assert counts[1] == 0
    assert counts[0] == 2
    for count, weight in zip(counts, np.asarray(weights), strict=True):
        assert np.floor(4 * weight) <= count <= np.ceil(4 * weight)


def test_zero_jump_rate_reproduces_the_kalman_filter(
    simulation: JumpSimulation,
) -> None:
    series = series_for(simulation)

    result = run_filter(NO_JUMPS, series, jax.random.key(0), SMALL)

    kalman = kalman_filter(POPULATION, *series)
    np.testing.assert_allclose(
        result.talent_mean, kalman.filtered_means[:, 0], atol=1e-10
    )
    np.testing.assert_allclose(
        result.talent_sd, np.sqrt(kalman.filtered_covs[:, 0, 0]), atol=1e-9
    )
    np.testing.assert_allclose(result.log_likelihood, kalman.log_likelihood)
    np.testing.assert_allclose(result.jump_prob, 0.0)
    np.testing.assert_allclose(result.effective_size, 64.0)


def test_second_appearance_matches_the_exact_filter(
    simulation: JumpSimulation,
) -> None:
    series = series_for(simulation, row=3)

    result = run_filter(MODEL, series, jax.random.key(0), SMALL)

    exact = assumed_density_filter(MODEL, *series)
    np.testing.assert_allclose(result.jump_prob[:2], exact.jump_probs[:2], rtol=1e-10)
    np.testing.assert_allclose(
        result.talent_mean[:2], exact.filtered_means[:2, 0], rtol=1e-12
    )


def test_padding_is_nan_and_adds_nothing(simulation: JumpSimulation) -> None:
    series = series_for(simulation)
    padded = PlayerSeries(
        gap_days=jnp.concatenate([series.gap_days, jnp.zeros(4)]),
        mean=jnp.concatenate([series.mean, jnp.zeros(4)]),
        sample_size=jnp.concatenate([series.sample_size, jnp.ones(4)]),
        within_ss=jnp.concatenate([series.within_ss, jnp.zeros(4)]),
        mask=jnp.concatenate([series.mask, jnp.zeros(4, dtype=bool)]),
    )

    plain = run_filter(MODEL, series, jax.random.key(0), SMALL)
    result = run_filter(MODEL, padded, jax.random.key(0), SMALL)

    assert bool(jnp.all(jnp.isnan(result.jump_prob[-4:])))
    np.testing.assert_allclose(result.jump_prob[:-4], plain.jump_prob)
    np.testing.assert_allclose(result.log_likelihood, plain.log_likelihood)


def test_lag_one_recent_window_is_the_current_appearance(
    simulation: JumpSimulation,
) -> None:
    series = series_for(simulation, row=7)

    result = run_filter(MODEL, series, jax.random.key(0), FilterSettings(64, 1))

    np.testing.assert_allclose(result.recent_jump_prob, result.jump_prob)


def test_jump_probabilities_are_calibrated(
    simulation: JumpSimulation, population_result: ParticleFilterResult
) -> None:
    jump_prob = np.asarray(population_result.jump_prob)
    recent, latest = recent_truth(
        np.asarray(simulation.jumped), np.asarray(simulation.jump_sizes), lag=3
    )

    np.testing.assert_allclose(
        jump_prob.mean(), np.asarray(simulation.jumped).mean(), rtol=0.1
    )
    np.testing.assert_allclose(
        np.asarray(population_result.recent_jump_prob).mean(), recent.mean(), rtol=0.1
    )
    np.testing.assert_allclose(
        np.asarray(population_result.recent_drop_prob).mean(),
        (recent & (latest < 0)).mean(),
        rtol=0.1,
    )


def test_recent_jumps_are_detected(
    simulation: JumpSimulation, population_result: ParticleFilterResult
) -> None:
    recent, latest = recent_truth(
        np.asarray(simulation.jumped), np.asarray(simulation.jump_sizes), lag=3
    )
    drops = recent & (latest < 0)
    recent_prob = np.asarray(population_result.recent_jump_prob).ravel()
    drop_prob = np.asarray(population_result.recent_drop_prob).ravel()
    weights = np.ones(recent_prob.size)

    assert weighted_auc(recent_prob, recent.ravel(), weights) > 0.8
    assert weighted_auc(drop_prob, drops.ravel(), weights) > 0.85


def test_probabilities_are_coherent(population_result: ParticleFilterResult) -> None:
    jump_prob = np.asarray(population_result.jump_prob)
    recent = np.asarray(population_result.recent_jump_prob)
    drop = np.asarray(population_result.recent_drop_prob)

    assert np.all((jump_prob >= 0.0) & (jump_prob <= 1.0))
    assert np.all(recent >= jump_prob - 1e-12)
    assert np.all(drop <= recent + 1e-12)
    assert float(np.min(np.asarray(population_result.effective_size))) > 1.0
