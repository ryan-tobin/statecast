from datetime import date

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from jax import Array
from jax.scipy.stats import multivariate_normal

import statecast.models  # noqa: F401  (enables 64-bit floats)
from statecast.models.kalman import (
    FilterResult,
    ObservationBatch,
    PopulationParams,
    batch_from_frame,
    constrain,
    initial_state,
    kalman_filter,
    population_log_likelihood,
    rts_smoother,
    unconstrain,
)

PARAMS = PopulationParams(
    mean_reversion=jnp.asarray(0.05),
    diffusion=jnp.asarray(0.3),
    observation_sd=jnp.asarray(1.1),
    population_mean=jnp.asarray(94.0),
    population_sd=jnp.asarray(1.5),
    day_effect_sd=jnp.asarray(0.6),
)
GAP_DAYS = jnp.array([0.0, 4.0, 1.0, 120.0])
PITCHES = (
    jnp.array([95.1, 94.2, 96.0]),
    jnp.array([93.8, 94.9]),
    jnp.array([95.5, 95.0, 94.1, 94.7]),
    jnp.array([92.9, 93.4, 93.1]),
)


def summaries(pitches: tuple[Array, ...]) -> tuple[Array, Array, Array]:
    means = jnp.array([group.mean() for group in pitches])
    sizes = jnp.array([float(group.size) for group in pitches])
    within = jnp.array([jnp.sum((group - group.mean()) ** 2) for group in pitches])
    return means, sizes, within


def talent_covariance(params: PopulationParams, gap_days: Array) -> Array:
    """Cov(theta_j, theta_k) = s^2 + sigma^2/(2 kappa) exp(-kappa |t_j - t_k|)."""
    times = jnp.cumsum(gap_days)
    stationary = params.diffusion**2 / (2.0 * params.mean_reversion)
    lags = jnp.abs(times[:, None] - times[None, :])
    return params.population_sd**2 + stationary * jnp.exp(-params.mean_reversion * lags)


def pitch_covariance(params: PopulationParams) -> tuple[Array, Array]:
    """Joint covariance of every raw pitch, and each pitch's appearance index.

    Pitches share talent across appearances, share the day effect within an
    appearance, and add independent pitch noise.
    """
    owner = jnp.concatenate(
        [jnp.full(group.size, index) for index, group in enumerate(PITCHES)]
    )
    talent = talent_covariance(params, GAP_DAYS)
    same_game = (owner[:, None] == owner[None, :]).astype(float)
    cov = (
        talent[owner][:, owner]
        + params.day_effect_sd**2 * same_game
        + params.observation_sd**2 * jnp.eye(owner.size)
    )
    return cov, owner


def run_filter() -> tuple[ObservationBatch, FilterResult]:
    means, sizes, within = summaries(PITCHES)
    mask = jnp.ones(len(PITCHES), dtype=bool)
    batch = ObservationBatch(GAP_DAYS, means, sizes, within, mask)
    return batch, kalman_filter(PARAMS, *batch)


def test_constrain_round_trips() -> None:
    np.testing.assert_allclose(
        unconstrain(constrain(unconstrain(PARAMS))), unconstrain(PARAMS)
    )


def test_initial_state_covariance_adds_level_and_stationary_spread() -> None:
    mean, cov = initial_state(PARAMS)

    level = 1.5**2
    stationary = 0.3**2 / (2 * 0.05)
    np.testing.assert_allclose(mean, [94.0, 94.0])
    np.testing.assert_allclose(cov, [[level + stationary, level], [level, level]])


def test_filter_likelihood_equals_brute_force_over_every_pitch() -> None:
    _, result = run_filter()
    cov, _ = pitch_covariance(PARAMS)
    all_pitches = jnp.concatenate(PITCHES)

    expected = multivariate_normal.logpdf(
        all_pitches, jnp.full(all_pitches.size, 94.0), cov
    )

    np.testing.assert_allclose(result.log_likelihood, expected, rtol=1e-10)


def test_smoother_matches_exact_gaussian_conditioning() -> None:
    _, result = run_filter()
    cov, owner = pitch_covariance(PARAMS)
    talent = talent_covariance(PARAMS, GAP_DAYS)
    all_pitches = jnp.concatenate(PITCHES)
    cross = talent[:, owner]

    exact_mean = 94.0 + cross @ jnp.linalg.solve(cov, all_pitches - 94.0)
    exact_var = jnp.diag(talent - cross @ jnp.linalg.solve(cov, cross.T))
    smoothed = rts_smoother(result)

    np.testing.assert_allclose(smoothed.means[:, 0], exact_mean, rtol=1e-10)
    np.testing.assert_allclose(smoothed.covs[:, 0, 0], exact_var, rtol=1e-8)


def test_padding_changes_nothing() -> None:
    batch, result = run_filter()
    padded = ObservationBatch(
        gap_days=jnp.concatenate([batch.gap_days, jnp.zeros(3)]),
        mean=jnp.concatenate([batch.mean, jnp.zeros(3)]),
        sample_size=jnp.concatenate([batch.sample_size, jnp.ones(3)]),
        within_ss=jnp.concatenate([batch.within_ss, jnp.zeros(3)]),
        mask=jnp.concatenate([batch.mask, jnp.zeros(3, dtype=bool)]),
    )

    padded_result = kalman_filter(PARAMS, *padded)
    smoothed = rts_smoother(result)
    padded_smoothed = rts_smoother(padded_result)

    np.testing.assert_allclose(padded_result.log_likelihood, result.log_likelihood)
    np.testing.assert_allclose(padded_smoothed.means[:4], smoothed.means, rtol=1e-10)


def test_population_log_likelihood_sums_players() -> None:
    batch, result = run_filter()
    stacked = jax.tree.map(lambda leaf: jnp.stack([leaf, leaf]), batch)

    total = population_log_likelihood(PARAMS, stacked)

    np.testing.assert_allclose(total, 2.0 * result.log_likelihood)


def test_batch_from_frame_pads_and_measures_gaps_across_seasons() -> None:
    frame = pl.DataFrame(
        {
            "player_id": [5, 5, 5, 8],
            "game_date": [
                date(2025, 9, 28),
                date(2026, 3, 27),
                date(2026, 4, 1),
                date(2026, 4, 2),
            ],
            "sample_size": [90, 88, 92, 15],
            "mean": [94.0, 94.5, 94.2, 97.0],
            "within_ss": [70.0, 65.0, 71.0, 9.0],
        }
    )

    batch, player_ids = batch_from_frame(frame)

    assert player_ids.tolist() == [5, 8]
    assert batch.gap_days.tolist() == [[0.0, 180.0, 5.0], [0.0, 0.0, 0.0]]
    assert batch.mask.tolist() == [[True, True, True], [True, False, False]]
    assert batch.sample_size[1].tolist() == [15.0, 1.0, 1.0]


def test_day_effect_inflates_measurement_variance() -> None:
    batch, result = run_filter()
    no_day_effect = PARAMS._replace(day_effect_sd=jnp.asarray(1e-9))

    quiet = kalman_filter(no_day_effect, *batch)

    first_prior_var = initial_state(PARAMS)[1][0, 0]
    first_measurement_var = 0.6**2 + 1.1**2 / 3.0
    expected_var = (
        first_prior_var
        * first_measurement_var
        / (first_prior_var + first_measurement_var)
    )
    np.testing.assert_allclose(result.filtered_covs[0, 0, 0], expected_var)
    assert float(quiet.filtered_covs[0, 0, 0]) < float(result.filtered_covs[0, 0, 0])
