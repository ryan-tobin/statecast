from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from statecast.evaluation.usage_model import (
    cluster_robust_covariance,
    coefficient_table,
    cross_validated_auc,
    fit_logistic,
    model_sample,
)
from statecast.features.usage import MIN_HISTORY, relative_usage


def test_relative_usage_scores_against_earlier_games_only() -> None:
    rests = [0, 4, 5, 6, 4, 5, 6, 12]
    days = np.cumsum(rests).tolist()
    usage = pl.DataFrame(
        {
            "player_id": [3] * len(days),
            "game_date": [date(2026, 4, 1) + timedelta(days=d) for d in days],
            "pitches": [80, 90, 85, 95, 88, 92, 86, 40],
        }
    )

    scored = relative_usage(usage)

    prior_pitches = np.array([80, 90, 85, 95, 88, 92, 86], dtype=float)
    expected = (40 - prior_pitches.mean()) / prior_pitches.std(ddof=1)
    np.testing.assert_allclose(scored["pitches_z"][-1], expected)
    prior_rest = np.log1p(np.array(rests[1:-1], dtype=float))
    expected_rest = (np.log1p(12) - prior_rest.mean()) / prior_rest.std(ddof=1)
    np.testing.assert_allclose(scored["rest_z"][-1], expected_rest)
    assert scored["pitches_z"][:MIN_HISTORY].null_count() == MIN_HISTORY
    assert scored["rest_days"].to_list() == [None, *rests[1:]]


def test_relative_usage_treats_identical_history_as_no_spread() -> None:
    rests = [0, 5, 5, 5, 5, 5, 5, 5, 4]
    days = np.cumsum(rests).tolist()
    usage = pl.DataFrame(
        {
            "player_id": [1] * len(days),
            "game_date": [date(2026, 4, 1) + timedelta(days=d) for d in days],
            "pitches": [90] * len(days),
        }
    )

    scored = relative_usage(usage)

    assert scored["rest_z"].null_count() == len(days)
    assert scored["pitches_z"].null_count() == len(days)


def logistic_data(seed: int, players: int = 400) -> pl.DataFrame:
    """Games where workload and a velocity drop both raise injury odds."""
    rng = np.random.default_rng(seed)
    games = 40
    cluster = np.repeat(np.arange(players), games)
    pitches_z = rng.normal(size=cluster.size)
    rest_z = rng.normal(size=cluster.size)
    velocity_drop = rng.normal(size=cluster.size) + rng.normal(size=players)[cluster]
    logit = -3.0 - 0.3 * pitches_z + 0.2 * rest_z + 0.6 * velocity_drop
    positive = rng.random(cluster.size) < 1.0 / (1.0 + np.exp(-logit))
    return pl.DataFrame(
        {
            "player_id": cluster + 1000,
            "cluster": cluster,
            "positive": positive,
            "pitches_z": pitches_z,
            "rest_z": rest_z,
            "velocity_drop": velocity_drop,
        }
    )


def test_coefficient_table_recovers_true_odds_ratios() -> None:
    table = coefficient_table(logistic_data(seed=1))

    rows = {row["term"]: row for row in table.iter_rows(named=True)}
    for term, truth in (("pitches_z", -0.3), ("rest_z", 0.2), ("velocity_drop", 0.6)):
        assert rows[term]["lower_95"] < np.exp(truth) < rows[term]["upper_95"]
    assert rows["velocity_drop"]["odds_ratio"] > 1.5


def test_cluster_robust_covariance_ignores_duplicated_rows() -> None:
    rng = np.random.default_rng(2)
    design = np.column_stack([np.ones(300), rng.normal(size=300)])
    outcome = (rng.random(300) < 0.3).astype(float)
    clusters = np.repeat(np.arange(60), 5)
    coefficients = fit_logistic(design, outcome)
    doubled_design = np.vstack([design, design])
    doubled_outcome = np.concatenate([outcome, outcome])
    doubled_clusters = np.concatenate([clusters, clusters])

    original = cluster_robust_covariance(design, outcome, coefficients, clusters)
    doubled = cluster_robust_covariance(
        doubled_design, doubled_outcome, coefficients, doubled_clusters
    )

    np.testing.assert_allclose(doubled, original, rtol=1e-10)


def test_cross_validated_auc_rewards_the_informative_feature() -> None:
    table = cross_validated_auc(logistic_data(seed=3))

    auc = dict(table.iter_rows())
    assert auc["usage + velocity"] > auc["usage"] + 0.05
    assert 0.5 < auc["usage"] < 0.7


def test_model_sample_joins_usage_and_drops_incomplete_games() -> None:
    labeled = pl.DataFrame(
        {
            "player_id": [5, 5, 6],
            "game_date": [date(2026, 5, 1), date(2026, 5, 6), date(2026, 5, 2)],
            "alert_z": [-1.5, 0.5, 0.2],
            "positive": [True, False, False],
        }
    )
    usage = pl.DataFrame(
        {
            "player_id": [5, 5, 6],
            "game_date": [date(2026, 5, 1), date(2026, 5, 6), date(2026, 5, 2)],
            "pitches_z": [0.3, None, -0.4],
            "rest_z": [0.1, 0.2, 1.0],
        }
    )

    sample = model_sample(labeled, usage)

    assert sample["velocity_drop"].to_list() == [1.5, -0.2]
    assert sample["cluster"].to_list() == [0, 1]


def test_fit_logistic_rejects_a_singular_design() -> None:
    design = np.column_stack([np.ones(4), np.ones(4)])

    with pytest.raises(np.linalg.LinAlgError):
        fit_logistic(design, np.array([0.0, 1.0, 0.0, 1.0]))
