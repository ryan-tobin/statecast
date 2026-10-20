"""Does a metric's alert warn of an arm injury beyond what workload already says?

Logistic regression of "arm IL placement within the horizon" on real-time
workload (pitch count and rest, each relative to the pitcher's own history),
with and without the metric's alert ``score`` (the ``signal`` term). Only
warning games and healthy games enter, as in ``injury_alerts.phase_sample``.

Two answers:

* **Inference.** The signal coefficient with cluster-robust (sandwich)
  standard errors, clustered by pitcher, because one pitcher's games are not
  independent.
* **Prediction.** Out-of-fold AUC of both models, with folds split by pitcher
  so no pitcher is in both training and test data.
"""

from typing import Final

import numpy as np
import polars as pl

from statecast.evaluation.injury_alerts import Z_95, weighted_auc

USAGE_TERMS: Final = ("pitches_z", "rest_z")
SIGNAL_TERM: Final = "signal"
MAX_ITERATIONS: Final = 50
TOLERANCE: Final = 1e-10
CV_FOLDS: Final = 5


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Logistic function, written to avoid overflow for large ``|x|``."""
    result: np.ndarray = np.exp(-np.logaddexp(0.0, -values))
    return result


def fit_logistic(design: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Maximum-likelihood logistic regression by Newton-Raphson (IRLS).

    Args:
        design: Covariates with an intercept column, shape ``(n, p)``.
        outcome: 0/1 outcomes, shape ``(n,)``.

    Returns:
        Coefficients, shape ``(p,)``.

    Raises:
        np.linalg.LinAlgError: If the information matrix is singular.

    Example:
        >>> design = np.column_stack([np.ones(8), np.repeat([-1.0, 1.0], 4)])
        >>> outcome = np.array([1, 0, 0, 0, 1, 1, 0, 0.0])
        >>> np.round(fit_logistic(design, outcome), 3).tolist()
        [-0.549, 0.549]
    """
    coefficients = np.zeros(design.shape[1])
    for _ in range(MAX_ITERATIONS):
        probability = _sigmoid(design @ coefficients)
        weight = probability * (1.0 - probability)
        information = design.T @ (design * weight[:, None])
        step = np.linalg.solve(information, design.T @ (outcome - probability))
        coefficients = coefficients + step
        if np.max(np.abs(step)) < TOLERANCE:
            break
    return coefficients


def cluster_robust_covariance(
    design: np.ndarray,
    outcome: np.ndarray,
    coefficients: np.ndarray,
    clusters: np.ndarray,
) -> np.ndarray:
    """Sandwich covariance clustered by pitcher.

    ``V = G/(G-1) * B^-1 (sum_g s_g s_g^T) B^-1``, where ``B`` is the
    information matrix and ``s_g`` is cluster ``g``'s summed score.

    Args:
        design: Covariates with an intercept column, shape ``(n, p)``.
        outcome: 0/1 outcomes, shape ``(n,)``.
        coefficients: Fitted coefficients, shape ``(p,)``.
        clusters: Integer cluster label per row, ``0..G-1``.

    Returns:
        Covariance matrix, shape ``(p, p)``.
    """
    probability = _sigmoid(design @ coefficients)
    weight = probability * (1.0 - probability)
    bread = np.linalg.inv(design.T @ (design * weight[:, None]))
    scores = design * (outcome - probability)[:, None]
    cluster_count = int(clusters.max()) + 1
    summed = np.zeros((cluster_count, design.shape[1]))
    np.add.at(summed, clusters, scores)
    meat = summed.T @ summed
    correction = cluster_count / (cluster_count - 1)
    covariance: np.ndarray = correction * bread @ meat @ bread
    return covariance


def _design(sample: pl.DataFrame, terms: tuple[str, ...]) -> np.ndarray:
    """Intercept plus the named columns."""
    columns = [np.ones(sample.height), *(sample[term].to_numpy() for term in terms)]
    return np.column_stack(columns)


def model_sample(labeled_sample: pl.DataFrame, usage: pl.DataFrame) -> pl.DataFrame:
    """Join workload scores onto a phase sample and drop incomplete rows.

    Args:
        labeled_sample: Output of ``phase_sample`` (``positive`` and
            ``score`` columns).
        usage: Output of ``relative_usage``.

    Returns:
        ``player_id``, ``cluster``, ``positive``, the usage terms and
        ``signal`` (the alert score), with no nulls.
    """
    return (
        labeled_sample.join(
            usage.select("player_id", "game_date", *USAGE_TERMS),
            on=["player_id", "game_date"],
            how="inner",
        )
        .with_columns(signal=pl.col("score"))
        .drop_nulls([*USAGE_TERMS, SIGNAL_TERM])
        .with_columns(cluster=pl.col("player_id").rank("dense").cast(pl.Int64) - 1)
        .select("player_id", "cluster", "positive", *USAGE_TERMS, SIGNAL_TERM)
    )


def coefficient_table(sample: pl.DataFrame) -> pl.DataFrame:
    """Fit usage + signal and report odds ratios with cluster-robust intervals.

    Args:
        sample: Output of ``model_sample``.

    Returns:
        ``term``, ``estimate``, ``std_error``, ``odds_ratio``, ``lower_95`` and
        ``upper_95`` (odds-ratio interval), one row per term. Odds ratios are
        per one standard deviation; above 1 means higher arm-injury risk.
    """
    terms = (*USAGE_TERMS, SIGNAL_TERM)
    design = _design(sample, terms)
    outcome = sample["positive"].cast(pl.Float64).to_numpy()
    coefficients = fit_logistic(design, outcome)
    covariance = cluster_robust_covariance(
        design, outcome, coefficients, sample["cluster"].to_numpy()
    )
    std_errors = np.sqrt(np.diag(covariance))
    return pl.DataFrame(
        {
            "term": ["intercept", *terms],
            "estimate": coefficients,
            "std_error": std_errors,
            "odds_ratio": np.exp(coefficients),
            "lower_95": np.exp(coefficients - Z_95 * std_errors),
            "upper_95": np.exp(coefficients + Z_95 * std_errors),
        }
    )


def cross_validated_auc(sample: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
    """Out-of-fold AUC of the usage-only and usage + signal models.

    Pitchers are split into ``CV_FOLDS`` folds; each fold is predicted by
    models fitted on the others, then one AUC is taken over every prediction.

    Args:
        sample: Output of ``model_sample``.
        seed: Random seed for the fold split.

    Returns:
        ``model`` and ``cv_auc`` for ``usage`` and ``usage + signal``.
    """
    clusters = sample["cluster"].to_numpy()
    cluster_count = int(clusters.max()) + 1
    fold_of_cluster = np.random.default_rng(seed).permutation(cluster_count) % CV_FOLDS
    folds = fold_of_cluster[clusters]
    outcome = sample["positive"].cast(pl.Float64).to_numpy()
    models = {"usage": USAGE_TERMS, "usage + signal": (*USAGE_TERMS, SIGNAL_TERM)}
    rows = []
    for name, terms in models.items():
        design = _design(sample, terms)
        predictions = np.zeros(sample.height)
        for fold in range(CV_FOLDS):
            test = folds == fold
            coefficients = fit_logistic(design[~test], outcome[~test])
            predictions[test] = design[test] @ coefficients
        auc = weighted_auc(predictions, outcome > 0.5, np.ones(sample.height))
        rows.append({"model": name, "cv_auc": auc})
    return pl.DataFrame(rows)
