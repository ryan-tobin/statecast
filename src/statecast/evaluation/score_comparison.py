"""Do jump-based alerts warn of IL placements better than the smooth alert?

Several alert scores are compared on the same appearances, labeled as in
``injury_alerts``: positives are games within the horizon before a placement,
split into warning games and the final game; negatives are healthy games.

For each score this reports AUCs against healthy games for any placement, for
arm placements and for the lower-body placebo, and the arm-minus-lower-body
contrast. Each later score is then compared with the first (the baseline).
Every number is recomputed on the same pitcher-cluster bootstrap draws, so
the differences between scores are paired: the shared variation in which
pitchers were drawn cancels.
"""

from collections.abc import Sequence
from statistics import NormalDist
from typing import Final, NamedTuple

import numpy as np
import polars as pl

from statecast.evaluation.injury_alerts import (
    BOOTSTRAP_REPS,
    PHASES,
    PLACEBO_GROUP,
    TARGET_GROUP,
    binned_auc,
    bootstrap_summary,
)
from statecast.evaluation.metric_scan import holm_adjust

ANY_GROUP: Final = "any"
CONTRAST: Final = f"{TARGET_GROUP} - {PLACEBO_GROUP}"
STATISTICS: Final = (ANY_GROUP, TARGET_GROUP, PLACEBO_GROUP, CONTRAST)
PROBABILITY_FLOOR: Final = 1e-12
COMPARISON_SCHEMA: Final = {
    "score": pl.String,
    "statistic": pl.String,
    "phase": pl.String,
    "estimate": pl.Float64,
    "lower_95": pl.Float64,
    "upper_95": pl.Float64,
    "p_value": pl.Float64,
}


class ScoreComparison(NamedTuple):
    """Every score's AUCs and its gains over the baseline score.

    Attributes:
        aucs: ``COMPARISON_SCHEMA`` columns plus ``p_holm``. ``statistic`` is
            ``any``, ``arm`` or ``lower_body`` (AUC against healthy games,
            tested against 0.5) or ``arm - lower_body`` (tested against 0).
            ``p_holm`` adjusts the contrast rows across scores and phases and
            is null elsewhere.
        gains: The same columns for each later score minus the baseline, for
            every statistic and phase, with ``p_holm`` across all gain rows.
    """

    aucs: pl.DataFrame
    gains: pl.DataFrame


def drop_evidence(lag: int, drop_share: float) -> pl.Expr:
    """Log Bayes factor that a recent jump happened and went down.

    For any event, the posterior odds divided by the prior odds is the Bayes
    factor of the data for it. The event here is "a jump just before one of
    the last ``lag`` appearances, and the latest such jump was downward". Its
    prior probability is the chance of at least one jump over those gaps,
    ``1 - prod(1 - prior_jump_prob)``, times ``drop_share``, the chance a jump
    goes down. Unlike ``recent_drop_prob``, this does not rise just because a
    long absence made a jump likely beforehand.

    It reads ``player_id``, ``prior_jump_prob`` and ``recent_drop_prob``, and
    the frame must be sorted by player and date.

    Args:
        lag: Appearances in the recent window, as the filter used.
        drop_share: Prior probability that a jump is downward,
            ``Phi(-mu_J / sigma_J)``.

    Returns:
        An expression named ``drop_evidence``. Probabilities are clipped to
        ``[1e-12, 1 - 1e-12]`` first, so the evidence stays finite.

    Example:
        >>> frame = pl.DataFrame({"player_id": [1, 1, 1],
        ...     "prior_jump_prob": [0.0, 0.01, 0.01],
        ...     "recent_drop_prob": [0.0, 0.02, 0.5]})
        >>> evidence = frame.select(drop_evidence(2, 0.5))
        >>> [round(value, 2) for value in evidence["drop_evidence"]]
        [0.0, 1.4, 4.6]
    """
    window_prior = 1.0 - (
        pl.col("prior_jump_prob")
        .neg()
        .log1p()
        .rolling_sum(window_size=lag, min_samples=1)
        .over("player_id")
        .exp()
    )

    def log_odds(probability: pl.Expr) -> pl.Expr:
        clipped = probability.clip(PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)
        return clipped.log() - (1.0 - clipped).log()

    posterior = log_odds(pl.col("recent_drop_prob"))
    prior = log_odds(window_prior * drop_share)
    return (posterior - prior).alias("drop_evidence")


def drop_share(jump_mean: float, jump_sd: float) -> float:
    """Prior probability that a jump is downward.

    Args:
        jump_mean: Mean jump size ``mu_J``.
        jump_sd: Spread of jump sizes ``sigma_J``.

    Returns:
        ``Phi(-mu_J / sigma_J)``.

    Example:
        >>> drop_share(0.0, 1.2)
        0.5
    """
    return NormalDist().cdf(-jump_mean / jump_sd)


def _positive_masks(
    labeled: pl.DataFrame, horizon_days: int
) -> tuple[dict[tuple[str, str], np.ndarray], np.ndarray]:
    """Positives for each group and phase, and which games are in a window."""
    days = labeled["days_to_il"].fill_null(horizon_days + 1).to_numpy()
    in_window = days <= horizon_days
    group = labeled["injury_group"].fill_null("").to_numpy()
    final = labeled["is_final"].fill_null(False).to_numpy()
    masks = {}
    for phase in PHASES:
        in_phase = in_window & (final == (phase == "detection"))
        masks[ANY_GROUP, phase] = in_phase
        for name in (TARGET_GROUP, PLACEBO_GROUP):
            masks[name, phase] = in_phase & (group == name)
    return masks, in_window


def _statistics(
    bins: tuple[np.ndarray, int],
    masks: dict[tuple[str, str], np.ndarray],
    in_window: np.ndarray,
    weights: np.ndarray,
) -> dict[tuple[str, str], float]:
    """Every statistic for one score under one set of bootstrap weights."""
    ranks, count = bins
    values = {
        key: binned_auc(ranks, count, mask, weights * (mask | ~in_window))
        for key, mask in masks.items()
    }
    for phase in PHASES:
        values[CONTRAST, phase] = (
            values[TARGET_GROUP, phase] - values[PLACEBO_GROUP, phase]
        )
    return values


def _row(
    names: tuple[str, str, str], point: float, draws: np.ndarray, null: float
) -> tuple[str, str, str, float, float, float, float]:
    """One output row: names, the estimate and its bootstrap summary."""
    return (*names, point, *bootstrap_summary(draws, null))


def compare_scores(
    labeled: pl.DataFrame,
    scores: Sequence[str],
    horizon_days: int,
    reps: int = BOOTSTRAP_REPS,
    seed: int = 0,
) -> ScoreComparison:
    """AUCs of several scores and their paired gains over the first one.

    Args:
        labeled: Output of ``label_appearances`` with one column per score,
            higher meaning more alarming, and no nulls in them.
        scores: Score columns; the first is the baseline.
        horizon_days: Games this many days or fewer before a placement count.
        reps: Bootstrap resamples.
        seed: Random seed.

    Returns:
        The AUC table and the gain table.

    Raises:
        ValueError: If fewer than two scores are given.
    """
    if len(scores) < 2:
        raise ValueError("Need a baseline score and at least one other")
    masks, in_window = _positive_masks(labeled, horizon_days)
    players = labeled["player_id"].rank("dense").cast(pl.Int64).to_numpy() - 1
    player_count = int(players.max()) + 1
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(player_count, np.full(player_count, 1 / player_count), reps)
    keys = [(statistic, phase) for statistic in STATISTICS for phase in PHASES]
    points: dict[str, dict[tuple[str, str], float]] = {}
    samples: dict[str, np.ndarray] = {}
    for score in scores:
        values, ranks = np.unique(labeled[score].to_numpy(), return_inverse=True)
        bins = (ranks, values.size)
        points[score] = _statistics(bins, masks, in_window, np.ones(labeled.height))
        replicates = [
            _statistics(bins, masks, in_window, draw[players].astype(float))
            for draw in draws
        ]
        samples[score] = np.array([[rep[key] for key in keys] for rep in replicates])
    baseline = scores[0]
    auc_rows = []
    gain_rows = []
    for score in scores:
        for column, (statistic, phase) in enumerate(keys):
            null = 0.0 if statistic == CONTRAST else 0.5
            auc_rows.append(
                _row(
                    (score, statistic, phase),
                    points[score][statistic, phase],
                    samples[score][:, column],
                    null,
                )
            )
            if score != baseline:
                gain_rows.append(
                    _row(
                        (score, statistic, phase),
                        points[score][statistic, phase]
                        - points[baseline][statistic, phase],
                        samples[score][:, column] - samples[baseline][:, column],
                        0.0,
                    )
                )
    aucs = pl.DataFrame(auc_rows, orient="row", schema=COMPARISON_SCHEMA)
    contrast = aucs["statistic"].to_numpy() == CONTRAST
    adjusted = np.full(aucs.height, np.nan)
    adjusted[contrast] = holm_adjust(aucs["p_value"].to_numpy()[contrast])
    gains = pl.DataFrame(gain_rows, orient="row", schema=COMPARISON_SCHEMA)
    return ScoreComparison(
        aucs=aucs.with_columns(p_holm=pl.Series(adjusted).fill_nan(None)),
        gains=gains.with_columns(p_holm=holm_adjust(gains["p_value"].to_numpy())),
    )
