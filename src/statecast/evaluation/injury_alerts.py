"""Do real-time changes in a pitch metric come before injured-list placements?

Each appearance carries a real-time alert z-score (``alert_z``, from
``models.trajectory.gap_scores``): how far the pitcher's current talent in one
metric sits from his own long-run level, using only games already pitched,
scaled so it is exactly standard normal for a pitcher whose talent follows the
model. A ``Direction`` turns it into a ``score`` where higher is always more
alarming: a ``drop`` (velocity, spin, extension), a ``rise``, or ``either``
(arm angle, which can drift up or down).

Appearances are labeled by the pitcher's next IL placement and split into two
phases. The *detection* game is the last appearance before a placement, often
the game the injury happened in. *Warning* games are the earlier ones, the only
ones an alert could act on. Arm injuries are the target; lower-body injuries
are the placebo, since a hamstring strain gives the arm no reason to change
beforehand. Trunk injuries (back, oblique, ribs) sit in between, because they
change a pitcher's delivery.

Appearances a pitcher makes are not a random sample of his days: pitchers who
are throwing badly get used less. So every comparison is against healthy
appearances (no placement in the lookback window), not against zero.
"""

from collections.abc import Sequence
from enum import StrEnum
from typing import Final

import numpy as np
import polars as pl

LOOKBACK_DAYS: Final = 60
Z_95: Final = 1.959963984540054
INJURY_GROUPS: Final = ("arm", "trunk", "lower_body", "other")
TARGET_GROUP: Final = "arm"
PLACEBO_GROUP: Final = "lower_body"
PHASES: Final = ("warning", "detection")
BOOTSTRAP_REPS: Final = 500
AUC_TABLE_SCHEMA: Final = {
    "group": pl.String,
    "phase": pl.String,
    "placements": pl.Int64,
    "auc": pl.Float64,
    "lower_95": pl.Float64,
    "upper_95": pl.Float64,
    "p_value": pl.Float64,
}


class Direction(StrEnum):
    """Which way a metric has to move to be alarming."""

    DROP = "drop"
    RISE = "rise"
    EITHER = "either"


def alert_score(direction: Direction) -> pl.Expr:
    """Turn ``alert_z`` into a score where higher is more alarming.

    Args:
        direction: ``drop`` scores ``-alert_z``, ``rise`` scores ``alert_z``
            and ``either`` scores ``|alert_z|``.

    Returns:
        An expression named ``score``.

    Example:
        >>> frame = pl.DataFrame({"alert_z": [-2.0, 1.5]})
        >>> frame.select(alert_score(Direction.EITHER))["score"].to_list()
        [2.0, 1.5]
    """
    z = pl.col("alert_z")
    scores = {Direction.DROP: -z, Direction.RISE: z, Direction.EITHER: z.abs()}
    return scores[direction].alias("score")


def label_appearances(states: pl.DataFrame, placements: pl.DataFrame) -> pl.DataFrame:
    """Attach each appearance's next IL placement for the same player.

    Args:
        states: Output of ``load_alert_states``: at least ``player_id``,
            ``game_date`` and ``alert_z``.
        placements: Output of ``il_placements``.

    Returns:
        ``states`` plus ``il_start``, ``injury_group``, ``injury`` and
        ``days_to_il`` of the first placement that starts after the game (null
        when there is none), and ``is_final``: True for the last appearance
        before that placement. Sorted by player and date.
    """
    upcoming = placements.select(
        "player_id", "il_start", "injury_group", "injury"
    ).sort("il_start")
    next_game = pl.col("game_date").shift(-1).over("player_id")
    return (
        states.sort("game_date")
        .join_asof(
            upcoming,
            left_on="game_date",
            right_on="il_start",
            by="player_id",
            strategy="forward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .sort("player_id", "game_date")
        .with_columns(
            days_to_il=(pl.col("il_start") - pl.col("game_date")).dt.total_days(),
            is_final=pl.col("il_start").is_not_null()
            & (next_game.is_null() | (next_game >= pl.col("il_start"))),
        )
    )


def phase_sample(
    labeled: pl.DataFrame, horizon_days: int, group: str, phase: str
) -> pl.DataFrame:
    """Appearances before a ``group`` placement in one phase, plus healthy ones.

    Args:
        labeled: Output of ``label_appearances``.
        horizon_days: Games this many days or fewer before a placement count.
        group: Injury group of the placement, such as ``arm``.
        phase: ``warning`` (earlier games) or ``detection`` (the final game).

    Returns:
        The positive games and every healthy game (no placement within the
        horizon), with a ``positive`` column. Games before other injury groups
        or in the other phase are left out.
    """
    in_window = (pl.col("days_to_il") <= horizon_days).fill_null(False)
    final = pl.col("is_final") if phase == "detection" else ~pl.col("is_final")
    positive = in_window & (pl.col("injury_group") == group).fill_null(False) & final
    return labeled.filter(positive | ~in_window).with_columns(positive=positive)


def weighted_auc(
    scores: np.ndarray, positive: np.ndarray, weights: np.ndarray
) -> float:
    """Weighted Mann-Whitney AUC: chance a positive outscores a negative.

    Ties count one half. Higher scores should mean more alarming.

    Args:
        scores: Score per observation.
        positive: True for positives.
        weights: Non-negative weight per observation (bootstrap counts).

    Returns:
        The AUC, or NaN if either class has no weight.

    Example:
        >>> weighted_auc(np.array([3.0, 1.0, 2.0, 0.0]),
        ...              np.array([True, False, True, False]), np.ones(4))
        1.0
    """
    values, bins = np.unique(scores, return_inverse=True)
    return _binned_auc(bins, values.size, positive, weights)


def _binned_auc(
    bins: np.ndarray, bin_count: int, positive: np.ndarray, weights: np.ndarray
) -> float:
    """AUC from scores already replaced by their rank among unique values."""
    positive_weight = np.bincount(bins, weights * positive, minlength=bin_count)
    negative_weight = np.bincount(bins, weights * ~positive, minlength=bin_count)
    total_positive = positive_weight.sum()
    total_negative = negative_weight.sum()
    if total_positive == 0 or total_negative == 0:
        return float("nan")
    below = np.cumsum(negative_weight) - negative_weight
    wins = np.sum(positive_weight * (below + 0.5 * negative_weight))
    return float(wins / (total_positive * total_negative))


def bootstrap_summary(draws: np.ndarray, null: float) -> tuple[float, float, float]:
    """Percentile interval and two-sided p-value from bootstrap draws.

    The p-value is twice the smaller share of draws on either side of the
    null value, with one pseudo-draw added to each side so it is never zero.

    Args:
        draws: Bootstrap replicates of a statistic (NaNs are ignored).
        null: Value under the null hypothesis, such as 0.5 for an AUC.

    Returns:
        ``(lower_95, upper_95, p_value)``, all NaN if no draw is finite.

    Example:
        >>> low, high, p = bootstrap_summary(np.linspace(0.55, 0.65, 99), 0.5)
        >>> round(p, 3)
        0.02
    """
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    low, high = np.percentile(finite, [2.5, 97.5])
    below = (np.sum(finite <= null) + 1) / (finite.size + 1)
    above = (np.sum(finite >= null) + 1) / (finite.size + 1)
    return float(low), float(high), float(min(1.0, 2.0 * min(below, above)))


def auc_table(
    labeled: pl.DataFrame, horizon_days: int, reps: int = BOOTSTRAP_REPS, seed: int = 0
) -> pl.DataFrame:
    """AUC of the alert score for every injury group and phase, with intervals.

    Each AUC compares games before a placement (positives) with healthy games
    (negatives), ranking by ``score``. Intervals and p-values come from a
    pitcher-cluster bootstrap: whole pitchers are resampled, because games of
    one pitcher are correlated. Every group is scored on the same resamples,
    so the ``arm - lower_body`` rows (target minus placebo) get paired
    intervals. Group rows test AUC = 0.5; difference rows test a difference
    of 0.

    Args:
        labeled: Output of ``label_appearances`` with a ``score`` column.
        horizon_days: Games this many days or fewer before a placement count.
        reps: Bootstrap resamples.
        seed: Random seed.

    Returns:
        ``group``, ``phase``, ``placements``, ``auc``, ``lower_95``,
        ``upper_95`` and ``p_value``.
    """
    values, bins = np.unique(labeled["score"].to_numpy(), return_inverse=True)
    players = labeled["player_id"].rank("dense").cast(pl.Int64).to_numpy() - 1
    player_count = int(players.max()) + 1
    days = labeled["days_to_il"].fill_null(horizon_days + 1).to_numpy()
    in_window = days <= horizon_days
    group = labeled["injury_group"].fill_null("").to_numpy()
    final = labeled["is_final"].to_numpy()
    keys = [(name, phase) for name in INJURY_GROUPS for phase in PHASES]
    positives = {
        (name, phase): in_window & (group == name) & (final == (phase == "detection"))
        for name, phase in keys
    }
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(player_count, np.full(player_count, 1 / player_count), reps)

    def aucs(weights: np.ndarray) -> dict[tuple[str, str], float]:
        return {
            key: _binned_auc(bins, values.size, mask, weights * (mask | ~in_window))
            for key, mask in positives.items()
        }

    point = aucs(np.ones(labeled.height))
    samples = [aucs(draw[players].astype(float)) for draw in draws]
    rows: list[tuple[str, str, int | None, float, float, float, float]] = []
    for name, phase in keys:
        events = labeled.filter(pl.Series(positives[name, phase]))
        replicates = np.array([sample[name, phase] for sample in samples])
        rows.append(
            (
                name,
                phase,
                events.select("player_id", "il_start").n_unique(),
                point[name, phase],
                *bootstrap_summary(replicates, 0.5),
            )
        )
    for phase in PHASES:
        target, placebo = (TARGET_GROUP, phase), (PLACEBO_GROUP, phase)
        replicates = np.array([sample[target] - sample[placebo] for sample in samples])
        rows.append(
            (
                f"{TARGET_GROUP} - {PLACEBO_GROUP}",
                phase,
                None,
                point[target] - point[placebo],
                *bootstrap_summary(replicates, 0.0),
            )
        )
    return pl.DataFrame(rows, orient="row", schema=AUC_TABLE_SCHEMA)


def threshold_table(
    labeled: pl.DataFrame, thresholds: Sequence[float], horizon_days: int
) -> pl.DataFrame:
    """How well ``score >= threshold`` gives warning of an arm placement.

    Only warning games count: the final game before a placement is left out,
    since an alert there comes too late to act on. Negatives are healthy games.

    Args:
        labeled: Output of ``label_appearances`` with a ``score`` column.
        thresholds: Alert thresholds in standard deviations, such as ``2.0``.
        horizon_days: Games this many days or fewer before a placement count.

    Returns:
        One row per threshold: ``threshold``, ``alert_rate``, ``precision``,
        ``lift`` (precision over the base rate), ``events_caught``, ``events``
        (arm placements with at least one warning game), ``recall`` and
        ``median_lead_days`` (from the first alert to the placement).
    """
    sample = phase_sample(labeled, horizon_days, TARGET_GROUP, "warning")
    base_rate = float(sample["positive"].mean())  # type: ignore[arg-type]
    windows = sample.filter(pl.col("positive"))
    events = windows.select("player_id", "il_start").n_unique()
    rows = []
    for threshold in thresholds:
        alerted = sample.filter(pl.col("score") >= threshold)
        caught = (
            windows.filter(pl.col("score") >= threshold)
            .group_by("player_id", "il_start")
            .agg(lead_days=pl.col("days_to_il").max())
        )
        precision = (
            float(alerted["positive"].mean())  # type: ignore[arg-type]
            if alerted.height
            else float("nan")
        )
        rows.append(
            {
                "threshold": threshold,
                "alert_rate": alerted.height / sample.height,
                "precision": precision,
                "lift": precision / base_rate if base_rate else float("nan"),
                "events_caught": caught.height,
                "events": events,
                "recall": caught.height / events if events else float("nan"),
                "median_lead_days": (
                    float(np.median(caught["lead_days"].to_numpy()))
                    if caught.height
                    else float("nan")
                ),
            }
        )
    return pl.DataFrame(rows)


def event_study(labeled: pl.DataFrame, max_starts: int = 8) -> pl.DataFrame:
    """Mean alert score by appearances remaining before an IL placement.

    Each placement contributes the pitcher's last ``max_starts`` appearances
    in the ``LOOKBACK_DAYS`` before it, numbered ``-1`` (the final game) back
    to ``-max_starts``. Appearances with no placement in the following
    ``LOOKBACK_DAYS`` form the ``healthy`` row, with ``starts_before`` null.

    Args:
        labeled: Output of ``label_appearances`` with a ``score`` column.
        max_starts: Appearances to keep before each placement.

    Returns:
        ``group``, ``starts_before``, ``mean_score``, ``standard_error`` and
        ``appearances``.
    """
    before = (
        labeled.filter(pl.col("days_to_il") <= LOOKBACK_DAYS)
        .with_columns(
            starts_before=-pl.col("game_date")
            .rank("ordinal", descending=True)
            .over("player_id", "il_start")
            .cast(pl.Int32)
        )
        .filter(pl.col("starts_before") >= -max_starts)
        .rename({"injury_group": "group"})
    )
    healthy = labeled.filter(
        pl.col("days_to_il").is_null() | (pl.col("days_to_il") > LOOKBACK_DAYS)
    ).with_columns(group=pl.lit("healthy"), starts_before=pl.lit(None, pl.Int32))
    summary = {
        "mean_score": pl.col("score").mean(),
        "standard_error": pl.col("score").std() / pl.len().sqrt(),
        "appearances": pl.len(),
    }
    keys = ["group", "starts_before"]
    return pl.concat(
        [before.group_by(keys).agg(**summary), healthy.group_by(keys).agg(**summary)]
    ).sort("group", "starts_before")


def relative_to_healthy(study: pl.DataFrame) -> pl.DataFrame:
    """Shift of each group's mean score from the healthy baseline.

    Args:
        study: Output of ``event_study``.

    Returns:
        ``group``, ``starts_before``, ``shift``, ``lower_95``, ``upper_95``
        and ``placements`` for every injury group and appearance.

    Raises:
        ValueError: If the study has no healthy row.
    """
    healthy = study.filter(pl.col("group") == "healthy")
    if healthy.is_empty():
        raise ValueError("Event study has no healthy baseline")
    baseline = healthy.row(0, named=True)
    standard_error = (
        pl.col("standard_error").pow(2) + baseline["standard_error"] ** 2
    ).sqrt()
    shift = pl.col("mean_score") - baseline["mean_score"]
    return study.filter(pl.col("group") != "healthy").select(
        "group",
        "starts_before",
        shift=shift,
        lower_95=shift - Z_95 * standard_error,
        upper_95=shift + Z_95 * standard_error,
        placements=pl.col("appearances"),
    )


def group_contrast(study: pl.DataFrame, group: str, placebo: str) -> pl.DataFrame:
    """Difference in mean score between two injury groups, game by game.

    The groups are different placements, so their errors are independent.

    Args:
        study: Output of ``event_study``.
        group: Target group, such as ``arm``.
        placebo: Comparison group, such as ``lower_body``.

    Returns:
        ``starts_before``, ``difference`` (group minus placebo), ``lower_95``
        and ``upper_95``.
    """
    columns = ["starts_before", "mean_score", "standard_error"]
    target = study.filter(pl.col("group") == group).select(columns)
    control = study.filter(pl.col("group") == placebo).select(columns)
    joined = target.join(control, on="starts_before", suffix="_placebo")
    difference = pl.col("mean_score") - pl.col("mean_score_placebo")
    standard_error = (
        pl.col("standard_error").pow(2) + pl.col("standard_error_placebo").pow(2)
    ).sqrt()
    return joined.select(
        "starts_before",
        difference=difference,
        lower_95=difference - Z_95 * standard_error,
        upper_95=difference + Z_95 * standard_error,
    ).sort("starts_before")


def current_alerts(
    states: pl.DataFrame, placements: pl.DataFrame, threshold: float, top: int = 10
) -> pl.DataFrame:
    """Pitchers on alert after their latest game, and what happened next.

    Args:
        states: Output of ``load_alert_states`` with a ``score`` column.
        placements: Output of ``il_placements``.
        threshold: Alert when ``score >= threshold``.
        top: Most severe alerts to return.

    Returns:
        ``full_name``, ``last_game``, ``talent``, ``gap`` (talent minus own
        level, in the metric's units), the signed ``alert_z``, and the next IL
        placement after the last game (``next_il``, ``injury``; null if none),
        most severe first.
    """
    latest_season = states["game_date"].dt.year().max()
    latest = (
        states.filter(pl.col("game_date").dt.year() == latest_season)
        .group_by("player_id")
        .agg(pl.all().sort_by("game_date").last())
        .filter(pl.col("score") >= threshold)
        .sort("score", descending=True)
        .head(top)
    )
    return (
        label_appearances(latest, placements)
        .sort("score", descending=True)
        .select(
            "full_name",
            last_game=pl.col("game_date"),
            talent=pl.col("filtered_mean"),
            gap=pl.col("filtered_gap_mean"),
            alert_z=pl.col("alert_z"),
            next_il=pl.col("il_start"),
            injury=pl.col("injury"),
        )
    )
