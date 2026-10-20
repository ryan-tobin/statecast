"""Do real-time velocity drops come before injured-list placements?

Each appearance carries a real-time alert score (``alert_z``, from
``models.trajectory.gap_scores``): how far the pitcher's current talent sits
below his own long-run level, using only games already pitched, scaled so it is
exactly standard normal for a pitcher whose talent follows the model. Strongly
negative scores mean he is throwing further below his own baseline than
ordinary drift explains.

Appearances are labeled by the pitcher's next IL placement. Arm injuries are
the target; other injuries (legs, back, illness) are a placebo, since a drop in
velocity has no reason to warn of a hamstring strain.
"""

from collections.abc import Sequence
from typing import Final

import numpy as np
import polars as pl

LOOKBACK_DAYS: Final = 60
Z_95: Final = 1.959963984540054
INJURY_GROUPS: Final = ("arm", "other")


def label_appearances(states: pl.DataFrame, placements: pl.DataFrame) -> pl.DataFrame:
    """Attach each appearance's next IL placement for the same player.

    Args:
        states: Output of ``load_alert_states``: at least ``player_id``,
            ``game_date`` and ``alert_z``.
        placements: Output of ``il_placements``.

    Returns:
        ``states`` plus ``il_start``, ``injury_group`` and ``days_to_il`` of the
        first placement that starts after the game (null when there is none),
        sorted by player and date.
    """
    upcoming = placements.select("player_id", "il_start", "injury_group").sort(
        "il_start"
    )
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
        .with_columns(
            days_to_il=(pl.col("il_start") - pl.col("game_date")).dt.total_days()
        )
        .sort("player_id", "game_date")
    )


def _mean_interval(frame: pl.DataFrame, keys: Sequence[str]) -> pl.DataFrame:
    """Mean score with a normal 95% interval for each group of ``keys``."""
    standard_error = pl.col("alert_z").std() / pl.len().sqrt()
    return frame.group_by(keys).agg(
        mean_z=pl.col("alert_z").mean(),
        lower_95=pl.col("alert_z").mean() - Z_95 * standard_error,
        upper_95=pl.col("alert_z").mean() + Z_95 * standard_error,
        appearances=pl.len(),
    )


def event_study(labeled: pl.DataFrame, max_starts: int = 8) -> pl.DataFrame:
    """Average alert score by appearances remaining before an IL placement.

    Each placement contributes the pitcher's last ``max_starts`` appearances
    in the ``LOOKBACK_DAYS`` before it, numbered ``-1`` (the last game before
    the IL) back to ``-max_starts``. Appearances with no placement in the
    following ``LOOKBACK_DAYS`` form the ``healthy`` baseline, with
    ``starts_before`` null.

    Args:
        labeled: Output of ``label_appearances``.
        max_starts: Appearances to keep before each placement.

    Returns:
        Columns ``group``, ``starts_before``, ``mean_z``, ``lower_95``,
        ``upper_95`` and ``appearances``.
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
    columns = ["group", "starts_before", "mean_z", "lower_95", "upper_95"]
    return (
        pl.concat(
            [
                _mean_interval(before, ["group", "starts_before"]),
                _mean_interval(healthy, ["group", "starts_before"]),
            ]
        )
        .select(*columns, "appearances")
        .sort("group", "starts_before")
    )


def _positive(horizon_days: int, group: str) -> pl.Expr:
    """True for appearances followed by a ``group`` placement within the horizon."""
    return (pl.col("days_to_il") <= horizon_days).fill_null(False) & (
        pl.col("injury_group") == group
    ).fill_null(False)


def roc_auc(labeled: pl.DataFrame, horizon_days: int, group: str) -> float:
    """Chance a pre-injury appearance scores lower than a random other one.

    This is the area under the ROC curve of ``-alert_z`` for predicting a ``group``
    placement within ``horizon_days``, from the Mann-Whitney rank sum. Ties
    count one half. 0.5 means the score carries no warning.

    Args:
        labeled: Output of ``label_appearances``.
        horizon_days: Days after an appearance that count as "before" an IL stint.
        group: ``arm`` or ``other``.

    Returns:
        The AUC, or NaN if either class is empty.

    Example:
        >>> frame = pl.DataFrame({"alert_z": [-3.0, -1.0, 0.5, 1.0],
        ...     "days_to_il": [5, None, 40, None],
        ...     "injury_group": ["arm", None, "arm", None]})
        >>> roc_auc(frame, 30, "arm")
        1.0
    """
    ranked = labeled.select(
        positive=_positive(horizon_days, group),
        rank=(-pl.col("alert_z")).rank("average"),
    )
    positives = int(ranked["positive"].sum())
    negatives = ranked.height - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    rank_sum = float(ranked.filter(pl.col("positive"))["rank"].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def threshold_table(
    labeled: pl.DataFrame, thresholds: Sequence[float], horizon_days: int
) -> pl.DataFrame:
    """How well ``alert_z <= -threshold`` warns of an arm IL placement.

    Args:
        labeled: Output of ``label_appearances``.
        thresholds: Alert thresholds in standard deviations, such as ``2.0``.
        horizon_days: An alert counts if an arm placement starts within this
            many days after the game.

    Returns:
        One row per threshold: ``threshold``, ``alert_rate`` (share of
        appearances alerted), ``precision`` (share of alerts followed by an
        arm placement), ``lift`` (precision over the base rate),
        ``events_caught``, ``events``, ``recall`` (share of arm placements
        with an alert in the horizon before them) and ``median_lead_days``
        (from the first such alert to the placement).
    """
    flagged = labeled.with_columns(positive=_positive(horizon_days, "arm"))
    base_rate = float(flagged["positive"].mean())  # type: ignore[arg-type]
    windows = flagged.filter(pl.col("positive"))
    events = windows.select("player_id", "il_start").n_unique()
    rows = []
    for threshold in thresholds:
        alerted = flagged.filter(pl.col("alert_z") <= -threshold)
        caught = (
            windows.filter(pl.col("alert_z") <= -threshold)
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
                "alert_rate": alerted.height / flagged.height,
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


def current_alerts(
    states: pl.DataFrame, threshold: float, top: int = 10
) -> pl.DataFrame:
    """Pitchers whose latest appearance of the latest season is below threshold.

    Args:
        states: Output of ``load_alert_states``.
        threshold: Alert when ``alert_z <= -threshold``.
        top: Most severe alerts to return.

    Returns:
        ``full_name``, ``last_game``, ``talent``, ``gap`` (talent minus own
        level, in the metric's units) and ``alert_z``, most severe first.
    """
    latest_season = states["game_date"].dt.year().max()
    return (
        states.filter(pl.col("game_date").dt.year() == latest_season)
        .group_by("player_id")
        .agg(pl.all().sort_by("game_date").last())
        .filter(pl.col("alert_z") <= -threshold)
        .sort("alert_z")
        .head(top)
        .select(
            "full_name",
            last_game=pl.col("game_date"),
            talent=pl.col("filtered_mean"),
            gap=pl.col("filtered_gap_mean"),
            alert_z=pl.col("alert_z"),
        )
    )
