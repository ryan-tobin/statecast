"""How each pitcher is being used, relative to his own history so far.

Managers use pitchers less when they are hurting, so workload carries injury
information of its own. Each appearance gets two real-time scores, computed
only from the pitcher's earlier appearances: total pitches thrown and log rest
days, each as a z-score against his own running mean and sd.
"""

from collections.abc import Sequence
from typing import Final

import duckdb
import polars as pl

MIN_HISTORY: Final = 5
MAX_REST_DAYS: Final = 30

USAGE_SQL: Final = """
SELECT p.pitcher_id AS player_id, g.game_date, count(*) AS pitches
FROM pitch AS p
JOIN game AS g USING (game_pk)
WHERE g.game_type = 'R'
  AND list_contains($seasons, g.season)
GROUP BY p.pitcher_id, g.game_date
ORDER BY player_id, game_date
"""


def pitcher_usage(
    connection: duckdb.DuckDBPyConnection, seasons: Sequence[int]
) -> pl.DataFrame:
    """Total pitches per pitcher per game date (doubleheaders combined).

    Args:
        connection: Warehouse connection.
        seasons: Seasons to include.

    Returns:
        ``player_id``, ``game_date`` and ``pitches``.
    """
    return connection.execute(USAGE_SQL, {"seasons": list(seasons)}).pl()


def _prior_z(column: str) -> pl.Expr:
    """Z-score of a value against the same player's earlier values.

    Uses running sums of the previous values, so nothing from the current or
    later games leaks in. Null until ``MIN_HISTORY`` earlier values exist or
    when they have no spread.
    """
    value = pl.col(column)
    count = value.shift(1).cum_count().over("player_id")
    total = value.shift(1).cum_sum().over("player_id")
    squares = value.pow(2).shift(1).cum_sum().over("player_id")
    mean = total / count
    variance = (squares - count * mean.pow(2)) / (count - 1)
    return (
        pl.when((count >= MIN_HISTORY) & (variance > 0))
        .then((value - mean) / variance.sqrt())
        .otherwise(None)
    )


def relative_usage(usage: pl.DataFrame) -> pl.DataFrame:
    """Add real-time workload z-scores to each appearance.

    Rest is capped at ``MAX_REST_DAYS`` before taking ``log1p``, so offseason
    and IL gaps do not swamp a pitcher's normal rest pattern.

    Args:
        usage: Output of ``pitcher_usage``.

    Returns:
        ``usage`` plus ``rest_days``, ``pitches_z`` and ``rest_z``, sorted by
        player and date.

    Example:
        >>> from datetime import date, timedelta
        >>> frame = pl.DataFrame({"player_id": [1] * 7,
        ...     "game_date": [date(2026, 4, 1) + timedelta(days=5 * k)
        ...                   for k in range(7)],
        ...     "pitches": [90, 95, 100, 92, 98, 94, 60]})
        >>> round(relative_usage(frame)["pitches_z"][-1], 2)
        -9.39
    """
    return (
        usage.sort("player_id", "game_date")
        .with_columns(
            rest_days=pl.col("game_date").diff().over("player_id").dt.total_days()
        )
        .with_columns(
            log_rest=pl.col("rest_days").clip(upper_bound=MAX_REST_DAYS).log1p()
        )
        .with_columns(pitches_z=_prior_z("pitches"), rest_z=_prior_z("log_rest"))
        .drop("log_rest")
    )
