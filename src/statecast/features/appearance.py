"""Per-appearance sufficient statistics pulled form the warehouse.

Each row summarizes one player's pitches (or swings) of one metric in one
regular-season game: the sample size, the mean and the within-game sum of sqaures.
"""

from collections.abc import Sequence

import duckdb
import polars as pl

PITCHER_METRICS = frozenset(
    {"release_speed", "spin_rate", "release_extension", "arm_angle"}
)
HITTER_METRICS = frozenset(
    {"bat_speed", "swing_length", "attack_angle", "swing_path_tilt"}
)
TOP_PITCH_TYPES = 3
MIN_COMPETITIVE_BAT_SPEED = 50.0

PITCHER_SQL = """
WITH usage AS (
    SELECT
        p.pitcher_id,
        p.pitch_type,
        row_number() OVER (
            PARTITION BY p.pitcher_id ORDER BY count(*) DESC, p.pitch_type
        ) AS usage_rank
    FROM pitch AS p
    JOIN game AS g USING (game_pk)
    WHERE g.game_type = 'R'
      AND list_contains($seasons, g.season)
      AND p.pitch_type IS NOT NULL
      AND p.release_speed IS NOT NULL
    GROUP BY p.pitcher_id, p.pitch_type
),
appearances AS (
    SELECT
        p.pitcher_id AS player_id,
        g.game_date,
        count(*) AS sample_size,
        avg(p.{metric}) AS mean,
        var_pop(p.{metric}) * count(*) AS within_ss
    FROM pitch AS p
    JOIN game AS g USING (game_pk)
    JOIN usage AS u USING (pitcher_id, pitch_type)
    WHERE g.game_type = 'R'
      AND list_contains($seasons, g.season)
      AND p.pitch_type = $pitch_type
      AND u.usage_rank <= $top_pitch_types
      AND p.{metric} IS NOT NULL
    GROUP BY p.pitcher_id, g.game_date
)
SELECT player_id, game_date, sample_size, mean, within_ss
FROM appearances
QUALIFY count(*) OVER (PARTITION BY player_id) >= $min_appearances
ORDER BY player_id, game_date
"""

HITTER_SQL = """
WITH appearances AS (
    SELECT
        p.batter_id AS player_id,
        g.game_date,
        count(*) AS sample_size,
        avg(s.{metric}) AS mean,
        var_pop(s.{metric}) * count(*) AS within_ss
    FROM swing AS s
    JOIN pitch AS p USING (game_pk, at_bat_number, pitch_number)
    JOIN game AS g USING (game_pk)
    WHERE g.game_type = 'R'
      AND list_contains($seasons, g.season)
      AND s.bat_speed >= $min_bat_speed
      AND p.description NOT LIKE '%bunt%'
      AND s.{metric} IS NOT NULL
    GROUP BY p.batter_id, g.game_date
)
SELECT player_id, game_date, sample_size, mean, within_ss
FROM appearances
QUALIFY count(*) OVER (PARTITION BY player_id) >= $min_appearances
ORDER BY player_id, game_date
"""


def pitcher_appearances(
    connection: duckdb.DuckDBPyConnection,
    pitch_type: str,
    seasons: Sequence[int],
    metric: str = "release_speed",
    min_appearances: int = 10,
) -> pl.DataFrame:
    """Summarize one pitch type's metric per pitcher per regular-season game.

    Only pitchers for whom the pitch type is among their three most-used
    pitches over ``seasons`` are included, and untracked pitches are dropped.

    Args:
        connection: Open warehouse connection.
        pitch_type: Statcast pitch-type code, such as ``FF``.
        seasons: Seasons to include.
        metric: A column in ``PITCHER_METRICS``.
        min_appearances: Drop pitchers with fewer appearances than this.

    Returns:
        Columns ``player_id``, ``game_date``, ``sample_size``, ``mean``, ``within_ss``.

    Raises:
        ValueError: If ``metric`` is not a supported pitcher metric.
    """
    if metric not in PITCHER_METRICS:
        raise ValueError(f"Unsupported pitcher metric {metric!r}")
    return connection.execute(
        PITCHER_SQL.format(metric=metric),
        {
            "seasons": list(seasons),
            "pitch_type": pitch_type,
            "top_pitch_types": TOP_PITCH_TYPES,
            "min_appearances": min_appearances,
        },
    ).pl()


def hitter_appearances(
    connection: duckdb.DuckDBPyConnection,
    seasons: Sequence[int],
    metric: str = "bat_speed",
    min_appearances: int = 10,
) -> pl.DataFrame:
    """Summarize one swing metric per hitter per regular-season game.

    Only competitive swings count: bunts and swings under 50 mph (check swings
    and defensive pokes) are excluded. This threshold is a project choice.

    Args:
        connection: Open warehouse connection.
        seasons: Seasons to include.
        metric: A column in ``HITTER_METRICS``.
        min_appearances: Drop hitters with fewer games than this.

    Returns:
        Columns ``player_id``, ``game_date``, ``sample_size``, ``mean``, ``within_ss``.

    Raises:
        ValueError: If ``metric`` is not a supported hitter metric.
    """
    if metric not in HITTER_METRICS:
        raise ValueError(f"Unsupported hitter metric {metric!r}")
    return connection.execute(
        HITTER_SQL.format(metric=metric),
        {
            "seasons": list(seasons),
            "min_bat_speed": MIN_COMPETITIVE_BAT_SPEED,
            "min_appearances": min_appearances,
        },
    ).pl()
