"""Pure transforms from cached raw frames to normalized warehouse tables."""

import polars as pl

PLATE_MIDDLE_FIRST_SEASON = 2026
PITCH_KEY = ["game_pk", "at_bat_number", "pitch_number"]
PLATE_APPEARANCE_KEY = ["game_pk", "at_bat_number"]
BAT_TRACKING_COLUMNS = [
    "bat_speed",
    "swing_length",
    "attack_angle",
    "attack_direction",
    "swing_path_tilt",
]


def pitch_table(statcast: pl.DataFrame) -> pl.DataFrame:
    """Build the ``pitch`` table: one row per pitch, pitch-level facts only.

    ``plate_reference`` records where ``plate_x`` and ``plate_z`` were
    measured: the front of the plate through 2025, the middle from 2026.

    Args:
        statcast: Parsed Statcast rows.

    Returns:
        Rows matching the ``pitch`` table columns.
    """
    return statcast.select(
        *PITCH_KEY,
        pl.col("pitcher").alias("pitcher_id"),
        pl.col("batter").alias("batter_id"),
        "pitch_type",
        "balls",
        "strikes",
        "release_speed",
        pl.col("release_spin_rate").alias("spin_rate"),
        "spin_axis",
        "pfx_x",
        "pfx_z",
        pl.col("api_break_x_arm").alias("break_x_arm"),
        pl.col("api_break_z_with_gravity").alias("break_z_with_gravity"),
        "release_pos_x",
        "release_pos_y",
        "release_pos_z",
        "release_extension",
        "arm_angle",
        "vx0",
        "vy0",
        "vz0",
        "ax",
        "ay",
        "az",
        "plate_x",
        "plate_z",
        pl.when(pl.col("game_date").dt.year() >= PLATE_MIDDLE_FIRST_SEASON)
        .then(pl.lit("middle"))
        .otherwise(pl.lit("front"))
        .alias("plate_reference"),
        "sz_top",
        "sz_bot",
        "zone",
        "description",
        pl.col("type").alias("result_type"),
        "launch_speed",
        "launch_angle",
        "delta_run_exp",
    )


def swing_table(statcast: pl.DataFrame) -> pl.DataFrame:
    """Build the ``swing`` table: bat-tracking metrics for tracked swings.

    Args:
        statcast: Parsed Statcast rows.

    Returns:
        One row per pitch with at least one bat-tracking value.
    """
    return statcast.filter(
        pl.any_horizontal(pl.col(BAT_TRACKING_COLUMNS).is_not_null())
    ).select(*PITCH_KEY, *BAT_TRACKING_COLUMNS)


def plate_appearance_table(statcast: pl.DataFrame) -> pl.DataFrame:
    """Build the ``plate_appearance`` table from the pitches in each one.

    Situation columns come from the first pitch; the event comes from the last
    pitch that recorded one.

    Args:
        statcast: Parsed Statcast rows.

    Returns:
        One row per plate appearance.
    """
    return (
        statcast.sort(PITCH_KEY)
        .group_by(PLATE_APPEARANCE_KEY, maintain_order=True)
        .agg(
            pl.col("inning").first(),
            pl.col("inning_topbot").first().alias("inning_half"),
            pl.col("outs_when_up").first().alias("outs"),
            pl.col("n_thruorder_pitcher").first().alias("times_through_order"),
            pl.col("events").drop_nulls().last().alias("event"),
        )
    )


def pitcher_appearance_table(statcast: pl.DataFrame) -> pl.DataFrame:
    """Build the ``pitcher_appearance`` table: one row per pitcher per game.

    Args:
        statcast: Parsed Statcast rows.

    Returns:
        Game, pitcher and days of rest before the game.
    """
    return (
        statcast.group_by("game_pk", "pitcher", maintain_order=True)
        .agg(
            pl.col("pitcher_days_since_prev_game").first().alias("days_since_prev_game")
        )
        .rename({"pitcher": "pitcher_id"})
    )


def pitch_type_table(statcast: pl.DataFrame) -> pl.DataFrame:
    """Build the ``pitch_type`` lookup of codes and names.

    Args:
        statcast: Parsed Statcast rows.

    Returns:
        One row per pitch-type code.
    """
    return (
        statcast.filter(pl.col("pitch_type").is_not_null())
        .group_by("pitch_type", maintain_order=True)
        .agg(pl.col("pitch_name").drop_nulls().first().alias("name"))
        .rename({"pitch_type": "code"})
    )


def game_table(schedule: pl.DataFrame) -> pl.DataFrame:
    """Build the ``game`` table from cached schedule rows.

    Args:
        schedule: Cached schedule rows.

    Returns:
        One row per game.
    """
    return schedule.select(
        "game_pk",
        "game_date",
        "season",
        "game_type",
        "venue_id",
        "home_team_id",
        "away_team_id",
        "first_pitch_at",
    )


def venue_table(schedule: pl.DataFrame) -> pl.DataFrame:
    """Build the ``venue`` table from cached schedule rows.

    Args:
        schedule: Cached schedule rows.

    Returns:
        One row per venue.
    """
    return schedule.select("venue_id", pl.col("venue_name").alias("name")).unique(
        subset="venue_id", keep="last", maintain_order=True
    )


def team_table(schedule: pl.DataFrame) -> pl.DataFrame:
    """Build the ``team`` table from both sides of each scheduled game.

    Args:
        schedule: Cached schedule rows.

    Returns:
        One row per team.
    """
    return pl.concat(
        [
            schedule.select(
                pl.col(f"{side}_team_id").alias("team_id"),
                pl.col(f"{side}_team_name").alias("name"),
            )
            for side in ("home", "away")
        ]
    ).unique(subset="team_id", keep="last", maintain_order=True)