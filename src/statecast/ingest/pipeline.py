"""End-to-end ingest: schedule, Statcast pitches, transactions and people."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from statecast.config import Settings
from statecast.ingest.http_client import open_client
from statecast.ingest.savant import StatcastIngestSummary, ingest_statcast
from statecast.ingest.statsapi import (
    fetch_people,
    fetch_schedule,
    fetch_transactions,
)
from statecast.storage.parquet_cache import statcast_path, statsapi_path, upsert_parquet

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestSummary:
    """Outcome of one ingest run.

    Attributes:
        games: Played games found in the schedule.
        statcast: Statcast ingest outcome.
        transactions: Player transactions fetched.
        new_players: Players added to the people cache.
    """

    games: int
    statcast: StatcastIngestSummary
    transactions: int
    new_players: int


def upsert_by_season(
    frame: pl.DataFrame,
    raw_dir: Path,
    dataset: str,
    date_column: str,
    key: Sequence[str],
) -> None:
    """Upsert a Stats API frame into its per-season cache files.

    Rows with a null ``date_column`` cannot be assigned a season and are
    skipped with a warning.

    Args:
        frame: Rows to cache.
        raw_dir: Root of the raw cache.
        dataset: Dataset name, such as ``schedule``.
        date_column: Date column that decides each row's season.
        key: Columns that identify a row.
    """
    dated = frame.filter(pl.col(date_column).is_not_null())
    if dated.height < frame.height:
        logger.warning(
            "Skipped %d %s rows with no %s",
            frame.height - dated.height,
            dataset,
            date_column,
        )
    for (season,), season_frame in dated.group_by(pl.col(date_column).dt.year()):
        upsert_parquet(season_frame, statsapi_path(raw_dir, dataset, int(season)), key)


def find_unknown_player_ids(raw_dir: Path, game_dates: Sequence[date]) -> list[int]:
    """List players in cached Statcast days who are not yet in the people cache.

    Args:
        raw_dir: Root of the raw cache.
        game_dates: Game dates whose cached pitches to scan.

    Returns:
        Sorted MLBAM ids of pitchers and batters missing from the people cache.
    """
    cached_days = [
        path
        for path in (statcast_path(raw_dir, game_date) for game_date in game_dates)
        if path.exists()
    ]
    if not cached_days:
        return []
    pitches = pl.scan_parquet(cached_days, hive_partitioning=False)
    seen_ids = pl.concat(
        [
            pitches.select(pl.col("pitcher").alias("player_id")),
            pitches.select(pl.col("batter").alias("player_id")),
        ]
    ).unique()
    people_path = statsapi_path(raw_dir, "people")
    if people_path.exists():
        known_ids = pl.scan_parquet(people_path).select("player_id")
        seen_ids = seen_ids.join(known_ids, on="player_id", how="anti")
    return sorted(seen_ids.drop_nulls().collect()["player_id"].to_list())


async def run_ingest(
    settings: Settings, start: date, end: date, today: date | None = None
) -> IngestSummary:
    """Ingest every data source for a date range into the raw cache.

    The range is clamped to end yesterday, because games still in progress
    would otherwise be cached incomplete. Reruns only fetch what is missing.

    Args:
        settings: Project settings.
        start: First date.
        end: Last date (inclusive).
        today: Today's date; defaults to the system date (overridable for tests).

    Returns:
        Counts of what was ingested.

    Raises:
        ValueError: If ``start`` is after the clamped end date.
    """
    last_complete_day = (today or date.today()) - timedelta(days=1)
    if end > last_complete_day:
        logger.warning("Clamping end date %s to %s", end, last_complete_day)
        end = last_complete_day
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    raw_dir = settings.paths.raw_dir
    async with open_client(settings.ingest) as client:
        schedule = await fetch_schedule(client, start, end, settings.ingest)
        upsert_by_season(schedule, raw_dir, "schedule", "game_date", ("game_pk",))
        game_dates: list[date] = schedule["game_date"].unique().to_list()
        statcast_summary = await ingest_statcast(
            client, game_dates, raw_dir, settings.ingest
        )
        transactions = await fetch_transactions(client, start, end, settings.ingest)
        upsert_by_season(
            transactions,
            raw_dir,
            "transactions",
            "transaction_date",
            ("transaction_id",),
        )
        unknown_ids = find_unknown_player_ids(raw_dir, game_dates)
        people = await fetch_people(client, unknown_ids, settings.ingest)
    if not people.is_empty():
        upsert_parquet(people, statsapi_path(raw_dir, "people"), ("player_id",))
    return IngestSummary(
        games=schedule.height,
        statcast=statcast_summary,
        transactions=transactions.height,
        new_players=people.height,
    )
