"""Build and update the normalized DuckDB warehouse from the raw cache."""

import logging
import re
from collections.abc import Iterable
from pathlib import Path

import duckdb
import polars as pl

from statecast.storage import normalize
from statecast.storage.parquet_cache import (
    STATCAST_DIR,
    statcast_season_glob,
    statsapi_path,
)

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
WAREHOUSE_TABLES = frozenset(
    {
        "player",
        "team",
        "venue",
        "game",
        "pitch_type",
        "pitcher_appearance",
        "plate_appearance",
        "pitch",
        "swing",
        "player_transaction",
        "model_run",
        "model_parameter",
        "latent_state",
        "jump_run",
        "jump_state",
    }
)
SEASON_DIR_PATTERN = re.compile(r"season=(\d{4})")


def initialize_schema(connection: duckdb.DuckDBPyConnection) -> None:
    """Create any missing warehouse tables.

    Args:
        connection: Open DuckDB connection.
    """
    connection.execute(SCHEMA_PATH.read_text(encoding="utf-8"))


def upsert_table(
    connection: duckdb.DuckDBPyConnection, table: str, frame: pl.DataFrame
) -> int:
    """Insert rows into a table, replacing rows with the same primary key.

    Columns are matched by name, so the frame's column order does not matter.

    Args:
        connection: Open DuckDB connection.
        table: Warehouse table name.
        frame: Rows to write.

    Returns:
        Number of rows written.

    Raises:
        ValueError: If ``table`` is not a warehouse table.
    """
    if table not in WAREHOUSE_TABLES:
        raise ValueError(f"Unknown warehouse table {table!r}")
    if frame.is_empty():
        return 0
    connection.register("incoming_rows", frame.to_arrow())
    try:
        connection.execute(
            f"INSERT OR REPLACE INTO {table} BY NAME SELECT * FROM incoming_rows"
        )
    finally:
        connection.unregister("incoming_rows")
    return frame.height


def cached_seasons(raw_dir: Path) -> list[int]:
    """List seasons that have cached Statcast days.

    Args:
        raw_dir: Root of the raw cache.

    Returns:
        Sorted season years.
    """
    statcast_dir = raw_dir / STATCAST_DIR
    if not statcast_dir.exists():
        return []
    return sorted(
        int(match.group(1))
        for child in statcast_dir.iterdir()
        if child.is_dir() and (match := SEASON_DIR_PATTERN.fullmatch(child.name))
    )


def season_tables(raw_dir: Path, season: int) -> dict[str, pl.DataFrame]:
    """Read one season from the raw cache and normalize it into tables.

    Args:
        raw_dir: Root of the raw cache.
        season: Season year.

    Returns:
        Warehouse table name to rows, for every source cached for the season.
    """
    tables: dict[str, pl.DataFrame] = {}
    schedule_path = statsapi_path(raw_dir, "schedule", season)
    if schedule_path.exists():
        schedule = pl.read_parquet(schedule_path)
        tables["venue"] = normalize.venue_table(schedule)
        tables["team"] = normalize.team_table(schedule)
        tables["game"] = normalize.game_table(schedule)
    statcast_dir = raw_dir / STATCAST_DIR / f"season={season}"
    if statcast_dir.exists() and any(statcast_dir.glob("date=*.parquet")):
        statcast = pl.read_parquet(
            statcast_season_glob(raw_dir, season), hive_partitioning=False
        )
        tables["pitch_type"] = normalize.pitch_type_table(statcast)
        tables["pitcher_appearance"] = normalize.pitcher_appearance_table(statcast)
        tables["plate_appearance"] = normalize.plate_appearance_table(statcast)
        tables["pitch"] = normalize.pitch_table(statcast)
        tables["swing"] = normalize.swing_table(statcast)
    transactions_path = statsapi_path(raw_dir, "transactions", season)
    if transactions_path.exists():
        tables["player_transaction"] = pl.read_parquet(transactions_path)
    return tables


def build_warehouse(
    warehouse_path: Path, raw_dir: Path, seasons: Iterable[int] | None = None
) -> dict[str, int]:
    """Load the raw cache into the warehouse, one season per transaction.

    Loading is idempotent: rerunning replaces rows by primary key. A season
    that fails to load is rolled back and the error is raised.

    Args:
        warehouse_path: DuckDB database file (created if missing).
        raw_dir: Root of the raw cache.
        seasons: Seasons to load; defaults to every cached season.

    Returns:
        Rows written per table, summed across seasons.

    Raises:
        duckdb.Error: If a season fails to load.
    """
    seasons_to_load = (
        sorted(seasons) if seasons is not None else cached_seasons(raw_dir)
    )
    rows_written: dict[str, int] = {}
    warehouse_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(warehouse_path)) as connection:
        initialize_schema(connection)
        people_path = statsapi_path(raw_dir, "people")
        if people_path.exists():
            rows_written["player"] = upsert_table(
                connection, "player", pl.read_parquet(people_path)
            )
        for season in seasons_to_load:
            connection.begin()
            try:
                for table, frame in season_tables(raw_dir, season).items():
                    written = upsert_table(connection, table, frame)
                    rows_written[table] = rows_written.get(table, 0) + written
            except duckdb.Error:
                connection.rollback()
                logger.error("Warehouse load failed for season %d", season)
                raise
            connection.commit()
            logger.info("Loaded season %d into %s", season, warehouse_path)
    return rows_written
