"""Baseball Savant Statcast Search CSV ingestion, one game date per request."""

import functools
import io
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
import polars as pl

from statecast.config import IngestSettings
from statecast.ingest.http_client import RetryableStatusError, get_bytes, run_bounded
from statecast.storage.normalize import PITCH_KEY
from statecast.storage.parquet_cache import statcast_path, write_parquet_atomic

logger = logging.getLogger(__name__)

SAVANT_CSV_URL = "https://baseballsavant.mlb.com/statcast_search/csv"

STATCAST_SCHEMA: dict[str, pl.DataType] = {
    "game_pk": pl.Int32(),
    "game_date": pl.Date(),
    "game_type": pl.String(),
    "at_bat_number": pl.Int16(),
    "pitch_number": pl.Int16(),
    "pitcher": pl.Int32(),
    "batter": pl.Int32(),
    "inning": pl.Int8(),
    "inning_topbot": pl.String(),
    "outs_when_up": pl.Int8(),
    "balls": pl.Int8(),
    "strikes": pl.Int8(),
    "n_thruorder_pitcher": pl.Int8(),
    "pitcher_days_since_prev_game": pl.Int16(),
    "pitch_type": pl.String(),
    "pitch_name": pl.String(),
    "release_speed": pl.Float32(),
    "release_spin_rate": pl.Float32(),
    "spin_axis": pl.Float32(),
    "pfx_x": pl.Float32(),
    "pfx_z": pl.Float32(),
    "api_break_x_arm": pl.Float32(),
    "api_break_z_with_gravity": pl.Float32(),
    "release_pos_x": pl.Float32(),
    "release_pos_y": pl.Float32(),
    "release_pos_z": pl.Float32(),
    "release_extension": pl.Float32(),
    "arm_angle": pl.Float32(),
    "vx0": pl.Float64(),
    "vy0": pl.Float64(),
    "vz0": pl.Float64(),
    "ax": pl.Float64(),
    "ay": pl.Float64(),
    "az": pl.Float64(),
    "plate_x": pl.Float32(),
    "plate_z": pl.Float32(),
    "sz_top": pl.Float32(),
    "sz_bot": pl.Float32(),
    "zone": pl.Int8(),
    "description": pl.String(),
    "type": pl.String(),
    "events": pl.String(),
    "launch_speed": pl.Float32(),
    "launch_angle": pl.Float32(),
    "delta_run_exp": pl.Float64(),
    "bat_speed": pl.Float32(),
    "swing_length": pl.Float32(),
    "attack_angle": pl.Float32(),
    "attack_direction": pl.Float32(),
    "swing_path_tilt": pl.Float32(),
}


@dataclass(frozen=True, slots=True)
class StatcastIngestSummary:
    """Outcome of a Statcast ingest run.

    Attributes:
        rows_written: Pitch rows written to the cache.
        days_written: Game dates newly cached.
        failed_dates: Game dates that could not be fetched or parsed.
    """

    rows_written: int
    days_written: int
    failed_dates: tuple[date, ...]


def build_query(game_date: date, game_types: Sequence[str]) -> dict[str, str]:
    """Build Savant query parameters for every pitch on one date.

    Args:
        game_date: Game date to request.
        game_types: Game-type codes, such as ``("R", "F")``.

    Returns:
        Query parameters for the Statcast Search CSV endpoint.
    """
    day = game_date.isoformat()
    return {
        "all": "true",
        "type": "details",
        "player_type": "pitcher",
        "hfGT": "".join(f"{game_type}|" for game_type in game_types),
        "game_date_gt": day,
        "game_date_lt": day,
    }


def parse_statcast_csv(payload: bytes) -> pl.DataFrame:
    r"""Parse a Savant CSV response into the typed Statcast schema.

    Every column in ``STATCAST_SCHEMA`` is present in the result; columns the
    response lacks (bat tracking before mid-2023, for example) are null. Values
    that fail to cast become null. Duplicate pitches are dropped.

    Args:
        payload: Raw CSV bytes from Savant.

    Returns:
        One row per pitch with the ``STATCAST_SCHEMA`` columns and types.

    Raises:
        ValueError: If the response is not a Statcast CSV.

    Example:
        >>> frame = parse_statcast_csv(b"game_pk,at_bat_number,pitch_number\n1,1,1\n")
        >>> frame.select("game_pk", "release_speed").row(0)
        (1, None)
    """
    if not payload.strip():
        return pl.DataFrame(schema=STATCAST_SCHEMA)
    raw_frame = pl.read_csv(io.BytesIO(payload), infer_schema=False)
    raw_frame = raw_frame.rename(lambda name: name.strip().lstrip("\ufeff"))
    missing_keys = [column for column in PITCH_KEY if column not in raw_frame.columns]
    if missing_keys:
        raise ValueError(f"Response is not a Statcast CSV; missing {missing_keys}")
    return raw_frame.select(
        _typed_column(column, dtype, column in raw_frame.columns)
        for column, dtype in STATCAST_SCHEMA.items()
    ).unique(subset=PITCH_KEY, keep="first", maintain_order=True)


def _typed_column(column: str, dtype: pl.DataType, present: bool) -> pl.Expr:
    """Build the expression that types one raw string column.

    Args:
        column: Column name.
        dtype: Target type.
        present: Whether the raw response has the column.

    Returns:
        A typed expression; null when the column is absent or a value is bad.
    """
    if not present:
        return pl.lit(None, dtype=dtype).alias(column)
    if dtype == pl.Date():
        return pl.col(column).str.to_date(strict=False)
    return pl.col(column).cast(dtype, strict=False)


async def fetch_statcast_day(
    client: httpx.AsyncClient, game_date: date, settings: IngestSettings
) -> pl.DataFrame:
    """Download and parse every pitch thrown on one date.

    Args:
        client: Open HTTP client.
        game_date: Game date to request.
        settings: Ingest settings (game types and retries).

    Returns:
        Parsed pitches for the date (possibly empty).
    """
    payload = await get_bytes(
        client,
        SAVANT_CSV_URL,
        build_query(game_date, settings.game_types),
        settings.max_retries,
    )
    return parse_statcast_csv(payload)


async def _cache_statcast_day(
    client: httpx.AsyncClient,
    game_date: date,
    raw_dir: Path,
    settings: IngestSettings,
) -> tuple[date, int | None]:
    """Fetch, parse and cache one date, logging instead of raising on failure.

    Args:
        client: Open HTTP client.
        game_date: Game date to request.
        raw_dir: Root of the raw cache.
        settings: Ingest settings.

    Returns:
        The date and its row count, or ``None`` as the count on failure.
    """
    try:
        frame = await fetch_statcast_day(client, game_date, settings)
    except (
        httpx.HTTPError,
        RetryableStatusError,
        ValueError,
        pl.exceptions.PolarsError,
    ) as error:
        logger.error("Statcast fetch failed for %s: %s", game_date, error)
        return game_date, None
    write_parquet_atomic(frame, statcast_path(raw_dir, game_date))
    return game_date, frame.height


async def ingest_statcast(
    client: httpx.AsyncClient,
    game_dates: Iterable[date],
    raw_dir: Path,
    settings: IngestSettings,
) -> StatcastIngestSummary:
    """Fetch and cache Statcast pitches for each uncached game date.

    Dates already in the cache are skipped, so reruns only fetch what is
    missing. A failed date is logged and reported, and the run continues.

    Args:
        client: Open HTTP client.
        game_dates: Dates to ingest.
        raw_dir: Root of the raw cache.
        settings: Ingest settings.

    Returns:
        Rows and days written, plus the dates that failed.
    """
    pending_dates = sorted(
        game_date
        for game_date in set(game_dates)
        if not statcast_path(raw_dir, game_date).exists()
    )
    outcomes = await run_bounded(
        [
            functools.partial(_cache_statcast_day, client, game_date, raw_dir, settings)
            for game_date in pending_dates
        ],
        settings.max_concurrency,
        "Statcast game dates",
    )
    failed_dates = tuple(sorted(day for day, rows in outcomes if rows is None))
    rows_written = sum(rows for _, rows in outcomes if rows is not None)
    return StatcastIngestSummary(
        rows_written=rows_written,
        days_written=len(outcomes) - len(failed_dates),
        failed_dates=failed_dates,
    )
