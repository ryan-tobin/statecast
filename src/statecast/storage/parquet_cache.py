"""Immutable raw Parquet cache: path layout, atomic writes and keyed upserts."""

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import polars as pl

STATCAST_DIR = "statcast"
STATSAPI_DIR = "statsapi"


def statcast_path(raw_dir: Path, game_date: date) -> Path:
    """Return the cache path for one day of Statcast pitches.

    Args:
        raw_dir: Root of the raw cache.
        game_date: Game date.

    Returns:
        ``<raw_dir>/statcast/season=YYYY/date=YYYY-MM-DD.parquet``.
    """
    season_dir = raw_dir / STATCAST_DIR / f"season={game_date.year}"
    return season_dir / f"date={game_date.isoformat()}.parquet"


def statcast_season_glob(raw_dir: Path, season: int) -> str:
    """Return a glob matching every cached Statcast day in a season.

    Args:
        raw_dir: Root of the raw cache.
        season: Season year.

    Returns:
        A glob pattern string for ``pl.scan_parquet``.
    """
    return str(raw_dir / STATCAST_DIR / f"season={season}" / "date=*.parquet")


def statsapi_path(raw_dir: Path, dataset: str, season: int | None = None) -> Path:
    """Return the cache path for an MLB Stats API dataset.

    Args:
        raw_dir: Root of the raw cache.
        dataset: Dataset name, such as ``schedule``, ``transactions`` or ``people``.
        season: Season partition, or ``None`` for an unpartitioned dataset.

    Returns:
        ``<raw_dir>/statsapi/<dataset>/season=YYYY.parquet`` when a season is
        given, else ``<raw_dir>/statsapi/<dataset>/<dataset>.parquet``.
    """
    file_name = f"{dataset}.parquet" if season is None else f"season={season}.parquet"
    return raw_dir / STATSAPI_DIR / dataset / file_name


def write_parquet_atomic(frame: pl.DataFrame, path: Path) -> None:
    """Write a frame to Parquet so a crash never leaves a partial file.

    The frame is written to a sibling temporary file and then renamed over the
    target, which is atomic on POSIX filesystems.

    Args:
        frame: Data to write.
        path: Destination file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    frame.write_parquet(temporary_path, compression="zstd")
    temporary_path.replace(path)


def upsert_parquet(frame: pl.DataFrame, path: Path, key: Sequence[str]) -> int:
    """Merge rows into a Parquet file, keeping the newest row for each key.

    Args:
        frame: Incoming rows, with the same schema as the file.
        path: Parquet file to create or update.
        key: Columns that identify a row.

    Returns:
        Row count of the file after the merge.
    """
    merged = frame
    if path.exists():
        merged = pl.concat([pl.read_parquet(path), frame], how="vertical_relaxed")
    merged = merged.unique(subset=list(key), keep="last", maintain_order=True)
    write_parquet_atomic(merged, path)
    return merged.height
