from datetime import date
from pathlib import Path

import polars as pl

from statecast.storage.parquet_cache import (
    statcast_path,
    statcast_season_glob,
    statsapi_path,
    upsert_parquet,
    write_parquet_atomic,
)


def test_statcast_path_partitions_by_season_and_date() -> None:
    path = statcast_path(Path("raw"), date(2026, 6, 1))

    assert path == Path("raw/statcast/season=2026/date=2026-06-01.parquet")
    assert statcast_season_glob(Path("raw"), 2026).endswith("date=*.parquet")


def test_statsapi_path_with_and_without_season() -> None:
    assert statsapi_path(Path("raw"), "schedule", 2026) == Path(
        "raw/statsapi/schedule/season=2026.parquet"
    )
    assert statsapi_path(Path("raw"), "people") == Path(
        "raw/statsapi/people/people.parquet"
    )


def test_write_parquet_atomic_leaves_no_temporary_file(output_dir: Path) -> None:
    path = output_dir / "nested" / "frame.parquet"

    write_parquet_atomic(pl.DataFrame({"value": [1, 2]}), path)

    assert pl.read_parquet(path)["value"].to_list() == [1, 2]
    assert [child.name for child in path.parent.iterdir()] == ["frame.parquet"]


def test_upsert_parquet_keeps_newest_row_per_key(output_dir: Path) -> None:
    path = output_dir / "people.parquet"
    upsert_parquet(pl.DataFrame({"id": [1, 2], "name": ["old", "b"]}), path, ["id"])

    row_count = upsert_parquet(
        pl.DataFrame({"id": [1, 3], "name": ["new", "c"]}), path, ["id"]
    )

    merged = pl.read_parquet(path).sort("id")
    assert row_count == 3
    assert merged.rows() == [(1, "new"), (2, "b"), (3, "c")]
