from datetime import date
from pathlib import Path

import duckdb
import polars as pl
import pytest

from conftest import StatcastFactory, schedule_game, schedule_payload
from statecast.ingest.statsapi import parse_schedule
from statecast.storage import warehouse
from statecast.storage.parquet_cache import (
    statcast_path,
    statsapi_path,
    write_parquet_atomic,
)


def build_cache(raw_dir: Path, make_statcast: StatcastFactory) -> None:
    write_parquet_atomic(
        make_statcast(
            [
                {
                    "game_pk": 777001,
                    "game_date": date(2026, 6, 1),
                    "at_bat_number": 1,
                    "pitch_number": pitch_number,
                    "pitcher": 605400,
                    "batter": 592450,
                    "pitch_type": "FF",
                    "pitch_name": "4-Seam Fastball",
                    "bat_speed": 74.0 if pitch_number == 2 else None,
                }
                for pitch_number in (1, 2)
            ]
        ),
        statcast_path(raw_dir, date(2026, 6, 1)),
    )
    write_parquet_atomic(
        parse_schedule(schedule_payload([schedule_game(777001, "2026-06-01")]), ("R",)),
        statsapi_path(raw_dir, "schedule", 2026),
    )
    write_parquet_atomic(
        pl.DataFrame(
            {
                "player_id": [605400],
                "full_name": ["Aaron Nola"],
                "birth_date": [date(1993, 6, 4)],
                "throws": ["R"],
                "bats": ["R"],
            },
            schema_overrides={"player_id": pl.Int32},
        ),
        statsapi_path(raw_dir, "people"),
    )


def test_upsert_table_replaces_rows_by_primary_key(output_dir: Path) -> None:
    with duckdb.connect(str(output_dir / "upsert.duckdb")) as connection:
        warehouse.initialize_schema(connection)
        warehouse.upsert_table(
            connection, "team", pl.DataFrame({"name": ["Old"], "team_id": [143]})
        )

        written = warehouse.upsert_table(
            connection, "team", pl.DataFrame({"team_id": [143], "name": ["New"]})
        )

        assert written == 1
        assert connection.execute("SELECT team_id, name FROM team").fetchall() == [
            (143, "New")
        ]


def test_upsert_table_rejects_unknown_table(output_dir: Path) -> None:
    with (
        duckdb.connect(str(output_dir / "unknown.duckdb")) as connection,
        pytest.raises(ValueError, match="Unknown warehouse table"),
    ):
        warehouse.upsert_table(connection, "pitch; DROP TABLE x", pl.DataFrame())


def test_upsert_table_skips_empty_frame(output_dir: Path) -> None:
    with duckdb.connect(str(output_dir / "empty.duckdb")) as connection:
        warehouse.initialize_schema(connection)

        assert warehouse.upsert_table(connection, "team", pl.DataFrame()) == 0


def test_cached_seasons_reads_partition_folders(output_dir: Path) -> None:
    for name in ("season=2025", "season=2026", "notes"):
        (output_dir / "statcast" / name).mkdir(parents=True)

    assert warehouse.cached_seasons(output_dir) == [2025, 2026]
    assert warehouse.cached_seasons(output_dir / "missing") == []


def test_build_warehouse_loads_every_table_and_is_idempotent(
    output_dir: Path, make_statcast: StatcastFactory
) -> None:
    raw_dir = output_dir / "raw"
    warehouse_path = output_dir / "warehouse" / "statecast.duckdb"
    build_cache(raw_dir, make_statcast)

    warehouse.build_warehouse(warehouse_path, raw_dir)
    rows_written = warehouse.build_warehouse(warehouse_path, raw_dir, [2026])

    assert rows_written["pitch"] == 2
    assert rows_written["swing"] == 1
    with duckdb.connect(str(warehouse_path), read_only=True) as connection:
        counts = {
            table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()
            for table in ("pitch", "swing", "game", "player", "plate_appearance")
        }
        reference = connection.execute(
            "SELECT DISTINCT plate_reference FROM pitch"
        ).fetchall()
    assert counts == {
        "pitch": (2,),
        "swing": (1,),
        "game": (1,),
        "player": (1,),
        "plate_appearance": (1,),
    }
    assert reference == [("middle",)]


def test_build_warehouse_rolls_back_failed_season(
    output_dir: Path, make_statcast: StatcastFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = output_dir / "raw"
    warehouse_path = output_dir / "warehouse" / "statecast.duckdb"
    build_cache(raw_dir, make_statcast)
    original_upsert = warehouse.upsert_table

    def failing_upsert(
        connection: duckdb.DuckDBPyConnection, table: str, frame: pl.DataFrame
    ) -> int:
        if table == "pitch":
            raise duckdb.ConstraintException("simulated failure")
        return original_upsert(connection, table, frame)

    monkeypatch.setattr(warehouse, "upsert_table", failing_upsert)

    with pytest.raises(duckdb.ConstraintException):
        warehouse.build_warehouse(warehouse_path, raw_dir, [2026])

    with duckdb.connect(str(warehouse_path), read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM game").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM player").fetchone() == (1,)
