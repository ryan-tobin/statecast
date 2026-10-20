from datetime import date
from pathlib import Path

import duckdb
import polars as pl
import pytest

from statecast.features.appearance import hitter_appearances, pitcher_appearances
from statecast.storage.warehouse import initialize_schema, upsert_table

GAMES = pl.DataFrame(
    {
        "game_pk": [1, 2, 3],
        "game_date": [date(2026, 4, 1), date(2026, 4, 6), date(2026, 10, 10)],
        "season": [2026, 2026, 2026],
        "game_type": ["R", "R", "D"],
    }
)


def pitch_rows() -> pl.DataFrame:
    """Pitcher 10 throws FF, SL, CH, CU (CU is fourth); pitcher 20 throws FF."""
    rows: list[dict[str, object]] = []
    for game_pk in (1, 2, 3):
        usage = {"FF": 4, "SL": 3, "CH": 2, "CU": 1}
        speeds = {"FF": 95.0, "SL": 86.0, "CH": 87.0, "CU": 79.0}
        for pitch_type, count in usage.items():
            for index in range(count):
                rows.append(
                    {
                        "game_pk": game_pk,
                        "at_bat_number": len(rows) + 1,
                        "pitch_number": 1,
                        "pitcher_id": 10,
                        "batter_id": 30,
                        "pitch_type": pitch_type,
                        "release_speed": speeds[pitch_type] + index,
                        "description": "swinging_strike",
                        "plate_reference": "middle",
                    }
                )
    rows.append(
        {
            "game_pk": 1,
            "at_bat_number": len(rows) + 1,
            "pitch_number": 1,
            "pitcher_id": 20,
            "batter_id": 30,
            "pitch_type": "FF",
            "release_speed": 99.0,
            "description": "foul_bunt",
            "plate_reference": "middle",
        }
    )
    return pl.DataFrame(rows)


@pytest.fixture
def warehouse(output_dir: Path) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(str(output_dir / "features.duckdb"))
    initialize_schema(connection)
    upsert_table(connection, "game", GAMES)
    pitches = pitch_rows()
    upsert_table(connection, "pitch", pitches)
    swings = pitches.select(
        "game_pk",
        "at_bat_number",
        "pitch_number",
        bat_speed=pl.when(pl.col("pitch_type") == "SL")
        .then(pl.lit(45.0))
        .otherwise(pl.col("release_speed") - 20.0),
    )
    upsert_table(connection, "swing", swings)
    return connection


def test_pitcher_appearances_summarize_regular_season_games(
    warehouse: duckdb.DuckDBPyConnection,
) -> None:
    frame = pitcher_appearances(warehouse, "FF", [2026], min_appearances=1)

    pitcher = frame.filter(pl.col("player_id") == 10)
    assert pitcher["game_date"].to_list() == [date(2026, 4, 1), date(2026, 4, 6)]
    assert pitcher["sample_size"].to_list() == [4, 4]
    assert pitcher["mean"].to_list() == pytest.approx([96.5, 96.5])
    assert pitcher["within_ss"].to_list() == pytest.approx([5.0, 5.0])


def test_pitcher_appearances_skip_pitch_types_outside_top_three(
    warehouse: duckdb.DuckDBPyConnection,
) -> None:
    assert pitcher_appearances(warehouse, "CU", [2026], min_appearances=1).is_empty()


def test_pitcher_appearances_drop_pitchers_with_too_few_games(
    warehouse: duckdb.DuckDBPyConnection,
) -> None:
    frame = pitcher_appearances(warehouse, "FF", [2026], min_appearances=2)

    assert frame["player_id"].unique().to_list() == [10]


def test_hitter_appearances_keep_competitive_swings_only(
    warehouse: duckdb.DuckDBPyConnection,
) -> None:
    frame = hitter_appearances(warehouse, [2026], min_appearances=1)

    first_game = frame.row(0, named=True)
    assert first_game["game_date"] == date(2026, 4, 1)
    assert first_game["sample_size"] == 7
    assert frame.height == 2


def test_unsupported_metrics_are_rejected(
    warehouse: duckdb.DuckDBPyConnection,
) -> None:
    with pytest.raises(ValueError, match="pitcher metric"):
        pitcher_appearances(warehouse, "FF", [2026], metric="description")
    with pytest.raises(ValueError, match="hitter metric"):
        hitter_appearances(warehouse, [2026], metric="release_speed")
