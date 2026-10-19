from datetime import date
from pathlib import Path

import httpx
import orjson
import polars as pl
import pytest
import respx

from conftest import StatcastFactory, schedule_game, schedule_payload
from statecast.config import Settings
from statecast.ingest.pipeline import (
    find_unknown_player_ids,
    run_ingest,
    upsert_by_season,
)
from statecast.ingest.savant import SAVANT_CSV_URL
from statecast.ingest.statsapi import STATSAPI_BASE_URL
from statecast.storage.parquet_cache import (
    statcast_path,
    statsapi_path,
    write_parquet_atomic,
)

DAY_CSV = (
    b"game_pk,game_date,at_bat_number,pitch_number,pitcher,batter\n"
    b"777001,2026-06-01,1,1,605400,592450\n"
)


def test_upsert_by_season_splits_rows_and_skips_undated(output_dir: Path) -> None:
    frame = pl.DataFrame(
        {
            "transaction_id": [1, 2, 3],
            "transaction_date": [date(2025, 9, 1), date(2026, 4, 1), None],
        }
    )

    upsert_by_season(
        frame, output_dir, "transactions", "transaction_date", ("transaction_id",)
    )

    for season, expected_ids in ((2025, [1]), (2026, [2])):
        cached = pl.read_parquet(statsapi_path(output_dir, "transactions", season))
        assert cached["transaction_id"].to_list() == expected_ids


def test_find_unknown_player_ids_excludes_cached_people(
    output_dir: Path, make_statcast: StatcastFactory
) -> None:
    game_date = date(2026, 6, 1)
    write_parquet_atomic(
        make_statcast(
            [
                {
                    "game_pk": 1,
                    "at_bat_number": 1,
                    "pitch_number": 1,
                    "pitcher": 10,
                    "batter": 20,
                },
                {
                    "game_pk": 1,
                    "at_bat_number": 2,
                    "pitch_number": 1,
                    "pitcher": 10,
                    "batter": 30,
                },
            ]
        ),
        statcast_path(output_dir, game_date),
    )
    write_parquet_atomic(
        pl.DataFrame({"player_id": [20]}, schema={"player_id": pl.Int32}),
        statsapi_path(output_dir, "people"),
    )

    unknown = find_unknown_player_ids(output_dir, [game_date, date(2026, 6, 2)])

    assert unknown == [10, 30]


def test_find_unknown_player_ids_without_cached_days(output_dir: Path) -> None:
    assert find_unknown_player_ids(output_dir, [date(2026, 6, 1)]) == []


@respx.mock
async def test_run_ingest_caches_every_source(settings: Settings) -> None:
    respx.get(f"{STATSAPI_BASE_URL}/schedule").mock(
        return_value=httpx.Response(
            200, content=schedule_payload([schedule_game(777001, "2026-06-01")])
        )
    )
    respx.get(SAVANT_CSV_URL).mock(return_value=httpx.Response(200, content=DAY_CSV))
    respx.get(f"{STATSAPI_BASE_URL}/transactions").mock(
        return_value=httpx.Response(200, content=b'{"transactions": []}')
    )
    people_route = respx.get(f"{STATSAPI_BASE_URL}/people").mock(
        return_value=httpx.Response(
            200,
            content=orjson.dumps({"people": [{"id": 605400}, {"id": 592450}]}),
        )
    )

    summary = await run_ingest(
        settings, date(2026, 6, 1), date(2026, 6, 30), today=date(2026, 6, 2)
    )

    raw_dir = settings.paths.raw_dir
    assert summary.games == 1
    assert summary.statcast.rows_written == 1
    assert summary.new_players == 2
    assert people_route.calls.last.request.url.params["personIds"] == "592450,605400"
    assert statsapi_path(raw_dir, "schedule", 2026).exists()
    assert pl.read_parquet(statsapi_path(raw_dir, "people")).height == 2


async def test_run_ingest_rejects_range_that_starts_after_yesterday(
    settings: Settings,
) -> None:
    with pytest.raises(ValueError, match="after"):
        await run_ingest(
            settings, date(2026, 6, 5), date(2026, 6, 9), today=date(2026, 6, 5)
        )
