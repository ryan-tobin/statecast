from datetime import date
from pathlib import Path

import httpx
import polars as pl
import pytest
import respx

from statecast.config import IngestSettings
from statecast.ingest.http_client import open_client
from statecast.ingest.savant import (
    SAVANT_CSV_URL,
    STATCAST_SCHEMA,
    build_query,
    ingest_statcast,
    parse_statcast_csv,
)
from statecast.storage.parquet_cache import statcast_path, write_parquet_atomic

SAMPLE_CSV = (
    "\ufeffpitch_type,game_date,release_speed,game_pk,at_bat_number,pitch_number,"
    "pitcher,batter,bat_speed,zone\n"
    "FF,2026-06-01,97.4,777001,1,1,605400,592450,74.1,5\n"
    "FF,2026-06-01,97.4,777001,1,1,605400,592450,74.1,5\n"
    "SL,2026-06-01,,777001,1,2,605400,592450,,not-a-zone\n"
).encode()


def test_build_query_requests_one_day_of_listed_game_types() -> None:
    query = build_query(date(2026, 6, 1), ("R", "W"))

    assert query["game_date_gt"] == query["game_date_lt"] == "2026-06-01"
    assert query["hfGT"] == "R|W|"
    assert query["type"] == "details"


def test_parse_statcast_csv_types_dedupes_and_fills_missing_columns() -> None:
    frame = parse_statcast_csv(SAMPLE_CSV)

    assert frame.schema == pl.Schema(STATCAST_SCHEMA)
    assert frame.height == 2
    first, second = frame.select(
        "pitch_type", "game_date", "release_speed", "zone"
    ).rows()
    assert first == ("FF", date(2026, 6, 1), pytest.approx(97.4), 5)
    assert second == ("SL", date(2026, 6, 1), None, None)
    assert frame["arm_angle"].null_count() == 2


def test_parse_statcast_csv_empty_payload_returns_typed_empty_frame() -> None:
    frame = parse_statcast_csv(b"  \n")

    assert frame.is_empty()
    assert frame.schema == pl.Schema(STATCAST_SCHEMA)


def test_parse_statcast_csv_rejects_non_statcast_response() -> None:
    with pytest.raises(ValueError, match="not a Statcast CSV"):
        parse_statcast_csv(b"error,message\n1,rate limited\n")


@respx.mock
async def test_ingest_statcast_skips_cached_days_and_reports_failures(
    output_dir: Path, ingest_settings: IngestSettings
) -> None:
    raw_dir = output_dir / "raw"
    cached_day, new_day, failing_day = (
        date(2026, 6, 1),
        date(2026, 6, 2),
        date(2026, 6, 3),
    )
    write_parquet_atomic(
        pl.DataFrame(schema=STATCAST_SCHEMA), statcast_path(raw_dir, cached_day)
    )
    new_day_csv = SAMPLE_CSV.replace(b"2026-06-01", b"2026-06-02")

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.params["game_date_gt"] == new_day.isoformat():
            return httpx.Response(200, content=new_day_csv)
        return httpx.Response(404)

    route = respx.get(SAVANT_CSV_URL).mock(side_effect=respond)

    async with open_client(ingest_settings) as client:
        summary = await ingest_statcast(
            client,
            [cached_day, new_day, failing_day, new_day],
            raw_dir,
            ingest_settings,
        )

    assert route.call_count == 2
    assert summary.rows_written == 2
    assert summary.days_written == 1
    assert summary.failed_dates == (failing_day,)
    assert pl.read_parquet(statcast_path(raw_dir, new_day)).height == 2
    assert not statcast_path(raw_dir, failing_day).exists()
