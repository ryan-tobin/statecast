from datetime import UTC, date, datetime

import httpx
import orjson
import pytest
import respx

from conftest import schedule_game, schedule_payload
from statecast.config import IngestSettings
from statecast.ingest.http_client import open_client
from statecast.ingest.statsapi import (
    PEOPLE_BATCH_SIZE,
    STATSAPI_BASE_URL,
    fetch_people,
    fetch_schedule,
    fetch_transactions,
    month_windows,
    parse_people,
    parse_schedule,
    parse_transactions,
)


def test_month_windows_split_at_month_and_year_boundaries() -> None:
    windows = month_windows(date(2025, 12, 30), date(2026, 2, 3))

    assert windows == [
        (date(2025, 12, 30), date(2025, 12, 31)),
        (date(2026, 1, 1), date(2026, 1, 31)),
        (date(2026, 2, 1), date(2026, 2, 3)),
    ]


def test_month_windows_single_day() -> None:
    assert month_windows(date(2026, 6, 1), date(2026, 6, 1)) == [
        (date(2026, 6, 1), date(2026, 6, 1))
    ]


def test_month_windows_rejects_reversed_range() -> None:
    with pytest.raises(ValueError, match="after"):
        month_windows(date(2026, 6, 2), date(2026, 6, 1))


def test_parse_schedule_keeps_played_games_of_listed_types() -> None:
    payload = schedule_payload(
        [
            schedule_game(1, "2026-06-01"),
            schedule_game(2, "2026-06-01", detailed_state="Postponed"),
            schedule_game(3, "2026-03-01", game_type="S"),
        ]
    )

    schedule = parse_schedule(payload, ("R",))

    assert schedule["game_pk"].to_list() == [1]
    assert schedule["game_date"].item() == date(2026, 6, 1)
    assert schedule["first_pitch_at"].item() == datetime(2026, 6, 1, 23, 5, tzinfo=UTC)
    assert schedule["home_team_name"].item() == "Philadelphia Phillies"


def test_parse_schedule_handles_days_without_games() -> None:
    assert parse_schedule(schedule_payload([]), ("R",)).is_empty()


def test_parse_transactions_skips_entries_without_a_player() -> None:
    payload = orjson.dumps(
        {
            "transactions": [
                {
                    "id": 10,
                    "person": {"id": 605400},
                    "date": "2026-06-01",
                    "effectiveDate": "2026-05-30",
                    "typeCode": "SC",
                    "typeDesc": "Status Change",
                    "description": "placed on the 15-day injured list",
                },
                {"id": 11, "typeCode": "TR"},
            ]
        }
    )

    transactions = parse_transactions(payload)

    assert transactions["transaction_id"].to_list() == [10]
    assert transactions["effective_date"].item() == date(2026, 5, 30)


def test_parse_people_reads_handedness_and_birth_date() -> None:
    payload = orjson.dumps(
        {
            "people": [
                {
                    "id": 605400,
                    "fullName": "Aaron Nola",
                    "birthDate": "1993-06-04",
                    "pitchHand": {"code": "R"},
                    "batSide": {"code": "R"},
                },
                {"id": 1},
            ]
        }
    )

    people = parse_people(payload)

    assert people.row(0) == (605400, "Aaron Nola", date(1993, 6, 4), "R", "R")
    assert people.row(1) == (1, None, None, None, None)


@respx.mock
async def test_fetch_schedule_keeps_latest_listing_of_replayed_game(
    ingest_settings: IngestSettings,
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.params["startDate"] == "2026-05-01":
            return httpx.Response(
                200, content=schedule_payload([schedule_game(5, "2026-05-31")])
            )
        return httpx.Response(
            200, content=schedule_payload([schedule_game(5, "2026-06-15")])
        )

    route = respx.get(f"{STATSAPI_BASE_URL}/schedule").mock(side_effect=respond)

    async with open_client(ingest_settings) as client:
        schedule = await fetch_schedule(
            client, date(2026, 5, 20), date(2026, 6, 20), ingest_settings
        )

    assert route.call_count == 2
    assert schedule.height == 1
    assert schedule["game_date"].item() == date(2026, 6, 15)


@respx.mock
async def test_fetch_transactions_uses_month_day_year_dates(
    ingest_settings: IngestSettings,
) -> None:
    route = respx.get(f"{STATSAPI_BASE_URL}/transactions").mock(
        return_value=httpx.Response(200, content=b'{"transactions": []}')
    )

    async with open_client(ingest_settings) as client:
        transactions = await fetch_transactions(
            client, date(2026, 6, 1), date(2026, 6, 30), ingest_settings
        )

    assert transactions.is_empty()
    assert route.calls.last.request.url.params["startDate"] == "06/01/2026"


@respx.mock
async def test_fetch_people_batches_requests(ingest_settings: IngestSettings) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        ids = request.url.params["personIds"].split(",")
        body = {"people": [{"id": int(player_id)} for player_id in ids]}
        return httpx.Response(200, content=orjson.dumps(body))

    route = respx.get(f"{STATSAPI_BASE_URL}/people").mock(side_effect=respond)
    player_ids = list(range(1, PEOPLE_BATCH_SIZE + 51))

    async with open_client(ingest_settings) as client:
        people = await fetch_people(client, [*player_ids, 1], ingest_settings)

    assert route.call_count == 2
    assert sorted(people["player_id"].to_list()) == player_ids


@respx.mock
async def test_fetch_people_without_ids_makes_no_request(
    ingest_settings: IngestSettings,
) -> None:
    route = respx.get(f"{STATSAPI_BASE_URL}/people")

    async with open_client(ingest_settings) as client:
        people = await fetch_people(client, [], ingest_settings)

    assert people.is_empty()
    assert route.call_count == 0
