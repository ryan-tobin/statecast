"""MLB Stats API ingestion: schedule, transactions and people."""

import functools
from collections.abc import Callable, Iterable, Sequence
from datetime import date, timedelta

import httpx
import orjson
import polars as pl

from statecast.config import IngestSettings
from statecast.ingest.http_client import get_bytes, run_bounded

STATSAPI_BASE_URL = "https://statsapi.mlb.com/api/v1"
MLB_SPORT_ID = "1"
PEOPLE_BATCH_SIZE = 100
UNPLAYED_STATES = frozenset({"Postponed", "Cancelled"})

SCHEDULE_SCHEMA: dict[str, pl.DataType] = {
    "game_pk": pl.Int32(),
    "game_date": pl.String(),
    "season": pl.Int16(),
    "game_type": pl.String(),
    "detailed_state": pl.String(),
    "venue_id": pl.Int32(),
    "venue_name": pl.String(),
    "home_team_id": pl.Int32(),
    "home_team_name": pl.String(),
    "away_team_id": pl.Int32(),
    "away_team_name": pl.String(),
    "first_pitch_at": pl.String(),
}
TRANSACTION_SCHEMA: dict[str, pl.DataType] = {
    "transaction_id": pl.Int64(),
    "player_id": pl.Int32(),
    "transaction_date": pl.String(),
    "effective_date": pl.String(),
    "type_code": pl.String(),
    "type_desc": pl.String(),
    "description": pl.String(),
}
PEOPLE_SCHEMA: dict[str, pl.DataType] = {
    "player_id": pl.Int32(),
    "full_name": pl.String(),
    "birth_date": pl.String(),
    "throws": pl.String(),
    "bats": pl.String(),
}

type JsonParser = Callable[[bytes], pl.DataFrame]


def month_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Split an inclusive date range into calendar-month windows.

    Args:
        start: First date.
        end: Last date (inclusive).

    Returns:
        ``(first, last)`` pairs covering the range, one per calendar month.

    Raises:
        ValueError: If ``start`` is after ``end``.

    Example:
        >>> month_windows(date(2026, 3, 30), date(2026, 4, 2))
        [(datetime.date(2026, 3, 30), datetime.date(2026, 3, 31)), \
(datetime.date(2026, 4, 1), datetime.date(2026, 4, 2))]
    """
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    windows: list[tuple[date, date]] = []
    window_start = start
    while window_start <= end:
        next_month = (window_start.replace(day=28) + timedelta(days=4)).replace(day=1)
        window_end = min(next_month - timedelta(days=1), end)
        windows.append((window_start, window_end))
        window_start = next_month
    return windows


def parse_schedule(payload: bytes, game_types: Sequence[str]) -> pl.DataFrame:
    """Parse a schedule response into one row per played game.

    Postponed and cancelled listings are dropped, as are game types not in
    ``game_types``.

    Args:
        payload: JSON body from ``/schedule``.
        game_types: Game-type codes to keep.

    Returns:
        Games with typed dates and a UTC first-pitch timestamp.
    """
    body = orjson.loads(payload)
    rows = [
        {
            "game_pk": game["gamePk"],
            "game_date": game["officialDate"],
            "season": int(game["season"]),
            "game_type": game["gameType"],
            "detailed_state": game["status"]["detailedState"],
            "venue_id": game["venue"]["id"],
            "venue_name": game["venue"]["name"],
            "home_team_id": game["teams"]["home"]["team"]["id"],
            "home_team_name": game["teams"]["home"]["team"]["name"],
            "away_team_id": game["teams"]["away"]["team"]["id"],
            "away_team_name": game["teams"]["away"]["team"]["name"],
            "first_pitch_at": game["gameDate"],
        }
        for schedule_date in body.get("dates", [])
        for game in schedule_date["games"]
    ]
    return (
        pl.DataFrame(rows, schema=SCHEDULE_SCHEMA)
        .filter(
            pl.col("game_type").is_in(list(game_types)),
            ~pl.col("detailed_state").is_in(list(UNPLAYED_STATES)),
        )
        .with_columns(
            pl.col("game_date").str.to_date(),
            pl.col("first_pitch_at").str.to_datetime(
                format="%Y-%m-%dT%H:%M:%SZ", time_zone="UTC"
            ),
        )
    )


def parse_transactions(payload: bytes) -> pl.DataFrame:
    """Parse a transactions response, keeping transactions tied to a player.

    Args:
        payload: JSON body from ``/transactions``.

    Returns:
        One row per player transaction with typed dates.
    """
    body = orjson.loads(payload)
    rows = [
        {
            "transaction_id": transaction["id"],
            "player_id": transaction["person"]["id"],
            "transaction_date": transaction.get("date"),
            "effective_date": transaction.get("effectiveDate"),
            "type_code": transaction.get("typeCode"),
            "type_desc": transaction.get("typeDesc"),
            "description": transaction.get("description"),
        }
        for transaction in body.get("transactions", [])
        if "person" in transaction
    ]
    return pl.DataFrame(rows, schema=TRANSACTION_SCHEMA).with_columns(
        pl.col("transaction_date", "effective_date").str.to_date(strict=False)
    )


def parse_people(payload: bytes) -> pl.DataFrame:
    """Parse a people response into player identity rows.

    Args:
        payload: JSON body from ``/people``.

    Returns:
        One row per player with name, birth date and handedness.
    """
    body = orjson.loads(payload)
    rows = [
        {
            "player_id": person["id"],
            "full_name": person.get("fullName"),
            "birth_date": person.get("birthDate"),
            "throws": person.get("pitchHand", {}).get("code"),
            "bats": person.get("batSide", {}).get("code"),
        }
        for person in body.get("people", [])
    ]
    return pl.DataFrame(rows, schema=PEOPLE_SCHEMA).with_columns(
        pl.col("birth_date").str.to_date(strict=False)
    )


async def _fetch_parsed(
    client: httpx.AsyncClient,
    endpoint: str,
    params: dict[str, str],
    settings: IngestSettings,
    parse: JsonParser,
) -> pl.DataFrame:
    """GET one Stats API endpoint and parse the body.

    Args:
        client: Open HTTP client.
        endpoint: Path under the API base, such as ``schedule``.
        params: Query parameters.
        settings: Ingest settings (retries).
        parse: Parser for the response body.

    Returns:
        The parsed frame.
    """
    payload = await get_bytes(
        client, f"{STATSAPI_BASE_URL}/{endpoint}", params, settings.max_retries
    )
    return parse(payload)


async def _fetch_all(
    client: httpx.AsyncClient,
    endpoint: str,
    param_sets: Sequence[dict[str, str]],
    settings: IngestSettings,
    parse: JsonParser,
) -> list[pl.DataFrame]:
    """Fetch and parse several requests to one endpoint concurrently.

    Args:
        client: Open HTTP client.
        endpoint: Path under the API base.
        param_sets: Query parameters, one dict per request.
        settings: Ingest settings (concurrency and retries).
        parse: Parser for each response body.

    Returns:
        Parsed frames, in completion order.
    """
    tasks = [
        functools.partial(_fetch_parsed, client, endpoint, params, settings, parse)
        for params in param_sets
    ]
    return await run_bounded(tasks, settings.max_concurrency, f"Stats API {endpoint}")


async def fetch_schedule(
    client: httpx.AsyncClient, start: date, end: date, settings: IngestSettings
) -> pl.DataFrame:
    """Fetch played MLB games between two dates, one request per month.

    A game postponed and replayed keeps its ``game_pk``; only its latest
    listing is kept.

    Args:
        client: Open HTTP client.
        start: First date.
        end: Last date (inclusive).
        settings: Ingest settings.

    Returns:
        One row per played game.
    """
    param_sets = [
        {
            "sportId": MLB_SPORT_ID,
            "startDate": window_start.isoformat(),
            "endDate": window_end.isoformat(),
        }
        for window_start, window_end in month_windows(start, end)
    ]
    parse = functools.partial(parse_schedule, game_types=settings.game_types)
    frames = await _fetch_all(client, "schedule", param_sets, settings, parse)
    return (
        pl.concat(frames)
        .sort("game_date")
        .unique(subset="game_pk", keep="last", maintain_order=True)
    )


async def fetch_transactions(
    client: httpx.AsyncClient, start: date, end: date, settings: IngestSettings
) -> pl.DataFrame:
    """Fetch MLB player transactions between two dates, one request per month.

    Args:
        client: Open HTTP client.
        start: First date.
        end: Last date (inclusive).
        settings: Ingest settings.

    Returns:
        One row per player transaction.
    """
    param_sets = [
        {
            "sportId": MLB_SPORT_ID,
            "startDate": window_start.strftime("%m/%d/%Y"),
            "endDate": window_end.strftime("%m/%d/%Y"),
        }
        for window_start, window_end in month_windows(start, end)
    ]
    frames = await _fetch_all(
        client, "transactions", param_sets, settings, parse_transactions
    )
    return pl.concat(frames).unique(subset="transaction_id", maintain_order=True)


async def fetch_people(
    client: httpx.AsyncClient, player_ids: Iterable[int], settings: IngestSettings
) -> pl.DataFrame:
    """Fetch identity rows for players, in batches of ``PEOPLE_BATCH_SIZE``.

    Args:
        client: Open HTTP client.
        player_ids: MLBAM player ids.
        settings: Ingest settings.

    Returns:
        One row per player found.
    """
    unique_ids = sorted(set(player_ids))
    if not unique_ids:
        return parse_people(b"{}")
    param_sets = [
        {
            "personIds": ",".join(
                str(player_id)
                for player_id in unique_ids[offset : offset + PEOPLE_BATCH_SIZE]
            )
        }
        for offset in range(0, len(unique_ids), PEOPLE_BATCH_SIZE)
    ]
    frames = await _fetch_all(client, "people", param_sets, settings, parse_people)
    return pl.concat(frames)
