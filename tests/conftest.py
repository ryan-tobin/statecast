"""Shared fixtures: settings, persistent test-output folders and sample data."""

import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import orjson
import polars as pl
import pytest

from statecast.config import IngestSettings, PathSettings, Settings
from statecast.ingest.savant import STATCAST_SCHEMA

TEST_OUTPUT_ROOT = Path(__file__).parent / "outputs"
RUN_ID = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S%f")

type StatcastFactory = Callable[[list[dict[str, object]]], pl.DataFrame]


@pytest.fixture
def output_dir(request: pytest.FixtureRequest) -> Path:
    """A per-test folder under tests/outputs that is kept after the run."""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", request.node.name)
    directory = TEST_OUTPUT_ROOT / RUN_ID / safe_name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture
def ingest_settings() -> IngestSettings:
    return IngestSettings(
        game_types=("R", "F", "D", "L", "W"),
        max_concurrency=2,
        timeout_seconds=5.0,
        max_retries=2,
        user_agent="statecast-tests",
    )


@pytest.fixture
def settings(output_dir: Path, ingest_settings: IngestSettings) -> Settings:
    return Settings(
        paths=PathSettings(
            raw_dir=output_dir / "raw",
            warehouse_path=output_dir / "warehouse" / "statecast.duckdb",
        ),
        ingest=ingest_settings,
    )


@pytest.fixture
def make_statcast() -> StatcastFactory:
    """Build typed Statcast frames from partial row dicts."""

    def build(rows: list[dict[str, object]]) -> pl.DataFrame:
        return pl.DataFrame(rows, schema=STATCAST_SCHEMA)

    return build


def schedule_game(
    game_pk: int,
    official_date: str,
    game_type: str = "R",
    detailed_state: str = "Final",
) -> dict[str, object]:
    """One game as the Stats API schedule endpoint returns it."""
    return {
        "gamePk": game_pk,
        "officialDate": official_date,
        "season": official_date[:4],
        "gameType": game_type,
        "gameDate": f"{official_date}T23:05:00Z",
        "status": {"detailedState": detailed_state},
        "venue": {"id": 2681, "name": "Citizens Bank Park"},
        "teams": {
            "home": {"team": {"id": 143, "name": "Philadelphia Phillies"}},
            "away": {"team": {"id": 121, "name": "New York Mets"}},
        },
    }


def schedule_payload(games: list[dict[str, object]]) -> bytes:
    """Wrap games in a schedule response body."""
    return orjson.dumps({"dates": [{"games": games}] if games else []})
