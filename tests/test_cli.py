from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from statecast import cli
from statecast.config import Settings
from statecast.ingest.pipeline import IngestSummary
from statecast.ingest.savant import StatcastIngestSummary

PROJECT_SETTINGS = Path(__file__).parent.parent / "config" / "settings.toml"
runner = CliRunner()


@pytest.fixture(autouse=True)
def keep_test_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop the CLI rebinding log handlers to the runner's short-lived stream."""
    monkeypatch.setattr(cli, "configure_logging", lambda level: None)


def summary_with(failed_dates: tuple[date, ...]) -> IngestSummary:
    return IngestSummary(
        games=1,
        statcast=StatcastIngestSummary(
            rows_written=10, days_written=1, failed_dates=failed_dates
        ),
        transactions=0,
        new_players=2,
    )


def ingest_args() -> list[str]:
    return [
        "ingest",
        "--start",
        "2026-06-01",
        "--end",
        "2026-06-02",
        "--settings",
        str(PROJECT_SETTINGS),
    ]


def test_ingest_passes_dates_and_exits_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[tuple[date, date]] = []

    async def fake_run_ingest(
        settings: Settings, start: date, end: date
    ) -> IngestSummary:
        received.append((start, end))
        return summary_with(())

    monkeypatch.setattr(cli, "run_ingest", fake_run_ingest)

    result = runner.invoke(cli.app, ingest_args())

    assert result.exit_code == 0
    assert received == [(date(2026, 6, 1), date(2026, 6, 2))]


def test_ingest_exits_nonzero_when_dates_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_ingest(
        settings: Settings, start: date, end: date
    ) -> IngestSummary:
        return summary_with((date(2026, 6, 2),))

    monkeypatch.setattr(cli, "run_ingest", fake_run_ingest)

    result = runner.invoke(cli.app, ingest_args())

    assert result.exit_code == 1


def test_ingest_exits_nonzero_on_invalid_range(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_ingest(
        settings: Settings, start: date, end: date
    ) -> IngestSummary:
        raise ValueError("start is after end")

    monkeypatch.setattr(cli, "run_ingest", fake_run_ingest)

    result = runner.invoke(cli.app, ingest_args())

    assert result.exit_code == 1


def test_load_passes_requested_seasons(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[list[int] | None] = []

    def fake_build_warehouse(
        warehouse_path: Path, raw_dir: Path, seasons: list[int] | None
    ) -> dict[str, int]:
        received.append(seasons)
        return {"pitch": 5}

    monkeypatch.setattr(cli, "build_warehouse", fake_build_warehouse)

    result = runner.invoke(
        cli.app,
        [
            "load",
            "--season",
            "2025",
            "--season",
            "2026",
            "--settings",
            str(PROJECT_SETTINGS),
        ],
    )

    assert result.exit_code == 0
    assert received == [[2025, 2026]]


def test_missing_settings_file_exits_nonzero(output_dir: Path) -> None:
    result = runner.invoke(
        cli.app, ["load", "--settings", str(output_dir / "missing.toml")]
    )

    assert result.exit_code == 1
