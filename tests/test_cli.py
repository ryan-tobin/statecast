from datetime import date
from pathlib import Path

import duckdb
import jax
import jax.numpy as jnp
import polars as pl
import pytest
from typer.testing import CliRunner

from statecast import cli
from statecast.config import Settings
from statecast.ingest.pipeline import IngestSummary
from statecast.ingest.savant import StatcastIngestSummary
from statecast.models.estimation import FitResult
from statecast.models.kalman import ObservationBatch, PopulationParams, unconstrain
from statecast.storage.warehouse import initialize_schema, upsert_table

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


def model_params(seasonal_knots: jax.Array) -> PopulationParams:
    return PopulationParams(
        mean_reversion=jnp.asarray(0.01),
        diffusion=jnp.asarray(0.1),
        observation_sd=jnp.asarray(0.9),
        population_mean=jnp.asarray(94.5),
        population_sd=jnp.asarray(2.0),
        day_effect_sd=jnp.asarray(0.5),
        seasonal_knots=seasonal_knots,
    )


def fit_settings(output_dir: Path) -> Path:
    settings_path = output_dir / "config" / "settings.toml"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        PROJECT_SETTINGS.read_text(encoding="utf-8").replace(
            "data/warehouse/statecast.duckdb", "warehouse.duckdb"
        ),
        encoding="utf-8",
    )
    duckdb.connect(str(output_dir / "warehouse.duckdb")).close()
    return settings_path


def test_fit_prints_parameter_table(
    output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    params = model_params(jnp.zeros(6))
    frame = pl.DataFrame(
        {
            "player_id": [1, 1],
            "game_date": [date(2026, 4, 1), date(2026, 4, 6)],
            "sample_size": [90, 88],
            "mean": [94.0, 94.2],
            "within_ss": [80.0, 85.0],
        }
    )

    def fake_fit(batch: ObservationBatch, initial: PopulationParams) -> FitResult:
        return FitResult(
            params=params,
            unconstrained=unconstrain(params),
            unconstrained_cov=jnp.eye(12) * 1e-4,
            log_likelihood=jnp.asarray(-100.0),
            iterations=jnp.asarray(12),
            gradient_norm=jnp.asarray(1e-9),
        )

    monkeypatch.setattr(cli, "pitcher_appearances", lambda *args: frame)
    monkeypatch.setattr(cli, "fit_population", fake_fit)

    result = runner.invoke(
        cli.app,
        [
            "fit",
            "pitcher",
            "--season",
            "2026",
            "--settings",
            str(fit_settings(output_dir)),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "half_life_days" in result.output
    assert "day_effect_sd" in result.output
    assert "season_apr" in result.output


def test_fit_exits_nonzero_without_data(
    output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "hitter_appearances", lambda *args: pl.DataFrame())

    result = runner.invoke(
        cli.app,
        [
            "fit",
            "hitter",
            "--season",
            "2026",
            "--settings",
            str(fit_settings(output_dir)),
        ],
    )

    assert result.exit_code == 1


def test_trajectory_then_chart_end_to_end(
    output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    params = model_params(jnp.array([-0.2, -0.1, -0.03, 0.04, 0.03, 0.0]))
    frame = pl.DataFrame(
        {
            "player_id": [554430] * 3,
            "game_date": [date(2026, 4, 1), date(2026, 4, 6), date(2026, 4, 11)],
            "sample_size": [30, 28, 31],
            "mean": [96.1, 95.8, 96.3],
            "within_ss": [25.0, 22.0, 26.0],
        }
    )

    def fake_fit(batch: ObservationBatch, initial: PopulationParams) -> FitResult:
        return FitResult(
            params=params,
            unconstrained=unconstrain(params),
            unconstrained_cov=jnp.eye(12) * 1e-4,
            log_likelihood=jnp.asarray(-100.0),
            iterations=jnp.asarray(12),
            gradient_norm=jnp.asarray(1e-9),
        )

    monkeypatch.setattr(cli, "pitcher_appearances", lambda *args: frame)
    monkeypatch.setattr(cli, "fit_population", fake_fit)
    settings_path = fit_settings(output_dir)
    with duckdb.connect(str(output_dir / "warehouse.duckdb")) as connection:
        initialize_schema(connection)
        upsert_table(
            connection,
            "player",
            pl.DataFrame({"player_id": [554430], "full_name": ["Zack Wheeler"]}),
        )

    stored = runner.invoke(
        cli.app,
        ["trajectory", "pitcher", "--season", "2026", "--settings", str(settings_path)],
    )
    drawn = runner.invoke(
        cli.app, ["chart", "zack wheeler", "--settings", str(settings_path)]
    )

    assert stored.exit_code == 0, stored.output
    assert drawn.exit_code == 0, drawn.output
    figures = output_dir / "outputs" / "figures"
    assert (figures / "zack_wheeler_ff_release_speed.png").exists()
    assert (figures / "league_seasonal_ff_release_speed.png").exists()
