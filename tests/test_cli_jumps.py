from datetime import date, timedelta
from pathlib import Path

import duckdb
import jax.numpy as jnp
import polars as pl
import pytest
from typer.testing import CliRunner

from statecast import cli
from statecast.models.estimation import FitResult
from statecast.models.jumps import (
    JumpFitResult,
    JumpModel,
    JumpParams,
    unconstrain_jump_model,
)
from statecast.models.kalman import ObservationBatch, PopulationParams, unconstrain
from statecast.models.rbpf import FilterSettings
from statecast.storage.warehouse import initialize_schema, upsert_table

PROJECT_SETTINGS = Path(__file__).parent.parent / "config" / "settings.toml"
POPULATION = PopulationParams(
    mean_reversion=jnp.asarray(0.01),
    diffusion=jnp.asarray(0.1),
    observation_sd=jnp.asarray(0.9),
    population_mean=jnp.asarray(94.5),
    population_sd=jnp.asarray(2.0),
    day_effect_sd=jnp.asarray(0.5),
    seasonal_knots=jnp.array([-0.2, -0.1, -0.03, 0.04, 0.03, 0.0]),
)
MODEL = JumpModel(
    POPULATION,
    JumpParams(rate=jnp.asarray(0.01), mean=jnp.asarray(-1.0), sd=jnp.asarray(1.5)),
)
runner = CliRunner()


@pytest.fixture(autouse=True)
def keep_test_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop the CLI rebinding log handlers to the runner's short-lived stream."""
    monkeypatch.setattr(cli, "configure_logging", lambda level: None)


def warehouse_settings(output_dir: Path) -> Path:
    settings_path = output_dir / "config" / "settings.toml"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        PROJECT_SETTINGS.read_text(encoding="utf-8").replace(
            "data/warehouse/statecast.duckdb", "warehouse.duckdb"
        ),
        encoding="utf-8",
    )
    with duckdb.connect(str(output_dir / "warehouse.duckdb")) as connection:
        initialize_schema(connection)
        upsert_table(
            connection,
            "player",
            pl.DataFrame(
                {
                    "player_id": [554430, 605400],
                    "full_name": ["Zack Wheeler", "Aaron Nola"],
                }
            ),
        )
    return settings_path


def appearances() -> pl.DataFrame:
    dates = [date(2026, 4, 1) + timedelta(days=5 * start) for start in range(8)]
    wheeler = [96.0, 96.2, 95.9, 96.1, 93.4, 93.6, 93.2, 93.5]
    nola = [92.5, 92.7, 92.4, 92.6, 92.5, 92.8, 92.6, 92.4]
    return pl.DataFrame(
        {
            "player_id": [554430] * 8 + [605400] * 8,
            "game_date": dates * 2,
            "sample_size": [90] * 16,
            "mean": wheeler + nola,
            "within_ss": [70.0] * 16,
        }
    )


def fake_ou_fit(batch: ObservationBatch, initial: PopulationParams) -> FitResult:
    return FitResult(
        params=POPULATION,
        unconstrained=unconstrain(POPULATION),
        unconstrained_cov=jnp.eye(12) * 1e-4,
        log_likelihood=jnp.asarray(-150.0),
        iterations=jnp.asarray(12),
        gradient_norm=jnp.asarray(1e-9),
    )


def fake_jump_fit(batch: ObservationBatch, initial: JumpModel) -> JumpFitResult:
    return JumpFitResult(
        model=MODEL,
        unconstrained=unconstrain_jump_model(MODEL),
        unconstrained_cov=jnp.eye(15) * 1e-4,
        log_likelihood=jnp.asarray(-120.0),
        iterations=jnp.asarray(40),
        gradient_norm=jnp.asarray(1e-9),
    )


def test_jumps_then_jump_chart_end_to_end(
    output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "pitcher_appearances", lambda *args: appearances())
    monkeypatch.setattr(cli, "fit_population", fake_ou_fit)
    monkeypatch.setattr(cli, "fit_jump_population", fake_jump_fit)
    monkeypatch.setattr(cli, "JUMP_FILTER_SETTINGS", FilterSettings(64, 3))
    settings_path = warehouse_settings(output_dir)

    fitted = runner.invoke(
        cli.app,
        ["jumps", "pitcher", "--season", "2026", "--settings", str(settings_path)],
    )
    drawn = runner.invoke(
        cli.app,
        ["jump-chart", "zack wheeler", "--settings", str(settings_path)],
    )

    assert fitted.exit_code == 0, fitted.output
    assert "days_between_jumps" in fitted.output
    assert "OU + jumps" in fitted.output
    assert "strongest evidence of a jump" in fitted.output
    assert "Likeliest recent drops" in fitted.output
    assert drawn.exit_code == 0, drawn.output
    figures = output_dir / "outputs" / "figures"
    assert (figures / "zack_wheeler_jumps_ff_release_speed.png").exists()
    with duckdb.connect(str(output_dir / "warehouse.duckdb")) as connection:
        stored = connection.execute(
            "SELECT count(*), max(jump_prob) FROM jump_state"
        ).fetchone()
        particles = connection.execute("SELECT particle_count FROM jump_run").fetchone()
    assert stored is not None
    assert stored[0] == 16
    assert 0.0 < stored[1] <= 1.0
    assert particles == (64,)


def test_jump_chart_exits_nonzero_without_a_jump_run(output_dir: Path) -> None:
    settings_path = warehouse_settings(output_dir)

    result = runner.invoke(
        cli.app, ["jump-chart", "Zack Wheeler", "--settings", str(settings_path)]
    )

    assert result.exit_code == 1


def test_jumps_exits_nonzero_without_data(
    output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "pitcher_appearances", lambda *args: pl.DataFrame())
    settings_path = warehouse_settings(output_dir)

    result = runner.invoke(
        cli.app,
        ["jumps", "pitcher", "--season", "2026", "--settings", str(settings_path)],
    )

    assert result.exit_code == 1