from datetime import date
from pathlib import Path

import duckdb
import jax.numpy as jnp
import polars as pl
import pytest

from statecast.models.estimation import FitResult
from statecast.models.jumps import (
    JumpFitResult,
    JumpModel,
    JumpParams,
    unconstrain_jump_model,
)
from statecast.models.kalman import PopulationParams, unconstrain
from statecast.models.rbpf import FilterSettings
from statecast.storage import jump_runs
from statecast.storage.jump_runs import (
    JUMP_MODEL_NAME,
    JumpRunDetails,
    latest_jump_run,
    latest_jump_run_for_player,
    load_jump_states,
    load_jump_trajectory,
    load_player_names,
    save_jump_run,
)
from statecast.storage.model_runs import MODEL_NAME, RunSpec, latest_run, save_run
from statecast.storage.warehouse import initialize_schema, upsert_table

POPULATION = PopulationParams(
    mean_reversion=jnp.asarray(0.01),
    diffusion=jnp.asarray(0.1),
    observation_sd=jnp.asarray(0.9),
    population_mean=jnp.asarray(94.5),
    population_sd=jnp.asarray(2.0),
    day_effect_sd=jnp.asarray(0.5),
    seasonal_knots=jnp.array([-0.3, -0.2, -0.1, 0.1, 0.05, 0.0]),
)
MODEL = JumpModel(
    POPULATION,
    JumpParams(rate=jnp.asarray(0.005), mean=jnp.asarray(-0.8), sd=jnp.asarray(1.4)),
)
JUMP_FIT = JumpFitResult(
    model=MODEL,
    unconstrained=unconstrain_jump_model(MODEL),
    unconstrained_cov=jnp.eye(15) * 1e-4,
    log_likelihood=jnp.asarray(-1200.0),
    iterations=jnp.asarray(80),
    gradient_norm=jnp.asarray(1e-9),
)
OU_FIT = FitResult(
    params=POPULATION,
    unconstrained=unconstrain(POPULATION),
    unconstrained_cov=jnp.eye(12) * 1e-4,
    log_likelihood=jnp.asarray(-1234.5),
    iterations=jnp.asarray(30),
    gradient_norm=jnp.asarray(1e-9),
)
DETAILS = JumpRunDetails(
    ou_log_likelihood=-1234.5,
    particle_log_likelihood=-1201.0,
    settings=FilterSettings(256, 3),
    seed=7,
)
SPEC = RunSpec("pitcher", "release_speed", "FF", (2025, 2026))


def jump_states(player_ids: list[int]) -> pl.DataFrame:
    count = len(player_ids)
    return pl.DataFrame(
        {
            "player_id": player_ids,
            "game_date": [date(2026, 4, day + 1) for day in range(count)],
            "sample_size": [25] * count,
            "appearance_mean": [95.0] * count,
            "seasonal_effect": [-0.2] * count,
            "talent_mean": [94.9] * count,
            "talent_sd": [0.4] * count,
            "prior_jump_prob": [0.02] * count,
            "jump_prob": [0.1] * count,
            "jump_evidence": [1.7] * count,
            "recent_jump_prob": [0.3] * count,
            "recent_drop_prob": [0.2] * count,
            "effective_size": [200.0] * count,
        },
        schema_overrides={"player_id": pl.Int32, "sample_size": pl.Int32},
    )


def ou_paths(player_ids: list[int]) -> pl.DataFrame:
    count = len(player_ids)
    return pl.DataFrame(
        {
            "player_id": player_ids,
            "game_date": [date(2026, 4, day + 1) for day in range(count)],
            "sample_size": [25] * count,
            "appearance_mean": [95.0] * count,
            "seasonal_effect": [-0.2] * count,
            "filtered_mean": [94.9] * count,
            "filtered_sd": [0.4] * count,
            "smoothed_mean": [94.8] * count,
            "smoothed_sd": [0.3] * count,
        },
        schema_overrides={"player_id": pl.Int32, "sample_size": pl.Int32},
    )


@pytest.fixture
def connection(output_dir: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(output_dir / "jumps.duckdb"))
    initialize_schema(con)
    upsert_table(
        con,
        "player",
        pl.DataFrame(
            {
                "player_id": [554430, 650911],
                "full_name": ["Zack Wheeler", "Ranger Suárez"],
            }
        ),
    )
    upsert_table(
        con, "pitch_type", pl.DataFrame({"code": ["FF"], "name": ["4-Seam Fastball"]})
    )
    return con


def test_save_jump_run_stores_every_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    run_id = save_jump_run(
        connection, SPEC, JUMP_FIT, jump_states([554430, 554430, 650911]), DETAILS
    )

    run = connection.execute(
        "SELECT model_name, player_count, appearance_count, log_likelihood "
        "FROM model_run"
    ).fetchone()
    parameters = connection.execute(
        "SELECT parameter FROM model_parameter WHERE run_id = $id", {"id": str(run_id)}
    ).fetchall()
    extra = connection.execute(
        "SELECT ou_log_likelihood, particle_log_likelihood, particle_count, "
        "recent_window, seed FROM jump_run"
    ).fetchone()
    assert run == (JUMP_MODEL_NAME, 2, 3, -1200.0)
    assert len(parameters) == 18
    assert ("days_between_jumps",) in parameters
    assert extra == (-1234.5, -1201.0, 256, 3, 7)
    stored = load_jump_trajectory(connection, run_id, 554430)
    assert stored.height == 2
    assert stored["recent_drop_prob"].to_list() == [0.2, 0.2]
    assert stored["jump_evidence"].to_list() == [1.7, 1.7]
    assert stored["prior_jump_prob"].to_list() == [0.02, 0.02]


def test_save_jump_run_rolls_back_when_a_write_fails(
    connection: duckdb.DuckDBPyConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = upsert_table

    def failing(con: duckdb.DuckDBPyConnection, table: str, frame: pl.DataFrame) -> int:
        if table == "jump_state":
            raise duckdb.ConstraintException("simulated failure")
        return original(con, table, frame)

    monkeypatch.setattr(jump_runs, "upsert_table", failing)

    with pytest.raises(duckdb.ConstraintException):
        save_jump_run(connection, SPEC, JUMP_FIT, jump_states([554430]), DETAILS)

    assert connection.execute("SELECT count(*) FROM model_run").fetchone() == (0,)
    assert connection.execute("SELECT count(*) FROM jump_run").fetchone() == (0,)


def test_latest_jump_run_for_player_skips_ou_runs(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    jump_run = save_jump_run(connection, SPEC, JUMP_FIT, jump_states([554430]), DETAILS)
    save_run(connection, SPEC, OU_FIT, ou_paths([554430]))

    found = latest_jump_run_for_player(connection, 554430, "pitcher", None, "FF")

    assert found.run_id == jump_run
    assert found.pitch_name == "4-Seam Fastball"
    with pytest.raises(LookupError, match="No pitcher jump run"):
        latest_jump_run_for_player(connection, 650911, "pitcher")


def test_latest_ou_run_ignores_newer_jump_runs(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    ou_run = save_run(connection, SPEC, OU_FIT, ou_paths([554430]))
    save_jump_run(connection, SPEC, JUMP_FIT, jump_states([554430]), DETAILS)

    found = latest_run(connection, SPEC)

    assert found.run_id == ou_run
    model_names = connection.execute(
        "SELECT model_name FROM model_run ORDER BY created_at"
    ).fetchall()
    assert model_names == [(MODEL_NAME,), (JUMP_MODEL_NAME,)]


def test_load_player_names_returns_int64_ids(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    names = load_player_names(connection)

    assert names.schema["player_id"] == pl.Int64
    assert sorted(names["full_name"].to_list()) == ["Ranger Suárez", "Zack Wheeler"]


def test_latest_jump_run_reads_window_and_jump_size(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    save_jump_run(connection, SPEC, JUMP_FIT, jump_states([554430]), DETAILS)
    newest = save_jump_run(connection, SPEC, JUMP_FIT, jump_states([650911]), DETAILS)

    summary = latest_jump_run(connection, SPEC)

    assert summary.run.run_id == newest
    assert summary.run.pitch_name == "4-Seam Fastball"
    assert summary.recent_window == 3
    assert summary.jump_mean == pytest.approx(-0.8)
    assert summary.jump_sd == pytest.approx(1.4)
    with pytest.raises(LookupError, match="No pitcher jump run"):
        latest_jump_run(connection, RunSpec("pitcher", "spin_rate", "FF", ()))


def test_load_jump_states_skips_runs_without_evidence(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    scored = save_jump_run(
        connection, SPEC, JUMP_FIT, jump_states([554430, 650911]), DETAILS
    )
    unscored = save_jump_run(
        connection,
        SPEC,
        JUMP_FIT,
        jump_states([554430]).with_columns(jump_evidence=pl.lit(None, pl.Float64)),
        DETAILS,
    )

    states = load_jump_states(connection, scored)

    assert states["player_id"].to_list() == [554430, 650911]
    assert states["recent_drop_prob"].to_list() == [0.2, 0.2]
    assert load_jump_states(connection, unscored).is_empty()
