from datetime import date
from pathlib import Path

import duckdb
import jax.numpy as jnp
import polars as pl
import pytest

from statecast.models.estimation import FitResult
from statecast.models.kalman import PopulationParams, unconstrain
from statecast.storage import model_runs
from statecast.storage.model_runs import (
    PlayerLookupError,
    RunSpec,
    find_player,
    latest_run_for_player,
    load_trajectory,
    pitch_type_name,
    save_run,
)
from statecast.storage.warehouse import initialize_schema, upsert_table

PARAMS = PopulationParams(
    mean_reversion=jnp.asarray(0.01),
    diffusion=jnp.asarray(0.1),
    observation_sd=jnp.asarray(0.9),
    population_mean=jnp.asarray(94.5),
    population_sd=jnp.asarray(2.0),
    day_effect_sd=jnp.asarray(0.5),
    seasonal_knots=jnp.array([-0.3, -0.2, -0.1, 0.1, 0.05, 0.0]),
)
FIT = FitResult(
    params=PARAMS,
    unconstrained=unconstrain(PARAMS),
    unconstrained_cov=jnp.eye(12) * 1e-4,
    log_likelihood=jnp.asarray(-1234.5),
    iterations=jnp.asarray(30),
    gradient_norm=jnp.asarray(1e-9),
)
PITCHER_SPEC = RunSpec("pitcher", "release_speed", "FF", (2025, 2026))


def paths(player_ids: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": player_ids,
            "game_date": [date(2026, 4, day + 1) for day in range(len(player_ids))],
            "sample_size": [25] * len(player_ids),
            "appearance_mean": [95.0] * len(player_ids),
            "seasonal_effect": [-0.2] * len(player_ids),
            "filtered_mean": [94.9] * len(player_ids),
            "filtered_sd": [0.4] * len(player_ids),
            "smoothed_mean": [94.8] * len(player_ids),
            "smoothed_sd": [0.3] * len(player_ids),
        },
        schema_overrides={"player_id": pl.Int32, "sample_size": pl.Int32},
    )


@pytest.fixture
def connection(output_dir: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(output_dir / "runs.duckdb"))
    initialize_schema(con)
    upsert_table(
        con,
        "player",
        pl.DataFrame(
            {
                "player_id": [554430, 650911, 1, 2],
                "full_name": [
                    "Zack Wheeler",
                    "Cristopher Sánchez",
                    "Will Smith",
                    "Will Smith",
                ],
            }
        ),
    )
    upsert_table(
        con, "pitch_type", pl.DataFrame({"code": ["FF"], "name": ["4-Seam Fastball"]})
    )
    return con


def test_save_run_stores_run_parameters_and_paths(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    run_id = save_run(connection, PITCHER_SPEC, FIT, paths([554430, 554430, 650911]))

    run = connection.execute(
        "SELECT role, seasons, player_count, appearance_count FROM model_run"
    ).fetchone()
    parameter_count = connection.execute(
        "SELECT count(*) FROM model_parameter WHERE run_id = $id", {"id": str(run_id)}
    ).fetchone()
    assert run == ("pitcher", [2025, 2026], 2, 3)
    assert parameter_count == (14,)
    stored = load_trajectory(connection, run_id, 554430)
    assert stored.height == 2
    assert stored["seasonal_effect"].to_list() == [-0.2, -0.2]


def test_save_run_rolls_back_when_a_write_fails(
    connection: duckdb.DuckDBPyConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = upsert_table

    def failing(con: duckdb.DuckDBPyConnection, table: str, frame: pl.DataFrame) -> int:
        if table == "latent_state":
            raise duckdb.ConstraintException("simulated failure")
        return original(con, table, frame)

    monkeypatch.setattr(model_runs, "upsert_table", failing)

    with pytest.raises(duckdb.ConstraintException):
        save_run(connection, PITCHER_SPEC, FIT, paths([554430]))

    assert connection.execute("SELECT count(*) FROM model_run").fetchone() == (0,)


def test_find_player_ignores_case_and_accents(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    assert find_player(connection, "cristopher sanchez") == (
        650911,
        "Cristopher Sánchez",
    )


def test_find_player_rejects_unknown_and_ambiguous_names(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    with pytest.raises(PlayerLookupError, match="No player"):
        find_player(connection, "Nobody Here")
    with pytest.raises(PlayerLookupError, match="several players"):
        find_player(connection, "Will Smith")


def test_latest_run_for_player_picks_newest_matching_run(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    save_run(connection, PITCHER_SPEC, FIT, paths([554430]))
    spin = RunSpec("pitcher", "spin_rate", "FF", (2026,))
    newest = save_run(connection, spin, FIT, paths([554430]))

    latest = latest_run_for_player(connection, 554430, "pitcher")
    velocity = latest_run_for_player(connection, 554430, "pitcher", "release_speed")

    assert latest.run_id == newest
    assert latest.pitch_name == "4-Seam Fastball"
    assert velocity.metric == "release_speed"


def test_latest_run_for_player_filters_by_pitch_type(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    fastball = save_run(connection, PITCHER_SPEC, FIT, paths([554430]))
    sinker = RunSpec("pitcher", "release_speed", "SI", (2025, 2026))
    newest = save_run(connection, sinker, FIT, paths([554430]))

    any_pitch = latest_run_for_player(connection, 554430, "pitcher")
    four_seam = latest_run_for_player(connection, 554430, "pitcher", None, "FF")

    assert any_pitch.run_id == newest
    assert four_seam.run_id == fastball
    assert four_seam.pitch_type == "FF"


def test_latest_run_for_player_requires_a_run(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    with pytest.raises(LookupError, match="No hitter run"):
        latest_run_for_player(connection, 554430, "hitter")


def test_pitch_type_name_looks_up_codes(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    assert pitch_type_name(connection, "FF") == "4-Seam Fastball"
    assert pitch_type_name(connection, "ZZ") is None
    assert pitch_type_name(connection, None) is None
