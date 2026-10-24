"""Save fitted model runs and read player trajectories back from the warehouse."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import duckdb
import polars as pl
from jax import Array

from statecast.models.estimation import FitResult, summarize_fit
from statecast.storage.warehouse import initialize_schema, upsert_table

MODEL_NAME = "ou_day_effect_seasonal"

LATEST_RUN_SQL = """
SELECT r.run_id, r.role, r.metric, r.pitch_type, pt.name
FROM model_run AS r
LEFT JOIN pitch_type AS pt ON pt.code = r.pitch_type
WHERE r.role = $role
  AND ($metric IS NULL OR r.metric = $metric)
  AND ($pitch_type IS NULL OR r.pitch_type = $pitch_type)
  AND EXISTS (
      SELECT 1 FROM latent_state AS s
      WHERE s.run_id = r.run_id AND s.player_id = $player_id
  )
ORDER BY r.created_at DESC
LIMIT 1
"""

FIND_PLAYER_SQL = """
SELECT player_id, full_name
FROM player
WHERE strip_accents(lower(full_name)) = strip_accents(lower($name))
ORDER BY player_id
"""

PITCH_NAME_SQL = "SELECT name FROM pitch_type WHERE code = $code"

LATEST_POPULATION_RUN_SQL = """
SELECT r.run_id, r.role, r.metric, r.pitch_type, pt.name
FROM model_run AS r
LEFT JOIN pitch_type AS pt ON pt.code = r.pitch_type
WHERE r.model_name = $model_name
  AND r.role = $role
  AND r.metric = $metric
  AND r.pitch_type IS NOT DISTINCT FROM $pitch_type
ORDER BY r.created_at DESC
LIMIT 1
"""

ALERT_STATES_SQL = """
SELECT s.player_id, p.full_name, s.game_date, s.filtered_mean,
       s.filtered_gap_mean, s.alert_z
FROM latent_state AS s
JOIN player AS p USING (player_id)
WHERE s.run_id = $run_id
  AND s.alert_z IS NOT NULL
ORDER BY s.player_id, s.game_date
"""

TRAJECTORY_SQL = """
SELECT game_date, sample_size, appearance_mean, seasonal_effect,
       filtered_mean, filtered_sd, smoothed_mean, smoothed_sd
FROM latent_state
WHERE run_id = $run_id AND player_id = $player_id
ORDER BY game_date
"""


@dataclass(frozen=True, slots=True)
class RunSpec:
    """What a model run was fitted to.

    Attributes:
        role: ``pitcher`` or ``hitter``.
        metric: Metric column, such as ``release_speed``.
        pitch_type: Pitch type for pitcher runs, ``None`` for hitters.
        seasons: Seasons included.
    """

    role: str
    metric: str
    pitch_type: str | None
    seasons: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RunInfo:
    """A stored run, as found by ``latest_run_for_player``.

    Attributes:
        run_id: Run identifier.
        role: ``pitcher`` or ``hitter``.
        metric: Metric column.
        pitch_type: Pitch-type code, or ``None`` for hitter runs.
        pitch_name: Pitch-type name, such as ``4-Seam Fastball``, or ``None``.
    """

    run_id: uuid.UUID
    role: str
    metric: str
    pitch_type: str | None
    pitch_name: str | None


class PlayerLookupError(LookupError):
    """Raised when a player name matches no player or several players."""


class FitDiagnostics(Protocol):
    """What every fit result reports about its optimization."""

    @property
    def log_likelihood(self) -> Array:
        """Maximized total log-likelihood."""
        ...

    @property
    def iterations(self) -> Array:
        """Optimizer iterations used."""
        ...


def model_run_row(
    run_id: uuid.UUID,
    spec: RunSpec,
    model_name: str,
    states: pl.DataFrame,
    fit: FitDiagnostics,
) -> pl.DataFrame:
    """Build the ``model_run`` row describing a fitted run.

    Args:
        run_id: New run identifier.
        spec: What the run was fitted to.
        model_name: Model version, such as ``ou_day_effect_seasonal``.
        states: Per-appearance output; one row per player-game.
        fit: The fit's log-likelihood and iteration count.

    Returns:
        A one-row frame matching the ``model_run`` table.
    """
    return pl.DataFrame(
        {
            "run_id": [str(run_id)],
            "created_at": [datetime.now(tz=UTC)],
            "model_name": [model_name],
            "role": [spec.role],
            "metric": [spec.metric],
            "pitch_type": [spec.pitch_type],
            "seasons": [list(spec.seasons)],
            "player_count": [states["player_id"].n_unique()],
            "appearance_count": [states.height],
            "log_likelihood": [float(fit.log_likelihood)],
            "iterations": [int(fit.iterations)],
        },
        schema_overrides={"seasons": pl.List(pl.Int16), "pitch_type": pl.String},
    )


def save_run(
    connection: duckdb.DuckDBPyConnection,
    spec: RunSpec,
    fit: FitResult,
    trajectories: pl.DataFrame,
) -> uuid.UUID:
    """Store a fitted run, its parameter table and every player trajectory.

    Everything is written in one transaction, so a failed save leaves nothing
    behind.

    Args:
        connection: Writable warehouse connection.
        spec: What the run was fitted to.
        fit: Output of ``fit_population``.
        trajectories: Output of ``trajectory_frame``.

    Returns:
        The new run's identifier.

    Raises:
        duckdb.Error: If any write fails (the transaction is rolled back).
    """
    run_id = uuid.uuid4()
    run = model_run_row(run_id, spec, MODEL_NAME, trajectories, fit)
    parameters = summarize_fit(fit).with_columns(run_id=pl.lit(str(run_id)))
    states = trajectories.with_columns(run_id=pl.lit(str(run_id)))
    initialize_schema(connection)
    connection.begin()
    try:
        upsert_table(connection, "model_run", run)
        upsert_table(connection, "model_parameter", parameters)
        upsert_table(connection, "latent_state", states)
    except duckdb.Error:
        connection.rollback()
        raise
    connection.commit()
    return run_id


def find_player(connection: duckdb.DuckDBPyConnection, name: str) -> tuple[int, str]:
    """Look up a player by full name, ignoring case and accents.

    Args:
        connection: Warehouse connection.
        name: Full name, such as ``Cristopher Sanchez``.

    Returns:
        The player's MLBAM id and stored full name.

    Raises:
        PlayerLookupError: If the name matches no player or several players.
    """
    rows: Sequence[tuple[int, str]] = connection.execute(
        FIND_PLAYER_SQL, {"name": name}
    ).fetchall()
    if not rows:
        raise PlayerLookupError(f"No player named {name!r}")
    if len(rows) > 1:
        ids = ", ".join(str(player_id) for player_id, _ in rows)
        raise PlayerLookupError(f"{name!r} matches several players: {ids}")
    return rows[0]


def latest_run_for_player(
    connection: duckdb.DuckDBPyConnection,
    player_id: int,
    role: str,
    metric: str | None = None,
    pitch_type: str | None = None,
) -> RunInfo:
    """Find the most recent run of a role that includes a player.

    Args:
        connection: Warehouse connection.
        player_id: MLBAM player id.
        role: ``pitcher`` or ``hitter``.
        metric: Restrict to this metric, or ``None`` for any.
        pitch_type: Restrict to this pitch type, or ``None`` for any.

    Returns:
        The run's identifier and what it was fitted to.

    Raises:
        LookupError: If no stored run includes the player.
    """
    parameters = {
        "role": role,
        "metric": metric,
        "pitch_type": pitch_type,
        "player_id": player_id,
    }
    row = connection.execute(LATEST_RUN_SQL, parameters).fetchone()
    if row is None:
        raise LookupError(f"No {role} run includes player {player_id}")
    run_id, run_role, run_metric, pitch_type, pitch_name = row
    return RunInfo(run_id, run_role, run_metric, pitch_type, pitch_name)


def load_trajectory(
    connection: duckdb.DuckDBPyConnection, run_id: uuid.UUID, player_id: int
) -> pl.DataFrame:
    """Read one player's trajectory from a stored run.

    Args:
        connection: Warehouse connection.
        run_id: Run identifier.
        player_id: MLBAM player id.

    Returns:
        One row per appearance, ordered by date.
    """
    return connection.execute(
        TRAJECTORY_SQL, {"run_id": str(run_id), "player_id": player_id}
    ).pl()


def pitch_type_name(
    connection: duckdb.DuckDBPyConnection, code: str | None
) -> str | None:
    """Look up a pitch type's display name.

    Args:
        connection: Warehouse connection.
        code: Pitch-type code such as ``FF``, or ``None``.

    Returns:
        The name, such as ``4-Seam Fastball``, or ``None`` if unknown.
    """
    if code is None:
        return None
    row = connection.execute(PITCH_NAME_SQL, {"code": code}).fetchone()
    return None if row is None else str(row[0])


def latest_run(connection: duckdb.DuckDBPyConnection, spec: RunSpec) -> RunInfo:
    """Find the most recent run fitted to one population, whoever is in it.

    Only runs of the OU model (``MODEL_NAME``) count, since the alert tables
    read their ``latent_state`` rows. ``spec.seasons`` is ignored; the newest
    run of the role, metric and pitch type wins.

    Args:
        connection: Warehouse connection.
        spec: Role, metric and pitch type to match.

    Returns:
        The run's identifier and what it was fitted to.

    Raises:
        LookupError: If no stored run matches.
    """
    parameters = {
        "model_name": MODEL_NAME,
        "role": spec.role,
        "metric": spec.metric,
        "pitch_type": spec.pitch_type,
    }
    row = connection.execute(LATEST_POPULATION_RUN_SQL, parameters).fetchone()
    if row is None:
        raise LookupError(
            f"No {spec.role} run of {spec.metric} for pitch type {spec.pitch_type}"
        )
    run_id, run_role, run_metric, pitch_type, pitch_name = row
    return RunInfo(run_id, run_role, run_metric, pitch_type, pitch_name)


def load_alert_states(
    connection: duckdb.DuckDBPyConnection, run_id: uuid.UUID
) -> pl.DataFrame:
    """Read every player's real-time talent, gap to own level and alert score.

    Runs stored before alert scores were recorded return no rows.

    Args:
        connection: Warehouse connection.
        run_id: Run identifier.

    Returns:
        ``player_id``, ``full_name``, ``game_date``, ``filtered_mean``,
        ``filtered_gap_mean`` and ``alert_z``, by player and date.
    """
    return connection.execute(ALERT_STATES_SQL, {"run_id": str(run_id)}).pl()
