"""Save jump-diffusion runs and read their per-game output back."""

import uuid
from dataclasses import dataclass

import duckdb
import polars as pl

from statecast.models.jumps import JumpFitResult, summarize_jump_fit
from statecast.models.rbpf import FilterSettings
from statecast.storage.model_runs import RunInfo, RunSpec, model_run_row
from statecast.storage.warehouse import initialize_schema, upsert_table

JUMP_MODEL_NAME = "ou_jump_seasonal"

LATEST_JUMP_RUN_SQL = """
SELECT r.run_id, r.role, r.metric, r.pitch_type, pt.name
FROM model_run AS r
LEFT JOIN pitch_type AS pt ON pt.code = r.pitch_type
WHERE r.model_name = $model_name
  AND r.role = $role
  AND ($metric IS NULL OR r.metric = $metric)
  AND ($pitch_type IS NULL OR r.pitch_type = $pitch_type)
  AND EXISTS (
      SELECT 1 FROM jump_state AS s
      WHERE s.run_id = r.run_id AND s.player_id = $player_id
  )
ORDER BY r.created_at DESC
LIMIT 1
"""

JUMP_TRAJECTORY_SQL = """
SELECT game_date, sample_size, appearance_mean, seasonal_effect,
       talent_mean, talent_sd, prior_jump_prob, jump_prob, jump_evidence,
       recent_jump_prob, recent_drop_prob
FROM jump_state
WHERE run_id = $run_id AND player_id = $player_id
ORDER BY game_date
"""

PLAYER_NAMES_SQL = "SELECT player_id, full_name FROM player"

LATEST_JUMP_POPULATION_SQL = """
SELECT r.run_id, r.role, r.metric, r.pitch_type, pt.name, j.recent_window,
       max(p.estimate) FILTER (WHERE p.parameter = 'jump_mean') AS jump_mean,
       max(p.estimate) FILTER (WHERE p.parameter = 'jump_sd') AS jump_sd
FROM model_run AS r
JOIN jump_run AS j USING (run_id)
JOIN model_parameter AS p USING (run_id)
LEFT JOIN pitch_type AS pt ON pt.code = r.pitch_type
WHERE r.model_name = $model_name
  AND r.role = $role
  AND r.metric = $metric
  AND r.pitch_type IS NOT DISTINCT FROM $pitch_type
GROUP BY ALL
ORDER BY max(r.created_at) DESC
LIMIT 1
"""

JUMP_STATES_SQL = """
SELECT player_id, game_date, prior_jump_prob, jump_prob, jump_evidence,
       recent_jump_prob, recent_drop_prob
FROM jump_state
WHERE run_id = $run_id
  AND jump_evidence IS NOT NULL
ORDER BY player_id, game_date
"""


@dataclass(frozen=True, slots=True)
class JumpRunDetails:
    """What a jump run stores beyond the shared run tables.

    Attributes:
        ou_log_likelihood: Log-likelihood of the OU fit to the same data.
        particle_log_likelihood: The particle filter's estimate of the jump
            model's log-likelihood, a check on the fitted approximation.
        settings: Particle count and recent-window length.
        seed: Random seed of the particle filter.
    """

    ou_log_likelihood: float
    particle_log_likelihood: float
    settings: FilterSettings
    seed: int


@dataclass(frozen=True, slots=True)
class JumpRunSummary:
    """A stored jump run and what scoring its output needs.

    Attributes:
        run: The run's identifier and what it was fitted to.
        recent_window: Appearances in the filter's recent window.
        jump_mean: Fitted mean jump size.
        jump_sd: Fitted spread of jump sizes.
    """

    run: RunInfo
    recent_window: int
    jump_mean: float
    jump_sd: float


def save_jump_run(
    connection: duckdb.DuckDBPyConnection,
    spec: RunSpec,
    fit: JumpFitResult,
    states: pl.DataFrame,
    details: JumpRunDetails,
) -> uuid.UUID:
    """Store a jump run, its parameters, its details and every player-game.

    Everything is written in one transaction, so a failed save leaves nothing
    behind.

    Args:
        connection: Writable warehouse connection.
        spec: What the run was fitted to.
        fit: Output of ``fit_jump_population``.
        states: Output of ``jump_trajectory_frame``.
        details: Comparison log-likelihoods and filter settings.

    Returns:
        The new run's identifier.

    Raises:
        duckdb.Error: If any write fails (the transaction is rolled back).
    """
    run_id = uuid.uuid4()
    run = model_run_row(run_id, spec, JUMP_MODEL_NAME, states, fit)
    parameters = summarize_jump_fit(fit).with_columns(run_id=pl.lit(str(run_id)))
    extra = pl.DataFrame(
        {
            "run_id": [str(run_id)],
            "ou_log_likelihood": [details.ou_log_likelihood],
            "particle_log_likelihood": [details.particle_log_likelihood],
            "particle_count": [details.settings.particle_count],
            "recent_window": [details.settings.lag],
            "seed": [details.seed],
        }
    )
    rows = states.with_columns(run_id=pl.lit(str(run_id)))
    initialize_schema(connection)
    connection.begin()
    try:
        upsert_table(connection, "model_run", run)
        upsert_table(connection, "model_parameter", parameters)
        upsert_table(connection, "jump_run", extra)
        upsert_table(connection, "jump_state", rows)
    except duckdb.Error:
        connection.rollback()
        raise
    connection.commit()
    return run_id


def latest_jump_run_for_player(
    connection: duckdb.DuckDBPyConnection,
    player_id: int,
    role: str,
    metric: str | None = None,
    pitch_type: str | None = None,
) -> RunInfo:
    """Find the most recent jump run of a role that includes a player.

    Args:
        connection: Warehouse connection.
        player_id: MLBAM player id.
        role: ``pitcher`` or ``hitter``.
        metric: Restrict to this metric, or ``None`` for any.
        pitch_type: Restrict to this pitch type, or ``None`` for any.

    Returns:
        The run's identifier and what it was fitted to.

    Raises:
        LookupError: If no stored jump run includes the player.
    """
    parameters = {
        "model_name": JUMP_MODEL_NAME,
        "role": role,
        "metric": metric,
        "pitch_type": pitch_type,
        "player_id": player_id,
    }
    row = connection.execute(LATEST_JUMP_RUN_SQL, parameters).fetchone()
    if row is None:
        raise LookupError(f"No {role} jump run includes player {player_id}")
    run_id, run_role, run_metric, run_pitch_type, pitch_name = row
    return RunInfo(run_id, run_role, run_metric, run_pitch_type, pitch_name)


def load_jump_trajectory(
    connection: duckdb.DuckDBPyConnection, run_id: uuid.UUID, player_id: int
) -> pl.DataFrame:
    """Read one player's real-time talent and jump probabilities from a run.

    Args:
        connection: Warehouse connection.
        run_id: Run identifier.
        player_id: MLBAM player id.

    Returns:
        One row per appearance, ordered by date.
    """
    return connection.execute(
        JUMP_TRAJECTORY_SQL, {"run_id": str(run_id), "player_id": player_id}
    ).pl()


def load_player_names(connection: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Every player's id and full name.

    Args:
        connection: Warehouse connection.

    Returns:
        ``player_id`` (Int64) and ``full_name``.
    """
    return (
        connection.execute(PLAYER_NAMES_SQL)
        .pl()
        .with_columns(pl.col("player_id").cast(pl.Int64))
    )


def latest_jump_run(
    connection: duckdb.DuckDBPyConnection, spec: RunSpec
) -> JumpRunSummary:
    """Find the most recent jump run fitted to one population.

    ``spec.seasons`` is ignored; the newest jump run of the role, metric and
    pitch type wins.

    Args:
        connection: Warehouse connection.
        spec: Role, metric and pitch type to match.

    Returns:
        The run, its recent-window length and its fitted jump size.

    Raises:
        LookupError: If no stored jump run matches.
    """
    parameters = {
        "model_name": JUMP_MODEL_NAME,
        "role": spec.role,
        "metric": spec.metric,
        "pitch_type": spec.pitch_type,
    }
    row = connection.execute(LATEST_JUMP_POPULATION_SQL, parameters).fetchone()
    if row is None:
        raise LookupError(
            f"No {spec.role} jump run of {spec.metric} for pitch type {spec.pitch_type}"
        )
    run_id, role, metric, pitch_type, pitch_name, window, mean, sd = row
    return JumpRunSummary(
        run=RunInfo(run_id, role, metric, pitch_type, pitch_name),
        recent_window=int(window),
        jump_mean=float(mean),
        jump_sd=float(sd),
    )


def load_jump_states(
    connection: duckdb.DuckDBPyConnection, run_id: uuid.UUID
) -> pl.DataFrame:
    """Read every player-game's jump probabilities and evidence from a run.

    Runs stored before jump evidence was recorded return no rows.

    Args:
        connection: Warehouse connection.
        run_id: Run identifier.

    Returns:
        ``player_id``, ``game_date``, ``prior_jump_prob``, ``jump_prob``,
        ``jump_evidence``, ``recent_jump_prob`` and ``recent_drop_prob``, by
        player and date.
    """
    return connection.execute(JUMP_STATES_SQL, {"run_id": str(run_id)}).pl()
