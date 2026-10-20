"""Command-line interface: ingest, load, fit, trajectory, chart and alerts."""

import asyncio
import logging
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final, NamedTuple

import duckdb
import jax
import polars as pl
import typer

from statecast.config import DEFAULT_SETTINGS_PATH, Settings, load_settings
from statecast.evaluation.injury_alerts import (
    PLACEBO_GROUP,
    TARGET_GROUP,
    auc_table,
    current_alerts,
    event_study,
    group_contrast,
    label_appearances,
    phase_sample,
    relative_to_healthy,
    threshold_table,
)
from statecast.evaluation.usage_model import (
    coefficient_table,
    cross_validated_auc,
    model_sample,
)
from statecast.features.appearance import hitter_appearances, pitcher_appearances
from statecast.features.injuries import il_placements
from statecast.features.usage import pitcher_usage, relative_usage
from statecast.ingest.pipeline import run_ingest
from statecast.logging_setup import configure_logging
from statecast.models.estimation import (
    FitResult,
    fit_population,
    initial_guess,
    summarize_fit,
)
from statecast.models.kalman import ObservationBatch, batch_from_frame
from statecast.models.trajectory import smooth_population, trajectory_frame
from statecast.storage.model_runs import (
    RunInfo,
    RunSpec,
    find_player,
    latest_run,
    latest_run_for_player,
    load_alert_states,
    load_trajectory,
    pitch_type_name,
    save_run,
)
from statecast.storage.warehouse import build_warehouse
from statecast.viz.injury_alerts import plot_event_study
from statecast.viz.seasonal import plot_seasonal_curve
from statecast.viz.trajectories import (
    chart_labels,
    figure_filename,
    plot_talent_fan,
    save_figure,
    season_span,
)

logger = logging.getLogger(__name__)
ALERT_THRESHOLDS: Final = (1.0, 1.5, 2.0, 2.5, 3.0)
MIN_MODEL_POSITIVES: Final = 20

app = typer.Typer(help="statecast data pipeline.", no_args_is_help=True)

SettingsOption = Annotated[
    Path, typer.Option("--settings", help="Path to settings.toml.")
]
DateOption = Annotated[datetime, typer.Option(formats=["%Y-%m-%d"])]
SeasonsOption = Annotated[
    list[int], typer.Option("--season", help="Season to fit; repeatable.")
]
MetricOption = Annotated[str | None, typer.Option(help="Metric column.")]
PitchTypeOption = Annotated[str, typer.Option(help="Pitch type (pitchers).")]
HorizonOption = Annotated[
    int, typer.Option(help="Days after a game that count as before an IL stint.")
]
ThresholdOption = Annotated[
    float, typer.Option(help="Alert when talent is this many sd below level.")
]
AnyPitchTypeOption = Annotated[
    str | None, typer.Option(help="Pitch type, such as FF; any when omitted.")
]


def _load_or_exit(settings_path: Path) -> Settings:
    """Load settings, exiting with an error message if they are invalid.

    Args:
        settings_path: Path to ``settings.toml``.

    Returns:
        The loaded settings.

    Raises:
        typer.Exit: With code 1 if the settings cannot be loaded.
    """
    try:
        return load_settings(settings_path)
    except (FileNotFoundError, ValueError) as error:
        logger.error("Could not load settings: %s", error)
        raise typer.Exit(code=1) from error


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    """Configure logging before any command runs.

    Args:
        verbose: Log at DEBUG level instead of INFO.
    """
    configure_logging(logging.DEBUG if verbose else logging.INFO)


@app.command()
def ingest(
    start: DateOption,
    end: DateOption,
    settings_path: SettingsOption = DEFAULT_SETTINGS_PATH,
) -> None:
    """Fetch schedule, Statcast, transactions and people into the raw cache.

    Args:
        start: First date, YYYY-MM-DD.
        end: Last date, YYYY-MM-DD (clamped to yesterday).
        settings_path: Path to ``settings.toml``.

    Raises:
        typer.Exit: With code 1 on invalid input or failed dates.
    """
    settings = _load_or_exit(settings_path)
    try:
        summary = asyncio.run(run_ingest(settings, start.date(), end.date()))
    except ValueError as error:
        logger.error("Ingest aborted: %s", error)
        raise typer.Exit(code=1) from error
    logger.info(
        "Ingested %d games, %d pitches over %d days, %d transactions, %d new players",
        summary.games,
        summary.statcast.rows_written,
        summary.statcast.days_written,
        summary.transactions,
        summary.new_players,
    )
    if summary.statcast.failed_dates:
        failed = ", ".join(day.isoformat() for day in summary.statcast.failed_dates)
        logger.error("Statcast failed for %s; rerun ingest to retry", failed)
        raise typer.Exit(code=1)


class Role(StrEnum):
    """Which side of the plate a population is drawn from."""

    PITCHER = "pitcher"
    HITTER = "hitter"


def _load_and_fit(
    connection: duckdb.DuckDBPyConnection, spec: RunSpec
) -> tuple[pl.DataFrame, ObservationBatch, FitResult]:
    """Pull one population's appearances and fit the talent model to them.

    Args:
        connection: Warehouse connection.
        spec: Role, metric, pitch type and seasons to fit.

    Returns:
        The appearances, their padded batch and the fit.

    Raises:
        typer.Exit: With code 1 if the metric is unsupported or no data matches.
    """
    try:
        frame = (
            pitcher_appearances(
                connection, spec.pitch_type or "FF", spec.seasons, spec.metric
            )
            if spec.role == Role.PITCHER
            else hitter_appearances(connection, spec.seasons, spec.metric)
        )
    except ValueError as error:
        logger.error("%s", error)
        raise typer.Exit(code=1) from error
    if frame.is_empty():
        logger.error("No appearances match; load these seasons first")
        raise typer.Exit(code=1)
    batch, player_ids = batch_from_frame(frame)
    logger.info("Fitting %d players, %d appearances", player_ids.shape[0], frame.height)
    result: FitResult = jax.jit(fit_population)(batch, initial_guess(batch))
    logger.info(
        "Converged in %d iterations (gradient norm %.2e)",
        int(result.iterations),
        float(result.gradient_norm),
    )
    return frame, batch, result


def _echo_table(table: pl.DataFrame) -> None:
    """Print a table in full; Polars hides middle rows past ten by default.

    Args:
        table: Table to print.
    """
    with pl.Config(tbl_rows=-1):
        typer.echo(table)


def _echo_section(title: str, table: pl.DataFrame) -> None:
    """Print a heading, then a table in full.

    Args:
        title: Heading.
        table: Table to print.
    """
    typer.echo(f"\n{title}")
    _echo_table(table)


def _run_spec(
    role: Role, seasons: list[int], metric: str | None, pitch_type: str
) -> RunSpec:
    """Fill in the default metric for a role and drop pitch type for hitters.

    Args:
        role: ``pitcher`` or ``hitter``.
        seasons: Seasons to fit.
        metric: Metric column, or ``None`` for the role's default.
        pitch_type: Pitch type (ignored for hitters).

    Returns:
        The run specification.
    """
    if role is Role.PITCHER:
        return RunSpec(role, metric or "release_speed", pitch_type, tuple(seasons))
    return RunSpec(role, metric or "bat_speed", None, tuple(seasons))


@app.command()
def fit(
    role: Role,
    season: SeasonsOption,
    metric: MetricOption = None,
    pitch_type: PitchTypeOption = "FF",
    settings_path: SettingsOption = DEFAULT_SETTINGS_PATH,
) -> None:
    """Fit the talent model to one population and print the estimates.

    Args:
        role: ``pitcher`` or ``hitter``.
        season: Seasons to include (regular season only).
        metric: Metric column; defaults to release_speed or bat_speed.
        pitch_type: Pitch type for pitchers, such as ``FF``.
        settings_path: Path to ``settings.toml``.
    """
    settings = _load_or_exit(settings_path)
    spec = _run_spec(role, season, metric, pitch_type)
    with duckdb.connect(str(settings.paths.warehouse_path), read_only=True) as con:
        _, _, result = _load_and_fit(con, spec)
    _echo_table(summarize_fit(result))


@app.command()
def trajectory(
    role: Role,
    season: SeasonsOption,
    metric: MetricOption = None,
    pitch_type: PitchTypeOption = "FF",
    settings_path: SettingsOption = DEFAULT_SETTINGS_PATH,
) -> None:
    """Fit, smooth every player's talent path and store the run in the warehouse.

    Also draws the fitted league seasonal curve into the figures folder.

    Args:
        role: ``pitcher`` or ``hitter``.
        season: Seasons to include (regular season only).
        metric: Metric column; defaults to release_speed or bat_speed.
        pitch_type: Pitch type for pitchers, such as ``FF``.
        settings_path: Path to ``settings.toml``.
    """
    settings = _load_or_exit(settings_path)
    spec = _run_spec(role, season, metric, pitch_type)
    with duckdb.connect(str(settings.paths.warehouse_path)) as con:
        frame, batch, result = _load_and_fit(con, spec)
        filtered, smoothed = smooth_population(result.params, batch)
        paths = trajectory_frame(frame, batch, result.params, filtered, smoothed)
        run_id = save_run(con, spec, result, paths)
        pitch_name = pitch_type_name(con, spec.pitch_type)
    summary = summarize_fit(result)
    _echo_table(summary)
    logger.info("Saved run %s with %d player-games", run_id, paths.height)
    labels = chart_labels("League seasonal curve", spec.metric, pitch_name, season)
    output = settings.paths.figures_dir / figure_filename(
        "league seasonal", spec.metric, spec.pitch_type
    )
    subtitle = (
        f"Shift from the July 1 level, {season_span(season)} regular season, "
        f"with 95% intervals ({labels.unit})"
    )
    save_figure(plot_seasonal_curve(summary, labels.title, subtitle), output)
    typer.echo(output)


@app.command()
def chart(
    player: Annotated[str, typer.Argument(help="Player's full name.")],
    role: Annotated[Role, typer.Option(help="Which run to draw from.")] = Role.PITCHER,
    metric: MetricOption = None,
    pitch_type: AnyPitchTypeOption = None,
    settings_path: SettingsOption = DEFAULT_SETTINGS_PATH,
) -> None:
    """Draw a player's talent fan chart from the latest matching stored run.

    Without ``--metric`` or ``--pitch-type`` the newest run that includes the
    player wins, so pass ``--pitch-type`` to pick between a pitcher's FF and
    SI runs.

    Args:
        player: Full name, such as ``Zack Wheeler`` (case and accents ignored).
        role: ``pitcher`` or ``hitter``.
        metric: Metric column; any metric when omitted.
        pitch_type: Pitch-type code; any pitch type when omitted.
        settings_path: Path to ``settings.toml``.

    Raises:
        typer.Exit: With code 1 if the player or a run for them is not found.
    """
    settings = _load_or_exit(settings_path)
    with duckdb.connect(str(settings.paths.warehouse_path), read_only=True) as con:
        try:
            player_id, full_name = find_player(con, player)
            run = latest_run_for_player(con, player_id, role, metric, pitch_type)
        except LookupError as error:
            logger.error("%s; run `statecast trajectory` first if needed", error)
            raise typer.Exit(code=1) from error
        path_frame = load_trajectory(con, run.run_id, player_id)
    seasons = sorted(set(path_frame["game_date"].dt.year().to_list()))
    labels = chart_labels(full_name, run.metric, run.pitch_name, seasons)
    output = settings.paths.figures_dir / figure_filename(
        full_name, run.metric, run.pitch_type
    )
    save_figure(plot_talent_fan(path_frame, labels), output)
    typer.echo(output)


class AlertInputs(NamedTuple):
    """Everything the alerts report reads from the warehouse.

    Attributes:
        run: The scored run.
        states: Real-time alert scores per appearance.
        placements: Every IL placement.
        usage: Real-time workload scores per appearance.
    """

    run: RunInfo
    states: pl.DataFrame
    placements: pl.DataFrame
    usage: pl.DataFrame


def _load_alert_inputs(warehouse_path: Path, spec: RunSpec) -> AlertInputs:
    """Read the latest scored run, IL placements and workload.

    Args:
        warehouse_path: DuckDB database file.
        spec: Role, metric and pitch type of the run.

    Returns:
        The inputs for the alerts report.

    Raises:
        typer.Exit: With code 1 if no run exists or it has no alert scores.
    """
    with duckdb.connect(str(warehouse_path), read_only=True) as con:
        try:
            run = latest_run(con, spec)
        except LookupError as error:
            logger.error("%s; run `statecast trajectory pitcher` first", error)
            raise typer.Exit(code=1) from error
        states = load_alert_states(con, run.run_id)
        if states.is_empty():
            logger.error("Run %s has no alert scores; rerun trajectory", run.run_id)
            raise typer.Exit(code=1)
        seasons = sorted(set(states["game_date"].dt.year().to_list()))
        usage = relative_usage(pitcher_usage(con, seasons))
        return AlertInputs(run, states, il_placements(con), usage)


def _echo_usage_model(labeled: pl.DataFrame, usage: pl.DataFrame, horizon: int) -> None:
    """Print whether velocity adds to workload, if there is enough data.

    Args:
        labeled: Output of ``label_appearances``.
        usage: Output of ``relative_usage``.
        horizon: Days before a placement that count.
    """
    warning = phase_sample(labeled, horizon, TARGET_GROUP, "warning")
    sample = model_sample(warning, usage)
    positives = int(sample["positive"].sum())
    if positives < MIN_MODEL_POSITIVES or sample.height - positives < positives:
        logger.warning("Only %d warning games before arm stints; skipping", positives)
        return
    _echo_section(
        "Arm IL odds ratio per sd, warning games (95% cluster-robust)",
        coefficient_table(sample),
    )
    _echo_section(
        "Out-of-fold AUC, folds split by pitcher", cross_validated_auc(sample)
    )


@app.command()
def alerts(
    pitch_type: PitchTypeOption = "FF",
    horizon_days: HorizonOption = 30,
    threshold: ThresholdOption = 2.0,
    settings_path: SettingsOption = DEFAULT_SETTINGS_PATH,
) -> None:
    """Test whether real-time velocity drops come before IL placements.

    Uses the latest pitcher velocity run for the pitch type. Prints the
    warning-game threshold table, AUCs by injury group and phase with
    pitcher-bootstrap intervals, the arm-minus-lower-body contrast, whether
    velocity adds to workload, and end-of-season alerts with what came next,
    then draws the event-study chart.

    Args:
        pitch_type: Pitch type of the run to score, such as ``FF``.
        horizon_days: Days after a game that count as before an IL stint.
        threshold: Alert threshold for the end-of-season table.
        settings_path: Path to ``settings.toml``.

    Raises:
        typer.Exit: With code 1 if no run has alert scores.
    """
    settings = _load_or_exit(settings_path)
    spec = RunSpec(Role.PITCHER, "release_speed", pitch_type, ())
    inputs = _load_alert_inputs(settings.paths.warehouse_path, spec)
    labeled = label_appearances(inputs.states, inputs.placements)
    study = event_study(labeled)
    _echo_section(
        f"Warning games before arm IL stints within {horizon_days} days",
        threshold_table(labeled, ALERT_THRESHOLDS, horizon_days),
    )
    _echo_section(
        "AUC against healthy games (95% pitcher-bootstrap intervals)",
        auc_table(labeled, horizon_days),
    )
    _echo_section(
        f"Mean alert score, {TARGET_GROUP} minus {PLACEBO_GROUP}, by appearance",
        group_contrast(study, TARGET_GROUP, PLACEBO_GROUP),
    )
    _echo_usage_model(labeled, inputs.usage, horizon_days)
    _echo_section(
        f"End-of-season alerts at or below -{threshold} sd, and what came next",
        current_alerts(inputs.states, inputs.placements, threshold),
    )
    seasons = sorted(set(inputs.states["game_date"].dt.year().to_list()))
    pitch_name = (inputs.run.pitch_name or pitch_type).lower()
    title = f"Velocity before injured-list stints, {pitch_name}"
    subtitle = (
        f"Mean real-time alert score by appearance relative to healthy games, "
        f"{season_span(seasons)} regular season"
    )
    try:
        figure = plot_event_study(relative_to_healthy(study), title, subtitle)
    except ValueError as error:
        logger.error("No event-study chart: %s", error)
        raise typer.Exit(code=1) from error
    output = settings.paths.figures_dir / figure_filename(
        "il event study", spec.metric, pitch_type
    )
    save_figure(figure, output)
    typer.echo(output)


@app.command()
def load(
    season: Annotated[
        list[int] | None, typer.Option(help="Season to load; repeatable.")
    ] = None,
    settings_path: SettingsOption = DEFAULT_SETTINGS_PATH,
) -> None:
    """Load cached seasons into the DuckDB warehouse.

    Args:
        season: Seasons to load; every cached season when omitted.
        settings_path: Path to ``settings.toml``.
    """
    settings = _load_or_exit(settings_path)
    rows_written = build_warehouse(
        settings.paths.warehouse_path, settings.paths.raw_dir, season
    )
    for table, rows in sorted(rows_written.items()):
        logger.info("%s: %d rows", table, rows)
