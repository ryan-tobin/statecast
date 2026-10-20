"""Command-line interface: ingest, load, fit, trajectory and chart."""

import asyncio
import logging
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import duckdb
import jax
import polars as pl
import typer

from statecast.config import DEFAULT_SETTINGS_PATH, Settings, load_settings
from statecast.features.appearance import hitter_appearances, pitcher_appearances
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
    RunSpec,
    find_player,
    latest_run_for_player,
    load_trajectory,
    pitch_type_name,
    save_run,
)
from statecast.storage.warehouse import build_warehouse
from statecast.viz.seasonal import plot_seasonal_curve
from statecast.viz.trajectories import (
    chart_labels,
    figure_filename,
    plot_talent_fan,
    save_figure,
    season_span,
)

logger = logging.getLogger(__name__)

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
        paths = trajectory_frame(
            frame, batch, result.params.seasonal_knots, filtered, smoothed
        )
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
    settings_path: SettingsOption = DEFAULT_SETTINGS_PATH,
) -> None:
    """Draw a player's talent fan chart from the latest stored run.

    Args:
        player: Full name, such as ``Zack Wheeler`` (case and accents ignored).
        role: ``pitcher`` or ``hitter``.
        metric: Metric column; any metric when omitted.
        settings_path: Path to ``settings.toml``.

    Raises:
        typer.Exit: With code 1 if the player or a run for them is not found.
    """
    settings = _load_or_exit(settings_path)
    with duckdb.connect(str(settings.paths.warehouse_path), read_only=True) as con:
        try:
            player_id, full_name = find_player(con, player)
            run = latest_run_for_player(con, player_id, role, metric)
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