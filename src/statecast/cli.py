"""Command-line interface: ``statecast ingest``, ``load`` and ``fit``."""

import asyncio
import logging
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import duckdb
import jax
import typer

from statecast.config import DEFAULT_SETTINGS_PATH, Settings, load_settings
from statecast.features.appearance import hitter_appearances, pitcher_appearances
from statecast.ingest.pipeline import run_ingest
from statecast.logging_setup import configure_logging
from statecast.models.estimation import fit_population, initial_guess, summarize_fit
from statecast.models.kalman import batch_from_frame
from statecast.storage.warehouse import build_warehouse

logger = logging.getLogger(__name__)

app = typer.Typer(help="statecast data pipeline.", no_args_is_help=True)

SettingsOption = Annotated[
    Path, typer.Option("--settings", help="Path to settings.toml.")
]
DateOption = Annotated[datetime, typer.Option(formats=["%Y-%m-%d"])]


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


@app.command()
def fit(
    role: Role,
    season: Annotated[list[int], typer.Option(help="Season to fit; repeatable.")],
    metric: Annotated[str | None, typer.Option(help="Metric column.")] = None,
    pitch_type: Annotated[str, typer.Option(help="Pitch type (pitchers).")] = "FF",
    settings_path: SettingsOption = DEFAULT_SETTINGS_PATH,
) -> None:
    """Fit the Phase 1 talent model to one population and print the estimates.

    Args:
        role: ``pitcher`` or ``hitter``.
        season: Seasons to include (regular season only).
        metric: Metric column; defaults to release_speed or bat_speed.
        pitch_type: Pitch type for pitchers, such as ``FF``.
        settings_path: Path to ``settings.toml``.

    Raises:
        typer.Exit: With code 1 if the metric is unsupported or no data matches.
    """
    settings = _load_or_exit(settings_path)
    with duckdb.connect(str(settings.paths.warehouse_path), read_only=True) as con:
        try:
            frame = (
                pitcher_appearances(con, pitch_type, season, metric or "release_speed")
                if role is Role.PITCHER
                else hitter_appearances(con, season, metric or "bat_speed")
            )
        except ValueError as error:
            logger.error("%s", error)
            raise typer.Exit(code=1) from error
    if frame.is_empty():
        logger.error("No appearances match; load these seasons first")
        raise typer.Exit(code=1)
    batch, player_ids = batch_from_frame(frame)
    logger.info("Fitting %d players, %d appearances", player_ids.shape[0], frame.height)
    result = jax.jit(fit_population)(batch, initial_guess(batch))
    logger.info(
        "Converged in %d iterations (gradient norm %.2e)",
        int(result.iterations),
        float(result.gradient_norm),
    )
    typer.echo(summarize_fit(result))


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
