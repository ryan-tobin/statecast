"""Project settings loaded from ``config/settings.toml``."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SETTINGS_PATH = Path("config/settings.toml")


@dataclass(frozen=True, slots=True)
class PathSettings:
    """Filesystem locations used by the pipeline.

    Attributes:
        raw_dir: Root of the immutable raw Parquet cache.
        warehouse_path: DuckDB database file.
        figures_dir: Folder for rendered charts.
    """

    raw_dir: Path
    warehouse_path: Path
    figures_dir: Path


@dataclass(frozen=True, slots=True)
class IngestSettings:
    """HTTP and scope settings for data ingestion.

    Attributes:
        game_types: MLB game-type codes to keep (for example ``R`` for regular season).
        max_concurrency: Maximum simultaneous requests to one data source.
        timeout_seconds: Per-request timeout.
        max_retries: Attempts per request before giving up.
        user_agent: User-Agent header sent with every request.
    """

    game_types: tuple[str, ...]
    max_concurrency: int
    timeout_seconds: float
    max_retries: int
    user_agent: str


@dataclass(frozen=True, slots=True)
class Settings:
    """All project settings.

    Attributes:
        paths: Filesystem locations.
        ingest: Ingestion settings.
    """

    paths: PathSettings
    ingest: IngestSettings


def load_settings(settings_path: Path = DEFAULT_SETTINGS_PATH) -> Settings:
    """Load settings from a TOML file.

    Relative paths in the file are resolved against the project root, which is
    the parent of the folder holding the settings file.

    Args:
        settings_path: Path to ``settings.toml``.

    Returns:
        The parsed settings.

    Raises:
        FileNotFoundError: If the settings file does not exist.
        ValueError: If a required key is missing or a value is out of range.

    Example:
        >>> settings = load_settings(Path("config/settings.toml"))
        >>> settings.ingest.max_concurrency
        4
    """
    resolved_path = settings_path.resolve()
    with resolved_path.open("rb") as settings_file:
        raw_settings = tomllib.load(settings_file)
    project_root = resolved_path.parent.parent
    try:
        paths = PathSettings(
            raw_dir=project_root / raw_settings["paths"]["raw_dir"],
            warehouse_path=project_root / raw_settings["paths"]["warehouse_path"],
            figures_dir=project_root / raw_settings["paths"]["figures_dir"],
        )
        ingest_table = raw_settings["ingest"]
        ingest = IngestSettings(
            game_types=tuple(ingest_table["game_types"]),
            max_concurrency=int(ingest_table["max_concurrency"]),
            timeout_seconds=float(ingest_table["timeout_seconds"]),
            max_retries=int(ingest_table["max_retries"]),
            user_agent=str(ingest_table["user_agent"]),
        )
    except KeyError as error:
        raise ValueError(f"Missing setting {error} in {resolved_path}") from error
    if ingest.max_concurrency < 1 or ingest.max_retries < 1:
        raise ValueError("max_concurrency and max_retries must both be at least 1")
    if not ingest.game_types:
        raise ValueError("game_types must list at least one game type")
    return Settings(paths=paths, ingest=ingest)
