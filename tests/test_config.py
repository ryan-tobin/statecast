from pathlib import Path

import pytest

from statecast.config import load_settings

VALID_SETTINGS = """
[paths]
raw_dir = "data/raw"
warehouse_path = "data/warehouse/statecast.duckdb"
figures_dir = "outputs/figures"

[ingest]
game_types = ["R", "W"]
max_concurrency = 3
timeout_seconds = 30.0
max_retries = 4
user_agent = "agent"
"""


def write_settings(output_dir: Path, text: str) -> Path:
    settings_path = output_dir / "config" / "settings.toml"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(text, encoding="utf-8")
    return settings_path


def test_load_settings_resolves_paths_against_project_root(output_dir: Path) -> None:
    settings_path = write_settings(output_dir, VALID_SETTINGS)

    settings = load_settings(settings_path)

    assert settings.paths.raw_dir == output_dir.resolve() / "data" / "raw"
    assert settings.ingest.game_types == ("R", "W")
    assert settings.ingest.max_concurrency == 3


def test_load_settings_reports_missing_key(output_dir: Path) -> None:
    settings_path = write_settings(
        output_dir, VALID_SETTINGS.replace('user_agent = "agent"', "")
    )

    with pytest.raises(ValueError, match="user_agent"):
        load_settings(settings_path)


def test_load_settings_rejects_zero_concurrency(output_dir: Path) -> None:
    settings_path = write_settings(
        output_dir, VALID_SETTINGS.replace("max_concurrency = 3", "max_concurrency = 0")
    )

    with pytest.raises(ValueError, match="at least 1"):
        load_settings(settings_path)


def test_load_settings_rejects_empty_game_types(output_dir: Path) -> None:
    settings_path = write_settings(
        output_dir, VALID_SETTINGS.replace('["R", "W"]', "[]")
    )

    with pytest.raises(ValueError, match="game_types"):
        load_settings(settings_path)


def test_project_settings_file_loads() -> None:
    project_settings = Path(__file__).parent.parent / "config" / "settings.toml"

    settings = load_settings(project_settings)

    assert settings.ingest.game_types == ("R", "F", "D", "L", "W")
