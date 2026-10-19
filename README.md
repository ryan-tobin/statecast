# statecast

Continuous-time jump-diffusion state-space models of latent MLB player talent,
built from Statcast data. The goal is to separate real skill changes from noise
and flag breakouts, mechanical changes and injury risk with calibrated
posterior probabilities.

**Status: Phase 0 (data foundation).** Ingestion, raw cache and warehouse are
in place. The models arrive in Phase 1.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                  # creates .venv with runtime + dev tools
uv pip install ipykernel ipywidgets      # notebook support, kept out of pyproject
```

## Usage

```bash
# 1. Fetch schedule, Statcast pitches, transactions and players into data/raw/
uv run statecast ingest --start 2026-03-26 --end 2026-10-17

# 2. Load the cache into the DuckDB warehouse at data/warehouse/statecast.duckdb
uv run statecast load                    # every cached season
uv run statecast load --season 2026      # or chosen seasons
```

Both commands are safe to rerun. `ingest` skips game dates already cached and
clamps the end date to yesterday. `load` replaces rows by primary key, one
season per transaction.

A full Hawk-Eye-era backfill (2020 to 2026) is about 1,300 game dates of
Savant requests at 4 concurrent requests. Run it season by season.

## Data flow

| Stage | Location | Contents |
| --- | --- | --- |
| Raw cache | `data/raw/statcast/season=YYYY/date=YYYY-MM-DD.parquet` | One file per game date, typed Statcast pitches |
| Raw cache | `data/raw/statsapi/{schedule,transactions}/season=YYYY.parquet`, `people/people.parquet` | MLB Stats API rows |
| Warehouse | `data/warehouse/statecast.duckdb` | Normalized tables: `player`, `team`, `venue`, `game`, `pitch_type`, `pitcher_appearance`, `plate_appearance`, `pitch`, `swing`, `player_transaction` |

`pitch.plate_reference` records where `plate_x`/`plate_z` were measured: the
front of the plate through 2025, the middle from 2026 (ABS alignment).

Data is © MLB Advanced Media, for personal, non-commercial use. Nothing under
`data/` is committed.

## Development

Run these before every commit:

```bash
uv run ruff format && uv run ruff check && uv run mypy && uv run pytest
```

Tests mock every HTTP call. Files written by tests are kept under
`tests/outputs/` (gitignored).