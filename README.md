<p align="center">
  <img src="docs/logo.svg" alt="statecast" width="500">
</p>

State-space models of latent MLB player talent, built from Statcast data.
Each player's true talent for a metric (fastball velocity, spin rate, bat
speed and so on) is treated as a hidden stochastic process. A Kalman filter
separates it from game-to-game noise, so the model can say at any date where a
player really is, how sure it is, and whether he has moved away from his own
baseline.

**Status.** The data pipeline, the talent model and a calibrated real-time
alert score are complete. The alert score has been evaluated against
injured-list placements in a placebo-controlled study.

| Document | Contents |
| --- | --- |
| [docs/injury-study.md](docs/injury-study.md) | Do velocity, spin, extension or arm angle warn of arm injuries? |
| [docs/math.md](docs/math.md) | The model, estimation, alert score and evaluation statistics |
| [docs/data_dictionary.md](docs/data_dictionary.md) | Every table, column and unit |

## Findings so far

- **Talent moves slowly and mostly stays put.** Four-seam velocity has a
  talent half-life of about 100 days. Only about 12% of the variance in
  velocity talent is drift within a pitcher; the rest is permanent
  differences between pitchers. Extension and arm angle are even more stable.
- **The season has a shape.** League four-seam velocity runs about 0.4 mph
  below its July level in April, and spin about 17 rpm below. The model
  removes this before scoring anyone.
- **Velocity drops come before IL stints of every kind, not just arm
  injuries.** In the final game before a placement, the alert score separates
  injured from healthy games with AUC 0.61 for both arm and lower-body stints.
  No metric tested is arm-specific after a lower-body placebo contrast and a
  Holm correction.

![Velocity before injured-list stints, four-seam fastball](docs/figures/il_event_study_ff_release_speed.png)

## The model

For each player, talent $\theta(t)$ follows an Ornstein-Uhlenbeck process
around his own long-run level $\mu$, which is drawn from the population. Each
game adds a one-day shock shared by its pitches, and the league shares a
seasonal curve:

$$
d\theta = -\kappa(\theta - \mu)\,dt + \sigma\,dW, \qquad
\bar y_k \sim N\!\left(\theta(t_k) + g(t_k),\ \tau^2 + r^2/n_k\right)
$$

- The state $[\theta, \mu]$ is filtered exactly, with a closed-form transition
  over any gap between games, and smoothed with a Rauch-Tung-Striebel pass.
- All 12 population parameters are fitted by maximum likelihood in JAX with
  L-BFGS, with intervals from the inverse Hessian.
- The alert score $z = E[\theta - \mu \mid \text{games so far}] / \sqrt{v - P}$
  is exactly $N(0, 1)$ under the model, so a threshold means the same thing for
  every player and metric.

Details and derivations are in [docs/math.md](docs/math.md).

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                  # creates .venv with runtime + dev tools
uv pip install ipykernel ipywidgets      # notebook support, kept out of pyproject
```

Settings, including data paths and request limits, live in
`config/settings.toml`. No API keys are needed: Baseball Savant and the MLB
Stats API are public.

## Usage

### 1. Data

```bash
# Fetch schedule, Statcast pitches, transactions and players into data/raw/
uv run statecast ingest --start 2024-03-20 --end 2026-10-17

# Load the cache into the DuckDB warehouse
uv run statecast load                    # every cached season
uv run statecast load --season 2026      # or chosen seasons
```

Both commands are safe to rerun. `ingest` skips game dates already cached and
clamps the end date to yesterday. `load` replaces rows by primary key, one
season per transaction. A season is about 190 game dates of Savant requests;
backfill season by season.

### 2. Talent model

```bash
# Fit and print the population parameters only
uv run statecast fit pitcher --season 2024 --season 2025 --season 2026

# Fit, filter and smooth every player, store the run, draw the seasonal curve
uv run statecast trajectory pitcher --season 2024 --season 2025 --season 2026
uv run statecast trajectory pitcher --pitch-type SI --metric arm_angle \
  --season 2024 --season 2025 --season 2026
uv run statecast trajectory hitter --season 2024 --season 2025 --season 2026

# Draw one player's talent fan chart from his latest stored run
uv run statecast chart "Zack Wheeler" --pitch-type FF
uv run statecast chart "Aaron Judge" --role hitter
```

| Role | Metrics (`--metric`) | Default |
| --- | --- | --- |
| `pitcher` | `release_speed`, `spin_rate`, `release_extension`, `arm_angle` | `release_speed`, `--pitch-type FF` |
| `hitter` | `bat_speed`, `swing_length`, `attack_angle`, `swing_path_tilt` | `bat_speed` |

Pitcher runs include a pitch type only for pitchers who have it among their
three most-used pitches. Hitter runs use competitive swings (50 mph and up, no
bunts).

### 3. Injury alerts

These commands read the latest stored `trajectory` run for each metric, so fit
that first.

```bash
# One metric: AUCs by injury group, placebo contrast, thresholds,
# workload model, current alerts, event-study chart
uv run statecast alerts --pitch-type FF
uv run statecast alerts --pitch-type FF --metric spin_rate

# Every stored metric at once, Holm-adjusted, with a forest chart
uv run statecast scan
uv run statecast scan --pitch-type SI
```

Each metric is scored in its worrying direction by default: a drop for
velocity, spin rate and extension, a change in either direction for arm angle.
Override with `--direction` and the IL window with `--horizon-days`.

Charts are written to `outputs/figures/`, and every command prints the path.

## Data flow

| Stage | Location | Contents |
| --- | --- | --- |
| Raw cache | `data/raw/statcast/season=YYYY/date=YYYY-MM-DD.parquet` | One file per game date, typed Statcast pitches |
| Raw cache | `data/raw/statsapi/{schedule,transactions}/season=YYYY.parquet`, `people/people.parquet` | MLB Stats API rows |
| Warehouse | `data/warehouse/statecast.duckdb` | Normalized source tables, plus `model_run`, `model_parameter` and `latent_state` for model output |
| Figures | `outputs/figures/` | Seasonal curves, fan charts, event studies, metric scans |

The raw cache is the source of truth; the warehouse can be rebuilt from it at
any time. Every table and column is described in
[docs/data_dictionary.md](docs/data_dictionary.md).

Data is © MLB Advanced Media, for personal, non-commercial use. Nothing under
`data/` or `outputs/` is committed.

## Project layout

```text
src/statecast/
├── cli.py              typer commands
├── config.py           settings.toml loader
├── ingest/             Savant and Stats API clients, raw-cache pipeline
├── storage/            Parquet cache, DuckDB schema and loader, model runs
├── features/           appearance statistics, IL placements, workload
├── models/             OU dynamics, Kalman filter and smoother, seasonal
│                       curve, maximum-likelihood fit, trajectories
├── evaluation/         simulation, injury alerts, metric scan, workload model
└── viz/                chart theme and figures
docs/                   injury study, math, data dictionary, committed figures
tests/                  pytest suite; outputs/ holds files written by tests
```

## Development

Run these before every commit:

```bash
uv run ruff format && uv run ruff check && uv run mypy && uv run pytest
```

- Ruff and mypy run in strict mode.
- Tests mock every HTTP call and include simulation checks: parameter recovery,
  identical fits in any units, a calibrated N(0, 1) alert score, and a
  simulated injury fade that the evaluation must find.
- Docstring examples run as doctests.
- Files written by tests are kept under `tests/outputs/` (gitignored).

## Roadmap

- [x] **Phase 0: data.** Ingestion, raw cache and normalized warehouse.
- [x] **Phase 1: talent model.** OU state-space model with game-day effects
  and a league seasonal curve; filtered and smoothed talent for pitchers and
  hitters.
- [x] **Phase 2: alerts.** Calibrated alert score, injured-list evaluation
  with a placebo contrast, workload control and a multi-metric scan.
- [ ] **Phase 3: jumps.** A jump component in the talent process, filtered
  with a Rao-Blackwellized particle filter, so sudden changes (injury,
  mechanical change) are caught in one game.
- [ ] **Phase 4: joint metrics.** Velocity, spin, extension and arm angle as
  one multivariate state.