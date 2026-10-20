# Data dictionary

Every table and derived dataset in statecast: where it comes from, its key, and
what each column means and is measured in. The model behind the `model_*` and
`latent_state` tables is described in [math.md](math.md).

## Sources and storage

| Layer | Location | Contents |
| --- | --- | --- |
| Raw Statcast cache | `data/raw/statcast/season=YYYY/date=YYYY-MM-DD.parquet` | One file per game date: every pitch from the Baseball Savant Statcast Search CSV endpoint, as parsed |
| Raw Stats API cache | `data/raw/statsapi/<dataset>/season=YYYY.parquet` | `schedule` and `transactions` from the MLB Stats API, one file per season |
| Raw people cache | `data/raw/statsapi/people/people.parquet` | Player identity rows for every player id seen in the pitch data |
| Warehouse | `data/warehouse/statecast.duckdb` | Normalized DuckDB tables built from the raw cache, plus model output |
| Figures | `outputs/figures/` | Rendered charts |

The raw cache is the source of truth. It is written atomically and only ever
upserted, so the warehouse can be rebuilt from it at any time with
`statecast load`. Paths are set in `config/settings.toml`; `data/` and
`outputs/` are gitignored.

Ingest keeps these game types: `R` (regular season), `F` (wild card), `D`
(division series), `L` (league championship series) and `W` (World Series).
Postponed and cancelled games are dropped. Every model and evaluation in the
project uses regular-season games only.

## Conventions

- **Units** are Statcast's: miles per hour, revolutions per minute, feet,
  degrees, feet per second.
- **Directions** follow Statcast's catcher's-eye view: `x` is horizontal (positive
  to the catcher's right), `y` runs from home plate toward the mound, `z` is
  height above the ground.
- **Dates** are the official game date in local time (`DATE`). Timestamps are
  UTC (`TIMESTAMPTZ`).
- **NULL** means not tracked or not applicable, never zero.
- **Keys.** Relationships are logical, not enforced: DuckDB foreign keys would
  block the `INSERT OR REPLACE` upserts the loader uses. Every table has a
  primary key, and every load is an upsert on it.

## How the tables relate

- `game.game_pk` is referenced by `pitch`, `plate_appearance`, `swing` and
  `pitcher_appearance`.
- `pitch` is keyed by `(game_pk, at_bat_number, pitch_number)`. `swing` shares
  that key; `plate_appearance` is keyed by its first two parts.
- `player.player_id` is referenced by `pitch.pitcher_id`, `pitch.batter_id`,
  `pitcher_appearance.pitcher_id`, `player_transaction.player_id` and
  `latent_state.player_id`.
- `game.venue_id` references `venue`; `game.home_team_id` and
  `game.away_team_id` reference `team`; `pitch.pitch_type` references
  `pitch_type.code`.
- `model_run.run_id` is referenced by `model_parameter` and `latent_state`.

## Reference tables

### `player`

One row per player. Source: Stats API `/people`. Key: `player_id`.

| Column | Type | Description |
| --- | --- | --- |
| `player_id` | INTEGER | MLBAM player id, shared by Statcast and the Stats API |
| `full_name` | VARCHAR | Full name, such as `Dylan Cease` |
| `birth_date` | DATE | Date of birth |
| `throws` | VARCHAR | Throwing hand: `R` or `L` |
| `bats` | VARCHAR | Batting side: `R`, `L` or `S` (switch) |

### `team`

One row per team. Source: both sides of the Stats API schedule. Key: `team_id`.

| Column | Type | Description |
| --- | --- | --- |
| `team_id` | INTEGER | MLB team id |
| `name` | VARCHAR | Team name, such as `Philadelphia Phillies` |

### `venue`

One row per ballpark. Source: Stats API schedule. Key: `venue_id`.

| Column | Type | Description |
| --- | --- | --- |
| `venue_id` | INTEGER | MLB venue id |
| `name` | VARCHAR | Venue name |

### `pitch_type`

One row per Statcast pitch-type code seen in the data. Key: `code`.

| Column | Type | Description |
| --- | --- | --- |
| `code` | VARCHAR | Statcast code, such as `FF` |
| `name` | VARCHAR | Statcast name, such as `4-Seam Fastball` |

Common codes: `FF` 4-seam fastball, `SI` sinker, `FC` cutter, `SL` slider,
`ST` sweeper, `CU` curveball, `KC` knuckle curve, `CH` changeup, `FS`
splitter.

## Game tables

### `game`

One row per played game. Source: Stats API `/schedule`. Key: `game_pk`.

| Column | Type | Description |
| --- | --- | --- |
| `game_pk` | INTEGER | MLB game id |
| `game_date` | DATE | Official game date |
| `season` | SMALLINT | Season year |
| `game_type` | VARCHAR | `R`, `F`, `D`, `L` or `W` (see Sources) |
| `venue_id` | INTEGER | Ballpark (`venue`) |
| `home_team_id` | INTEGER | Home team (`team`) |
| `away_team_id` | INTEGER | Away team (`team`) |
| `first_pitch_at` | TIMESTAMPTZ | Scheduled first pitch, UTC |

### `pitcher_appearance`

One row per pitcher per game. Source: Statcast. Key: `(game_pk, pitcher_id)`.

| Column | Type | Units | Description |
| --- | --- | --- | --- |
| `game_pk` | INTEGER | | Game (`game`) |
| `pitcher_id` | INTEGER | | Pitcher (`player`) |
| `days_since_prev_game` | SMALLINT | days | Days since the pitcher's previous game, from Statcast |

### `plate_appearance`

One row per plate appearance. Source: Statcast, aggregated from its pitches.
Key: `(game_pk, at_bat_number)`.

| Column | Type | Description |
| --- | --- | --- |
| `game_pk` | INTEGER | Game (`game`) |
| `at_bat_number` | SMALLINT | Plate appearance number within the game |
| `inning` | TINYINT | Inning, from the first pitch |
| `inning_half` | VARCHAR | `Top` or `Bot` |
| `outs` | TINYINT | Outs when the plate appearance began (0 to 2) |
| `times_through_order` | TINYINT | Times the pitcher has faced the order, counting this time |
| `event` | VARCHAR | Outcome from the last pitch that recorded one, such as `single`, `strikeout`, `walk`; NULL if none was recorded |

### `pitch`

One row per pitch. Source: Statcast. Key: `(game_pk, at_bat_number, pitch_number)`.

| Column | Type | Units | Description |
| --- | --- | --- | --- |
| `game_pk` | INTEGER | | Game (`game`) |
| `at_bat_number` | SMALLINT | | Plate appearance (`plate_appearance`) |
| `pitch_number` | SMALLINT | | Pitch number within the plate appearance |
| `pitcher_id` | INTEGER | | Pitcher (`player`) |
| `batter_id` | INTEGER | | Batter (`player`) |
| `pitch_type` | VARCHAR | | Statcast pitch-type code (`pitch_type`); NULL if unclassified |
| `balls` | TINYINT | | Balls before the pitch |
| `strikes` | TINYINT | | Strikes before the pitch |
| `release_speed` | REAL | mph | Velocity out of the hand |
| `spin_rate` | REAL | rpm | Spin rate at release (Statcast `release_spin_rate`) |
| `spin_axis` | REAL | degrees | Spin axis in the x-z plane, 0 to 360; 180 is pure backspin, 0 pure topspin |
| `pfx_x` | REAL | ft | Horizontal movement versus a spinless pitch |
| `pfx_z` | REAL | ft | Vertical movement versus a spinless pitch |
| `break_x_arm` | REAL | ft | Horizontal break, positive toward the pitcher's arm side (Statcast `api_break_x_arm`) |
| `break_z_with_gravity` | REAL | ft | Vertical break including gravity (Statcast `api_break_z_with_gravity`) |
| `release_pos_x` | REAL | ft | Horizontal release point |
| `release_pos_y` | REAL | ft | Distance from home plate at release |
| `release_pos_z` | REAL | ft | Release height |
| `release_extension` | REAL | ft | How far in front of the rubber the ball is released |
| `arm_angle` | REAL | degrees | Arm angle at release; higher is more over the top |
| `vx0`, `vy0`, `vz0` | DOUBLE | ft/s | Velocity components at y = 50 ft |
| `ax`, `ay`, `az` | DOUBLE | ft/s² | Acceleration components (constant-acceleration trajectory fit) |
| `plate_x` | REAL | ft | Horizontal position at the plate, 0 at the plate's center |
| `plate_z` | REAL | ft | Height at the plate |
| `plate_reference` | VARCHAR | | Where `plate_x` and `plate_z` were measured: `front` of the plate through 2025, `middle` from 2026 |
| `sz_top` | REAL | ft | Top of the batter's strike zone |
| `sz_bot` | REAL | ft | Bottom of the batter's strike zone |
| `zone` | TINYINT | | Statcast zone: 1 to 9 inside the strike zone, 11 to 14 outside |
| `description` | VARCHAR | | Pitch result, such as `ball`, `called_strike`, `swinging_strike`, `foul`, `hit_into_play` |
| `result_type` | VARCHAR | | `B` ball, `S` strike, `X` in play |
| `launch_speed` | REAL | mph | Exit velocity; NULL unless the ball was hit |
| `launch_angle` | REAL | degrees | Launch angle; NULL unless the ball was hit |
| `delta_run_exp` | DOUBLE | runs | Change in run expectancy from before to after the pitch |

`plate_reference` exists because Statcast moved the plate-crossing point in
2026. Location comparisons across that change should use it.

### `swing`

One row per pitch with at least one bat-tracking value. Source: Statcast. Key:
`(game_pk, at_bat_number, pitch_number)`, the same as `pitch`.

| Column | Type | Units | Description |
| --- | --- | --- | --- |
| `game_pk`, `at_bat_number`, `pitch_number` | | | The pitch swung at (`pitch`) |
| `bat_speed` | REAL | mph | Speed of the bat's sweet spot at contact or closest approach |
| `swing_length` | REAL | ft | Distance the bat head travels from the start of the swing |
| `attack_angle` | REAL | degrees | Vertical angle of the bat's path at contact |
| `attack_direction` | REAL | degrees | Horizontal angle of the bat's path at contact |
| `swing_path_tilt` | REAL | degrees | Tilt of the swing plane |

### `player_transaction`

One row per player transaction. Source: Stats API `/transactions`. Key:
`transaction_id`.

| Column | Type | Description |
| --- | --- | --- |
| `transaction_id` | BIGINT | Stats API transaction id |
| `player_id` | INTEGER | Player (`player`) |
| `transaction_date` | DATE | Date the transaction was made |
| `effective_date` | DATE | Date it took effect |
| `type_code` | VARCHAR | Transaction type, such as `SC` (status change) or `CU` (recalled) |
| `type_desc` | VARCHAR | Type name, such as `Status Change` |
| `description` | VARCHAR | Free text, such as `Atlanta Braves placed RHP Joe Jiménez on the 15-day injured list. Knee surgery.` |

Injured-list placements are `SC` rows whose description reads "placed … on the
… injured list". See IL placements below for how they are parsed.

## Model output tables

### `model_run`

One row per fitted model. Key: `run_id`.

| Column | Type | Description |
| --- | --- | --- |
| `run_id` | UUID | Run id |
| `created_at` | TIMESTAMPTZ | When the run was saved, UTC |
| `model_name` | VARCHAR | Model version; currently `ou_day_effect_seasonal` |
| `role` | VARCHAR | `pitcher` or `hitter` |
| `metric` | VARCHAR | Modeled column, such as `release_speed` or `bat_speed` |
| `pitch_type` | VARCHAR | Pitch type for pitcher runs, such as `FF`; NULL for hitter runs |
| `seasons` | SMALLINT[] | Seasons fitted, such as `[2024, 2025, 2026]` |
| `player_count` | INTEGER | Players in the fit |
| `appearance_count` | INTEGER | Player-games in the fit |
| `log_likelihood` | DOUBLE | Maximized log-likelihood |
| `iterations` | INTEGER | L-BFGS iterations used |

The latest run for a given role, metric and pitch type is the one the CLI
reads.

### `model_parameter`

One row per estimated quantity per run. Key: `(run_id, parameter)`.

| Column | Type | Description |
| --- | --- | --- |
| `run_id` | UUID | Run (`model_run`) |
| `parameter` | VARCHAR | Name from the list below |
| `estimate` | DOUBLE | Maximum-likelihood estimate |
| `lower_95` | DOUBLE | Lower end of the 95% Wald interval |
| `upper_95` | DOUBLE | Upper end of the 95% Wald interval |

| `parameter` | Symbol | Units | Meaning |
| --- | --- | --- | --- |
| `mean_reversion` | κ | per day | Rate at which talent returns to the player's own level |
| `diffusion` | σ | metric units per √day | Talent volatility |
| `observation_sd` | r | metric units | Pitch-to-pitch noise within a game |
| `population_mean` | m | metric units | Mean long-run level across players |
| `population_sd` | s | metric units | Spread of long-run levels across players |
| `day_effect_sd` | τ | metric units | One-game shock shared by all of a game's pitches |
| `season_apr` … `season_oct` (no `season_jul`) | g | metric units | League seasonal shift on the first of each month, relative to July 1 |
| `half_life_days` | ln 2 / κ | days | Time for half of a talent deviation to decay |
| `stationary_sd` | σ / √(2κ) | metric units | Typical distance of talent from the player's own level |

Intervals for positive parameters are computed on the log scale and mapped
back, so they are asymmetric and stay positive.

### `latent_state`

One row per player per game per run: the model's estimate of talent after that
game. Key: `(run_id, player_id, game_date)`.

| Column | Type | Units | Description |
| --- | --- | --- | --- |
| `run_id` | UUID | | Run (`model_run`) |
| `player_id` | INTEGER | | Player (`player`) |
| `game_date` | DATE | | Game date |
| `sample_size` | INTEGER | count | Pitches (or swings) in the appearance, n |
| `appearance_mean` | DOUBLE | metric | Raw mean of the metric in the appearance |
| `seasonal_effect` | DOUBLE | metric | League seasonal shift g(day) on that date |
| `filtered_mean` | DOUBLE | metric | Real-time talent: estimate using games up to and including this one |
| `filtered_sd` | DOUBLE | metric | Its standard deviation |
| `smoothed_mean` | DOUBLE | metric | Hindsight talent: estimate using every game in the run |
| `smoothed_sd` | DOUBLE | metric | Its standard deviation |
| `filtered_gap_mean` | DOUBLE | metric | Real-time talent minus the player's own long-run level |
| `filtered_gap_sd` | DOUBLE | metric | Its standard deviation |
| `alert_z` | DOUBLE | sd | Calibrated alert score: the gap scaled to be N(0, 1) when nothing is wrong; negative means below the player's own level |

The talent columns are season-adjusted to their July 1 equivalent. Talent on
the day itself is `filtered_mean + seasonal_effect` (or the smoothed
equivalent). `seasonal_effect` is 0 for runs fitted before the seasonal curve
existed. The three gap columns are NULL for runs fitted before alert scores
existed, and such runs need refitting before `alerts` or `scan` can use them.

## Derived datasets

These are computed in Python from the warehouse when a command runs. They are
not stored as tables.

### Appearance statistics

`features/appearance.py`. One row per player per game date: the sufficient
statistics the model is fitted to.

| Column | Type | Description |
| --- | --- | --- |
| `player_id` | Int64 | Pitcher or hitter |
| `game_date` | Date | Game date |
| `sample_size` | Int64 | Pitches or swings, n |
| `mean` | Float64 | Mean of the metric |
| `within_ss` | Float64 | Within-game sum of squared deviations from `mean`, S |

- **Pitchers:** regular-season pitches of one pitch type with the metric
  tracked, for pitchers whose pitch type is among their three most-used over
  the seasons fitted. Metrics: `release_speed`, `spin_rate`,
  `release_extension`, `arm_angle`.
- **Hitters:** competitive swings only: bat speed of at least 50 mph and no
  bunts. Metrics: `bat_speed`, `swing_length`, `attack_angle`,
  `swing_path_tilt`.
- Players with fewer than 10 appearances are dropped.
- Rows are grouped by game date, so both games of a doubleheader form one
  appearance.

### IL placements

`features/injuries.py`. One row per injured-list placement.

| Column | Type | Description |
| --- | --- | --- |
| `transaction_id` | Int64 | Source transaction (`player_transaction`) |
| `player_id` | Int64 | Player |
| `il_start` | Date | Retroactive date if the text gives one, else the transaction date |
| `il_days` | Int16 | IL length: 7, 10, 15 or 60 |
| `injury` | String | Injury text after the placement sentence; empty if none given |
| `injury_group` | String | `arm`, `trunk`, `lower_body` or `other` |

Injury groups are assigned by keyword, and the first matching group wins:

| Group | Keywords |
| --- | --- |
| `arm` | elbow, shoulder, forearm, UCL, ulnar, flexor, biceps, triceps, rotator, labrum, Tommy John, arm, upper extremity, thoracic outlet, lat, latissimus, teres |
| `trunk` | oblique, back, lumbar, spine, spinal, rib(s), intercostal, abdominal, abdomen, side, core, chest, pectoral, neck, trapezius, thoracic |
| `lower_body` | hip, groin, adductor, hamstring, quad(riceps), thigh, knee, meniscus, ACL, patellar, calf, shin, ankle, Achilles, foot, heel, toe, plantar, leg, glute |
| `other` | everything else: illness, concussion, hand and finger injuries, unspecified |

### Relative usage

`features/usage.py`. One row per pitcher per game date: workload compared with
the pitcher's own history.

| Column | Type | Units | Description |
| --- | --- | --- | --- |
| `player_id` | Int64 | | Pitcher |
| `game_date` | Date | | Game date |
| `pitches` | Int64 | count | Regular-season pitches of every type that day |
| `rest_days` | Int64 | days | Days since the pitcher's previous game; NULL for his first |
| `pitches_z` | Float64 | sd | `pitches` against the pitcher's earlier games |
| `rest_z` | Float64 | sd | log(1 + rest, capped at 30 days) against the pitcher's earlier games |

Both z-scores use only earlier games. They are NULL until a pitcher has 5
earlier games, or when those games have essentially no spread (variance below
1e-6).

### Labeled appearances

`evaluation/injury_alerts.py`. `latent_state` rows for one run, each joined to
the pitcher's next IL placement.

| Column | Type | Description |
| --- | --- | --- |
| `il_start` | Date | Start of the next placement after this game; NULL if none |
| `injury_group`, `injury` | String | That placement's group and injury text |
| `days_to_il` | Int64 | Days from this game to `il_start` |
| `is_final` | Boolean | True if this is the last game before that placement |
| `score` | Float64 | `alert_z` turned so higher is more alarming: `-alert_z` for drops, `alert_z` for rises, its absolute value for either |

In the injury study, games within 30 days of a placement are positives,
split into the final game and earlier warning games. Healthy games are those
with no placement in the next 30 days (60 days in the event-study charts).