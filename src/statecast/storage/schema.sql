-- statecast warehouse schema (DuckDB). Normalized: each fact lives in one table.
-- Relationships are logical (pitch.pitcher_id -> player.player_id and so on);
-- DuckDB foreign keys would block the INSERT OR REPLACE upserts the loader uses.

CREATE TABLE IF NOT EXISTS player (
    player_id INTEGER PRIMARY KEY,
    full_name VARCHAR,
    birth_date DATE,
    throws VARCHAR,
    bats VARCHAR
);

CREATE TABLE IF NOT EXISTS team (
    team_id INTEGER PRIMARY KEY,
    name VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS venue (
    venue_id INTEGER PRIMARY KEY,
    name VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS game (
    game_pk INTEGER PRIMARY KEY,
    game_date DATE NOT NULL,
    season SMALLINT NOT NULL,
    game_type VARCHAR NOT NULL,
    venue_id INTEGER,
    home_team_id INTEGER,
    away_team_id INTEGER,
    first_pitch_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS pitch_type (
    code VARCHAR PRIMARY KEY,
    name VARCHAR
);

CREATE TABLE IF NOT EXISTS pitcher_appearance (
    game_pk INTEGER,
    pitcher_id INTEGER,
    days_since_prev_game SMALLINT,
    PRIMARY KEY (game_pk, pitcher_id)
);

CREATE TABLE IF NOT EXISTS plate_appearance (
    game_pk INTEGER,
    at_bat_number SMALLINT,
    inning TINYINT,
    inning_half VARCHAR,
    outs TINYINT,
    times_through_order TINYINT,
    event VARCHAR,
    PRIMARY KEY (game_pk, at_bat_number)
);

CREATE TABLE IF NOT EXISTS pitch (
    game_pk INTEGER,
    at_bat_number SMALLINT,
    pitch_number SMALLINT,
    pitcher_id INTEGER NOT NULL,
    batter_id INTEGER NOT NULL,
    pitch_type VARCHAR,
    balls TINYINT,
    strikes TINYINT,
    release_speed REAL,
    spin_rate REAL,
    spin_axis REAL,
    pfx_x REAL,
    pfx_z REAL,
    break_x_arm REAL,
    break_z_with_gravity REAL,
    release_pos_x REAL,
    release_pos_y REAL,
    release_pos_z REAL,
    release_extension REAL,
    arm_angle REAL,
    vx0 DOUBLE,
    vy0 DOUBLE,
    vz0 DOUBLE,
    ax DOUBLE,
    ay DOUBLE,
    az DOUBLE,
    plate_x REAL,
    plate_z REAL,
    plate_reference VARCHAR NOT NULL CHECK (plate_reference IN ('front', 'middle')),
    sz_top REAL,
    sz_bot REAL,
    zone TINYINT,
    description VARCHAR,
    result_type VARCHAR,
    launch_speed REAL,
    launch_angle REAL,
    delta_run_exp DOUBLE,
    PRIMARY KEY (game_pk, at_bat_number, pitch_number)
);

CREATE TABLE IF NOT EXISTS swing (
    game_pk INTEGER,
    at_bat_number SMALLINT,
    pitch_number SMALLINT,
    bat_speed REAL,
    swing_length REAL,
    attack_angle REAL,
    attack_direction REAL,
    swing_path_tilt REAL,
    PRIMARY KEY (game_pk, at_bat_number, pitch_number)
);

CREATE TABLE IF NOT EXISTS player_transaction (
    transaction_id BIGINT PRIMARY KEY,
    player_id INTEGER NOT NULL,
    transaction_date DATE,
    effective_date DATE,
    type_code VARCHAR,
    type_desc VARCHAR,
    description VARCHAR
);