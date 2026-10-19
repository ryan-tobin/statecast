from datetime import date

import polars as pl

from conftest import StatcastFactory, schedule_game, schedule_payload
from statecast.ingest.statsapi import parse_schedule
from statecast.storage import normalize


def pitch(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "game_pk": 1,
        "game_date": date(2026, 6, 1),
        "at_bat_number": 1,
        "pitch_number": 1,
        "pitcher": 10,
        "batter": 20,
        "pitch_type": "FF",
        "pitch_name": "4-Seam Fastball",
        "inning": 1,
        "inning_topbot": "Top",
        "outs_when_up": 0,
        "n_thruorder_pitcher": 1,
        "pitcher_days_since_prev_game": 5,
        "release_spin_rate": 2400.0,
        "type": "S",
    }
    row.update(overrides)
    return row


def test_pitch_table_renames_columns_and_sets_plate_reference(
    make_statcast: StatcastFactory,
) -> None:
    statcast = make_statcast(
        [
            pitch(game_pk=1, game_date=date(2025, 9, 1)),
            pitch(game_pk=2, game_date=date(2026, 4, 1)),
        ]
    )

    pitches = normalize.pitch_table(statcast)

    assert pitches["plate_reference"].to_list() == ["front", "middle"]
    assert pitches["spin_rate"].to_list() == [2400.0, 2400.0]
    assert {"pitcher_id", "batter_id", "result_type"} <= set(pitches.columns)
    assert "pitch_name" not in pitches.columns


def test_swing_table_keeps_only_tracked_swings(make_statcast: StatcastFactory) -> None:
    statcast = make_statcast(
        [pitch(pitch_number=1), pitch(pitch_number=2, attack_angle=12.0)]
    )

    swings = normalize.swing_table(statcast)

    assert swings["pitch_number"].to_list() == [2]
    assert swings.columns == [*normalize.PITCH_KEY, *normalize.BAT_TRACKING_COLUMNS]


def test_plate_appearance_table_uses_first_situation_and_last_event(
    make_statcast: StatcastFactory,
) -> None:
    statcast = make_statcast(
        [
            pitch(pitch_number=2, outs_when_up=1, events="strikeout"),
            pitch(pitch_number=1, outs_when_up=0),
        ]
    )

    plate_appearances = normalize.plate_appearance_table(statcast)

    assert plate_appearances.rows() == [(1, 1, 1, "Top", 0, 1, "strikeout")]


def test_pitcher_appearance_table_one_row_per_pitcher_game(
    make_statcast: StatcastFactory,
) -> None:
    statcast = make_statcast([pitch(pitch_number=1), pitch(pitch_number=2)])

    appearances = normalize.pitcher_appearance_table(statcast)

    assert appearances.rows() == [(1, 10, 5)]


def test_pitch_type_table_lists_each_code_once(make_statcast: StatcastFactory) -> None:
    statcast = make_statcast(
        [
            pitch(pitch_number=1),
            pitch(pitch_number=2, pitch_type="SL", pitch_name="Slider"),
            pitch(pitch_number=3, pitch_type=None, pitch_name=None),
        ]
    )

    pitch_types = normalize.pitch_type_table(statcast)

    assert pitch_types.sort("code").rows() == [
        ("FF", "4-Seam Fastball"),
        ("SL", "Slider"),
    ]


def test_schedule_tables_split_games_teams_and_venues() -> None:
    schedule = parse_schedule(
        schedule_payload(
            [schedule_game(1, "2026-06-01"), schedule_game(2, "2026-06-02")]
        ),
        ("R",),
    )

    games = normalize.game_table(schedule)
    teams = normalize.team_table(schedule)
    venues = normalize.venue_table(schedule)

    assert games["game_pk"].to_list() == [1, 2]
    assert sorted(teams["team_id"].to_list()) == [121, 143]
    assert venues.rows() == [(2681, "Citizens Bank Park")]
    assert isinstance(games.schema["first_pitch_at"], pl.Datetime)
