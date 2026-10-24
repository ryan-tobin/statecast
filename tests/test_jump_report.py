from datetime import date

import numpy as np
import polars as pl

from statecast.evaluation.jump_report import (
    ACTIVE_DAYS,
    JUMP_PARAMETER_COUNT,
    OU_PARAMETER_COUNT,
    current_drops,
    model_comparison,
    strongest_jumps,
    with_names,
)


def states() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": [2, 1, 1, 1, 2, 3, 3],
            "game_date": [
                date(2026, 9, 1),
                date(2024, 8, 20),
                date(2026, 8, 25),
                date(2026, 8, 30),
                date(2026, 9, 6),
                date(2024, 5, 1),
                date(2024, 5, 6),
            ],
            "talent_mean": [92.0, 95.0, 95.1, 93.3, 91.0, 94.0, 90.5],
            "prior_jump_prob": [0.0, 0.0, 0.99, 0.01, 0.01, 0.0, 0.01],
            "jump_prob": [0.0, 0.0, 0.995, 0.6, 0.4, 0.0, 0.99],
            "jump_evidence": [0.0, 0.0, 0.7, 5.0, 4.2, 0.0, 9.1],
            "recent_jump_prob": [0.0, 0.0, 0.995, 0.9, 0.45, 0.0, 0.99],
            "recent_drop_prob": [0.0, 0.0, 0.2, 0.88, 0.2, 0.0, 0.99],
        },
        schema_overrides={"player_id": pl.Int32},
    )


def names() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": [1, 2, 3],
            "full_name": ["Zack Wheeler", "Aaron Nola", "Retired Pitcher"],
        }
    )


def test_model_comparison_reports_gain_and_aic() -> None:
    table = model_comparison(-1000.0, -950.0)

    rows = {row["model"]: row for row in table.iter_rows(named=True)}
    assert rows["OU"]["parameters"] == OU_PARAMETER_COUNT == 12
    assert rows["OU + jumps"]["parameters"] == JUMP_PARAMETER_COUNT == 15
    assert rows["OU + jumps"]["log_likelihood_gain"] == 50.0
    assert rows["OU"]["aic"] == 2 * 12 + 2000.0
    assert rows["OU + jumps"]["delta_aic"] == 0.0
    assert rows["OU"]["delta_aic"] == 94.0


def test_model_comparison_prefers_ou_when_jumps_add_too_little() -> None:
    table = model_comparison(-1000.0, -999.0)

    assert table.filter(pl.col("delta_aic") == 0.0)["model"].to_list() == ["OU"]


def test_with_names_joins_and_sorts() -> None:
    named = with_names(states(), names())

    assert named["player_id"].to_list() == [1, 1, 1, 2, 2, 3, 3]
    assert named["full_name"].to_list()[:3] == ["Zack Wheeler"] * 3
    assert named.schema["player_id"] == pl.Int64
    assert named.filter(pl.col("player_id") == 1)["game_date"].is_sorted()


def test_strongest_jumps_rank_by_evidence_not_by_posterior() -> None:
    table = strongest_jumps(with_names(states(), names()), top=3)

    assert table["full_name"].to_list() == [
        "Retired Pitcher",
        "Zack Wheeler",
        "Aaron Nola",
    ]
    assert table["game_date"].to_list() == [
        date(2024, 5, 6),
        date(2026, 8, 30),
        date(2026, 9, 6),
    ]
    np.testing.assert_allclose(table["change"].to_numpy(), [-3.5, -1.8, -1.0])
    assert table["gap_days"].to_list() == [5, 5, 5]


def test_strongest_jumps_show_long_gaps_and_skip_runs_without_evidence() -> None:
    named = with_names(states(), names())
    returning = strongest_jumps(named, top=10).filter(
        pl.col("game_date") == date(2026, 8, 25)
    )
    unscored = named.with_columns(jump_evidence=pl.lit(None, pl.Float64))

    assert returning["gap_days"].to_list() == [735]
    assert returning["jump_prob"].to_list() == [0.995]
    assert strongest_jumps(unscored, top=10).is_empty()


def test_current_drops_keep_only_active_players() -> None:
    table = current_drops(with_names(states(), names()), top=5)

    assert ACTIVE_DAYS == 30
    assert table["full_name"].to_list() == ["Zack Wheeler", "Aaron Nola"]
    assert table["last_game"].to_list() == [date(2026, 8, 30), date(2026, 9, 6)]
    assert table["recent_drop_prob"].to_list() == [0.88, 0.2]
