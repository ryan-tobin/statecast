from datetime import date

import numpy as np
import polars as pl

from statecast.evaluation.jump_report import (
    JUMP_PARAMETER_COUNT,
    OU_PARAMETER_COUNT,
    current_drops,
    likeliest_jumps,
    model_comparison,
    with_names,
)


def states() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": [2, 1, 1, 1, 2],
            "game_date": [
                date(2026, 9, 1),
                date(2026, 8, 20),
                date(2026, 8, 25),
                date(2026, 8, 30),
                date(2026, 9, 6),
            ],
            "talent_mean": [92.0, 95.0, 93.4, 93.3, 91.0],
            "jump_prob": [0.0, 0.0, 0.92, 0.05, 0.4],
            "recent_jump_prob": [0.0, 0.0, 0.95, 0.9, 0.45],
            "recent_drop_prob": [0.0, 0.0, 0.94, 0.88, 0.2],
        },
        schema_overrides={"player_id": pl.Int32},
    )


def names() -> pl.DataFrame:
    return pl.DataFrame({"player_id": [1, 2], "full_name": ["Zack Wheeler", "Nola"]})


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

    assert named["full_name"].to_list() == ["Zack Wheeler"] * 3 + ["Nola"] * 2
    assert named["player_id"].to_list() == [1, 1, 1, 2, 2]
    assert named.schema["player_id"] == pl.Int64
    assert named.filter(pl.col("player_id") == 1)["game_date"].is_sorted()


def test_likeliest_jumps_ranks_by_probability_and_measures_the_change() -> None:
    table = likeliest_jumps(with_names(states(), names()), top=2)

    assert table["full_name"].to_list() == ["Zack Wheeler", "Nola"]
    assert table["game_date"].to_list() == [date(2026, 8, 25), date(2026, 9, 6)]
    np.testing.assert_allclose(table["change"].to_numpy(), [-1.6, -1.0])
    np.testing.assert_allclose(table["talent_before"].to_numpy(), [95.0, 92.0])


def test_current_drops_use_each_players_last_game() -> None:
    table = current_drops(with_names(states(), names()), top=5)

    assert table["full_name"].to_list() == ["Zack Wheeler", "Nola"]
    assert table["last_game"].to_list() == [date(2026, 8, 30), date(2026, 9, 6)]
    assert table["recent_drop_prob"].to_list() == [0.88, 0.2]
