import numpy as np
import polars as pl
import pytest

from statecast.evaluation.score_comparison import (
    ANY_GROUP,
    CONTRAST,
    STATISTICS,
    compare_scores,
    drop_evidence,
    drop_share,
)

PLAYERS = 200
GAMES_PER_PLAYER = 30


def labeled_games(seed: int) -> pl.DataFrame:
    """Healthy games plus arm and lower-body placements; ``sharp`` sees arm only."""
    rng = np.random.default_rng(seed)
    count = PLAYERS * GAMES_PER_PLAYER
    games = pl.DataFrame(
        {
            "player_id": np.repeat(np.arange(PLAYERS), GAMES_PER_PLAYER),
            "game": np.tile(np.arange(GAMES_PER_PLAYER), PLAYERS),
            "noise": rng.normal(size=count),
            "jitter": rng.normal(size=count),
        }
    )
    group = (
        pl.when(pl.col("player_id") % 4 == 0)
        .then(pl.lit("arm"))
        .when(pl.col("player_id") % 4 == 1)
        .then(pl.lit("lower_body"))
    )
    injured = group.is_not_null() & (pl.col("game") >= GAMES_PER_PLAYER - 5)
    return games.select(
        "player_id",
        "noise",
        days_to_il=pl.when(injured).then(5 * (GAMES_PER_PLAYER - pl.col("game"))),
        injury_group=pl.when(injured).then(group),
        is_final=injured & (pl.col("game") == GAMES_PER_PLAYER - 1),
        sharp=pl.col("jitter") + 1.5 * (injured & (group == "arm")).fill_null(False),
    )


def test_drop_evidence_ignores_a_long_absence() -> None:
    frame = pl.DataFrame(
        {
            "player_id": [1, 1, 2, 2],
            "prior_jump_prob": [0.0, 0.9999, 0.0, 0.007],
            "recent_drop_prob": [0.0, 0.5, 0.0, 0.5],
        }
    )

    evidence = frame.select(drop_evidence(3, 0.5))["drop_evidence"].to_numpy()

    np.testing.assert_allclose(evidence[1], 0.0, atol=1e-3)
    assert evidence[3] > 4.0


def test_drop_evidence_pools_the_window_prior_per_player() -> None:
    frame = pl.DataFrame(
        {
            "player_id": [1, 1, 1, 2],
            "prior_jump_prob": [0.1, 0.1, 0.1, 0.1],
            "recent_drop_prob": [0.5, 0.5, 0.5, 0.5],
        }
    )

    evidence = frame.select(drop_evidence(2, 1.0))["drop_evidence"].to_numpy()

    single = -np.log(0.1 / 0.9)
    double = -np.log(0.19 / 0.81)
    np.testing.assert_allclose(evidence, [single, double, double, single])


def test_drop_share_follows_the_jump_mean() -> None:
    assert drop_share(0.0, 1.0) == pytest.approx(0.5)
    assert drop_share(-1.0, 1.0) == pytest.approx(0.8413, abs=1e-4)
    assert drop_share(1.0, 1.0) == pytest.approx(0.1587, abs=1e-4)


def test_compare_scores_finds_the_arm_specific_score() -> None:
    labeled = labeled_games(seed=1)

    comparison = compare_scores(labeled, ["noise", "sharp"], 30, reps=500, seed=0)

    aucs = {
        (row["score"], row["statistic"], row["phase"]): row
        for row in comparison.aucs.iter_rows(named=True)
    }
    gains = {
        (row["statistic"], row["phase"]): row
        for row in comparison.gains.iter_rows(named=True)
    }
    assert aucs["sharp", "arm", "warning"]["estimate"] > 0.8
    assert aucs["sharp", "lower_body", "warning"]["estimate"] == pytest.approx(
        0.5, abs=0.06
    )
    assert aucs["sharp", CONTRAST, "warning"]["p_holm"] < 0.05
    assert aucs["noise", ANY_GROUP, "warning"]["estimate"] == pytest.approx(
        0.5, abs=0.06
    )
    assert gains["arm", "warning"]["lower_95"] > 0.0
    assert gains["arm", "warning"]["p_holm"] < 0.05


def test_compare_scores_layout_and_holm_scope() -> None:
    labeled = labeled_games(seed=2).with_columns(noise_copy=pl.col("noise"))

    comparison = compare_scores(
        labeled, ["noise", "sharp", "noise_copy"], 30, reps=50, seed=0
    )

    assert comparison.aucs.height == 3 * len(STATISTICS) * 2
    assert comparison.gains.height == 2 * len(STATISTICS) * 2
    contrast = comparison.aucs.filter(pl.col("statistic") == CONTRAST)
    others = comparison.aucs.filter(pl.col("statistic") != CONTRAST)
    assert contrast["p_holm"].null_count() == 0
    assert others["p_holm"].null_count() == others.height
    copy_gains = comparison.gains.filter(pl.col("score") == "noise_copy")
    np.testing.assert_allclose(copy_gains["estimate"].to_numpy(), 0.0)
    np.testing.assert_allclose(copy_gains["upper_95"].to_numpy(), 0.0)


def test_compare_scores_needs_a_baseline_and_a_rival() -> None:
    with pytest.raises(ValueError, match="baseline"):
        compare_scores(labeled_games(seed=3), ["noise"], 30)
