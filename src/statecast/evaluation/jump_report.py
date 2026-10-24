"""Summaries of a jump-diffusion run: model comparison and notable jumps."""

from typing import Final

import polars as pl

from statecast.models.estimation import PARAMETER_NAMES
from statecast.models.jumps import JUMP_PARAMETER_NAMES

OU_PARAMETER_COUNT: Final = len(PARAMETER_NAMES)
JUMP_PARAMETER_COUNT: Final = OU_PARAMETER_COUNT + len(JUMP_PARAMETER_NAMES)
ACTIVE_DAYS: Final = 30


def model_comparison(
    ou_log_likelihood: float, jump_log_likelihood: float
) -> pl.DataFrame:
    """Compare the OU and jump-diffusion fits to the same data.

    AIC is ``2k - 2 log L``. A formal likelihood-ratio test does not apply:
    with no jumps the jump size parameters are undefined, so the usual
    chi-squared reference fails. AIC is reported instead.

    Args:
        ou_log_likelihood: Maximized OU log-likelihood.
        jump_log_likelihood: Maximized jump-diffusion log-likelihood.

    Returns:
        One row per model: ``model``, ``parameters``, ``log_likelihood``,
        ``log_likelihood_gain`` (over the OU fit), ``aic`` and ``delta_aic``
        (over the better model).

    Example:
        >>> model_comparison(-1000.0, -950.0)["delta_aic"].to_list()
        [94.0, 0.0]
    """
    table = pl.DataFrame(
        {
            "model": ["OU", "OU + jumps"],
            "parameters": [OU_PARAMETER_COUNT, JUMP_PARAMETER_COUNT],
            "log_likelihood": [ou_log_likelihood, jump_log_likelihood],
        }
    ).with_columns(
        log_likelihood_gain=pl.col("log_likelihood") - ou_log_likelihood,
        aic=2.0 * pl.col("parameters") - 2.0 * pl.col("log_likelihood"),
    )
    return table.with_columns(delta_aic=pl.col("aic") - pl.col("aic").min())


def with_names(states: pl.DataFrame, names: pl.DataFrame) -> pl.DataFrame:
    """Attach full names to per-game jump output.

    Args:
        states: Output of ``jump_trajectory_frame``.
        names: ``player_id`` and ``full_name``.

    Returns:
        ``states`` with ``full_name``, sorted by player and date.
    """
    return (
        states.with_columns(pl.col("player_id").cast(pl.Int64))
        .join(names.with_columns(pl.col("player_id").cast(pl.Int64)), on="player_id")
        .sort("player_id", "game_date")
    )


def strongest_jumps(states: pl.DataFrame, top: int) -> pl.DataFrame:
    """The games whose data most strongly point to a jump in talent.

    Ranked by ``jump_evidence`` (the log Bayes factor), not by ``jump_prob``:
    the posterior also carries the prior chance of a jump, which approaches 1
    after a long absence whatever the game showed. ``gap_days`` and
    ``prior_jump_prob`` are shown so the two can be compared.

    Args:
        states: Named per-game output from ``with_names``.
        top: Rows to keep.

    Returns:
        ``full_name``, ``game_date``, ``gap_days``, ``prior_jump_prob``,
        ``jump_prob``, ``jump_evidence``, real-time talent before and after
        the game, and the ``change``, strongest evidence first. A player's
        first game never has a jump and is left out.
    """
    return (
        states.with_columns(
            talent_before=pl.col("talent_mean").shift(1).over("player_id"),
            gap_days=pl.col("game_date").diff().over("player_id").dt.total_days(),
        )
        .filter(
            pl.col("talent_before").is_not_null()
            & pl.col("jump_evidence").is_not_null()
        )
        .sort("jump_evidence", descending=True)
        .head(top)
        .select(
            "full_name",
            "game_date",
            "gap_days",
            "prior_jump_prob",
            "jump_prob",
            "jump_evidence",
            "talent_before",
            pl.col("talent_mean").alias("talent_after"),
            change=pl.col("talent_mean") - pl.col("talent_before"),
        )
    )


def current_drops(states: pl.DataFrame, top: int) -> pl.DataFrame:
    """Active players whose latest game most likely follows a downward jump.

    Only players who pitched (or batted) in the last ``ACTIVE_DAYS`` days of
    the data count, so a player whose final game was seasons ago does not
    crowd out current cases.

    Args:
        states: Named per-game output from ``with_names``.
        top: Rows to keep.

    Returns:
        ``full_name``, ``last_game``, ``talent_mean``, ``recent_jump_prob``
        and ``recent_drop_prob`` at each active player's last game, highest
        drop probability first.
    """
    cutoff = pl.col("game_date").max() - pl.duration(days=ACTIVE_DAYS)
    return (
        states.group_by("player_id", maintain_order=True)
        .last()
        .filter(pl.col("game_date") >= cutoff)
        .sort("recent_drop_prob", descending=True)
        .head(top)
        .select(
            "full_name",
            pl.col("game_date").alias("last_game"),
            "talent_mean",
            "recent_jump_prob",
            "recent_drop_prob",
        )
    )
