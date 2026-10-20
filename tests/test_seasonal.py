from datetime import date

import jax.numpy as jnp
import numpy as np
import polars as pl

import statecast.models  # noqa: F401  (enables 64-bit floats)
from statecast.models.seasonal import (
    FREE_KNOT_COUNT,
    KNOT_DAYS,
    knot_values,
    season_day,
    seasonal_effect,
)

KNOTS = jnp.array([-0.3, -0.2, -0.1, 0.1, 0.05, 0.0])


def test_knot_values_anchor_july_at_zero() -> None:
    assert FREE_KNOT_COUNT == 6
    np.testing.assert_allclose(
        knot_values(KNOTS), [-0.3, -0.2, -0.1, 0.0, 0.1, 0.05, 0.0]
    )


def test_seasonal_effect_interpolates_between_knots() -> None:
    days = jnp.array([31.0, 46.0, 122.0, 137.5])

    np.testing.assert_allclose(
        seasonal_effect(KNOTS, days), [-0.3, -0.25, 0.0, 0.05], atol=1e-12
    )


def test_seasonal_effect_is_flat_outside_the_knots() -> None:
    early, late = seasonal_effect(KNOTS, jnp.array([10.0, 260.0]))

    assert float(early) == -0.3
    assert float(late) == 0.0
    assert float(KNOT_DAYS[0]) == 31.0


def test_season_day_lines_up_across_leap_years() -> None:
    frame = pl.DataFrame(
        {"game_date": [date(2024, 4, 1), date(2025, 4, 1), date(2024, 3, 1)]}
    )

    days = frame.select(season_day(pl.col("game_date")))["game_date"].to_list()

    assert days == [31, 31, 0]
