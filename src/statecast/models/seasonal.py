"""League-wide seasonal curve: a shared shift by time of year.

The curve is piecewise linear with a knot on the first of each month from
April to October, measured in days since March 1 so leap years line up. It is
anchored at zero on July 1, so talent is expressed as its July 1 equivalent.
Dates before April 1 or after October 1 take the nearest knot's value.
"""

from typing import Final

import jax.numpy as jnp
import polars as pl
from jax import Array

KNOT_NAMES: Final = ("apr", "may", "jun", "jul", "aug", "sep", "oct")
KNOT_DAYS: Final = jnp.array([31.0, 61.0, 92.0, 122.0, 153.0, 184.0, 214.0])
ANCHOR_INDEX: Final = KNOT_NAMES.index("jul")
FREE_KNOT_NAMES: Final = tuple(
    name for index, name in enumerate(KNOT_NAMES) if index != ANCHOR_INDEX
)
FREE_KNOT_COUNT: Final = len(FREE_KNOT_NAMES)


def knot_values(free_knots: Array) -> Array:
    """Insert the anchored July 1 knot (zero) among the free knots.

    Args:
        free_knots: Curve values at every knot except July 1, shape ``(6,)``.

    Returns:
        Values at all seven knots, shape ``(7,)``.
    """
    return jnp.insert(free_knots, ANCHOR_INDEX, 0.0)


def seasonal_effect(free_knots: Array, season_day: Array) -> Array:
    """Evaluate the seasonal curve.

    Args:
        free_knots: Curve values at every knot except July 1, shape ``(6,)``.
        season_day: Days since March 1, any shape.

    Returns:
        The shift at each day, same shape as ``season_day``.

    Example:
        >>> import statecast.models
        >>> knots = jnp.array([-0.3, -0.2, -0.1, 0.1, 0.1, 0.0])
        >>> float(seasonal_effect(knots, jnp.asarray(46.0)))
        -0.25
    """
    return jnp.interp(season_day, KNOT_DAYS, knot_values(free_knots))


def season_day(game_date: pl.Expr) -> pl.Expr:
    """Days from March 1 of the same year to each date.

    Args:
        game_date: A date expression.

    Returns:
        An integer expression; April 1 is 31 in every year.
    """
    march_first = pl.date(game_date.dt.year(), 3, 1)
    return (game_date - march_first).dt.total_days()
