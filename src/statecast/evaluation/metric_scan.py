"""Does any pitch-tracking metric warn of arm injuries specifically?

Runs the injury-alert study on several metrics, each with one direction fixed
in advance, and compares arm placements with the lower-body placebo. A metric
that warns of arm injuries specifically has an arm-minus-lower-body AUC
difference above zero in the warning games. With several metrics and two
phases, p-values are Holm-adjusted so the chance of any false positive across
the whole scan stays at 5%.
"""

from collections.abc import Mapping
from typing import Final

import numpy as np
import polars as pl

from statecast.evaluation.injury_alerts import (
    PHASES,
    PLACEBO_GROUP,
    TARGET_GROUP,
    Direction,
)

METRIC_DIRECTIONS: Final[dict[str, Direction]] = {
    "release_speed": Direction.DROP,
    "spin_rate": Direction.DROP,
    "release_extension": Direction.DROP,
    "arm_angle": Direction.EITHER,
}
SCAN_SCHEMA: Final = {
    "metric": pl.String,
    "direction": pl.String,
    "phase": pl.String,
    "arm_auc": pl.Float64,
    "placebo_auc": pl.Float64,
    "difference": pl.Float64,
    "lower_95": pl.Float64,
    "upper_95": pl.Float64,
    "p_value": pl.Float64,
}


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjustment for a family of p-values.

    The ``i``-th smallest p-value is multiplied by ``m - i + 1``; adjusted
    values are made non-decreasing in that order and capped at 1.

    Args:
        p_values: Raw p-values, any order (NaNs stay NaN and are not counted).

    Returns:
        Adjusted p-values in the input order.

    Example:
        >>> holm_adjust(np.array([0.01, 0.04, 0.03])).round(3).tolist()
        [0.03, 0.06, 0.06]
    """
    adjusted = np.full(p_values.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p_values))
    order = finite[np.argsort(p_values[finite], kind="stable")]
    count = order.size
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def scan_rows(metric: str, direction: Direction, aucs: pl.DataFrame) -> pl.DataFrame:
    """Pull the target, placebo and difference rows for one metric.

    Args:
        metric: Metric column, such as ``spin_rate``.
        direction: Direction the metric was scored in.
        aucs: Output of ``auc_table`` for that metric.

    Returns:
        One row per phase with ``SCAN_SCHEMA`` columns.
    """
    difference_group = f"{TARGET_GROUP} - {PLACEBO_GROUP}"
    rows = []
    for phase in PHASES:
        by_group = {
            row["group"]: row
            for row in aucs.filter(pl.col("phase") == phase).iter_rows(named=True)
        }
        difference = by_group[difference_group]
        rows.append(
            (
                metric,
                direction.value,
                phase,
                by_group[TARGET_GROUP]["auc"],
                by_group[PLACEBO_GROUP]["auc"],
                difference["auc"],
                difference["lower_95"],
                difference["upper_95"],
                difference["p_value"],
            )
        )
    return pl.DataFrame(rows, orient="row", schema=SCAN_SCHEMA)


def metric_scan(results: Mapping[tuple[str, Direction], pl.DataFrame]) -> pl.DataFrame:
    """Combine every metric's AUC table and Holm-adjust across the scan.

    Args:
        results: ``auc_table`` output keyed by ``(metric, direction)``.

    Returns:
        ``SCAN_SCHEMA`` columns plus ``p_holm``, one row per metric and phase,
        in the order given.
    """
    if not results:
        return pl.DataFrame(schema={**SCAN_SCHEMA, "p_holm": pl.Float64})
    scan = pl.concat(
        [
            scan_rows(metric, direction, aucs)
            for (metric, direction), aucs in results.items()
        ]
    )
    return scan.with_columns(p_holm=holm_adjust(scan["p_value"].to_numpy()))
