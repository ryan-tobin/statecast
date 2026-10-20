"""Injured-list placements parsed from Stats API transaction descriptions.

Placements read like ``Atlanta Braves placed RHP Joe Jimenez on the 15-day
injured list retroactive to March 22, 2025. Right elbow inflammation.`` The IL
stint starts on the retroactive date when there is one, otherwise on the
transaction date.
"""

from typing import Final

import duckdb
import polars as pl

PLACEMENT_SQL: Final = """
SELECT transaction_id, player_id, transaction_date, description
FROM player_transaction
WHERE type_code = 'SC'
  AND transaction_date IS NOT NULL
  AND description ILIKE '%placed%injured list%'
ORDER BY player_id, transaction_date
"""

IL_DAYS_PATTERN: Final = r"on the (\d+)-day injured list"
RETROACTIVE_PATTERN: Final = r"retroactive to ([A-Z][a-z]+ \d{1,2}, \d{4})"
INJURY_PATTERN: Final = (
    r"injured list(?: retroactive to [A-Z][a-z]+ \d{1,2}, \d{4})?\.\s*(.*?)\.?\s*$"
)
ARM_PATTERN: Final = (
    r"(?i)\b(elbow|shoulder|forearm|ucl|ulnar|flexor|biceps|triceps|rotator"
    r"|labrum|tommy john|arm|upper extremity|thoracic outlet|lat|latissimus"
    r"|teres)\b"
)
TRUNK_PATTERN: Final = (
    r"(?i)\b(oblique|back|lumbar|spine|spinal|rib|ribs|intercostal|abdominal"
    r"|abdomen|side|core|chest|pectoral|neck|trapezius|thoracic)\b"
)
LOWER_BODY_PATTERN: Final = (
    r"(?i)\b(hip|groin|adductor|hamstring|quad|quadriceps|thigh|knee|meniscus"
    r"|acl|patellar|calf|shin|ankle|achilles|foot|heel|toe|plantar|leg|glute)\b"
)


def parse_placements(transactions: pl.DataFrame) -> pl.DataFrame:
    """Extract IL length, start date and injury from placement descriptions.

    Args:
        transactions: Columns ``transaction_id``, ``player_id``,
            ``transaction_date`` and ``description``.

    Returns:
        One row per placement: ``transaction_id``, ``player_id``,
        ``il_start``, ``il_days``, ``injury`` (empty when not given) and
        ``injury_group``: ``arm`` (throwing arm and shoulder), ``trunk``
        (back, oblique, ribs), ``lower_body`` (hips down) or ``other``
        (illness, concussion, hand, unspecified). The first matching group in
        that order wins, so "thoracic outlet" is an arm injury.

    Example:
        >>> from datetime import date
        >>> frame = pl.DataFrame({
        ...     "transaction_id": [1], "player_id": [7],
        ...     "transaction_date": [date(2025, 3, 27)],
        ...     "description": ["Miami Marlins placed RHP A on the 15-day injured "
        ...                     "list retroactive to March 24, 2025. Right elbow "
        ...                     "inflammation."]})
        >>> parse_placements(frame).select("il_start", "il_days").row(0)
        (datetime.date(2025, 3, 24), 15)
    """
    description = pl.col("description")
    injury = description.str.extract(INJURY_PATTERN, 1).fill_null("")
    return transactions.select(
        "transaction_id",
        "player_id",
        il_start=pl.coalesce(
            description.str.extract(RETROACTIVE_PATTERN, 1).str.to_date(
                "%B %d, %Y", strict=False
            ),
            pl.col("transaction_date"),
        ),
        il_days=description.str.extract(IL_DAYS_PATTERN, 1).cast(pl.Int16),
        injury=injury,
        injury_group=pl.when(injury.str.contains(ARM_PATTERN))
        .then(pl.lit("arm"))
        .when(injury.str.contains(TRUNK_PATTERN))
        .then(pl.lit("trunk"))
        .when(injury.str.contains(LOWER_BODY_PATTERN))
        .then(pl.lit("lower_body"))
        .otherwise(pl.lit("other")),
    )


def il_placements(connection: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Every injured-list placement in the warehouse.

    Args:
        connection: Warehouse connection.

    Returns:
        Output of ``parse_placements``, ordered by player and start date.
    """
    transactions = connection.execute(PLACEMENT_SQL).pl()
    return parse_placements(transactions).sort("player_id", "il_start")
