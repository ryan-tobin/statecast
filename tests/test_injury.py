from datetime import date
from pathlib import Path

import duckdb
import polars as pl

from statecast.features.injuries import il_placements, parse_placements
from statecast.storage.warehouse import initialize_schema, upsert_table

DESCRIPTIONS = [
    "Atlanta Braves placed RHP Joe Jiménez on the 15-day injured list. Knee surgery.",
    "Miami Marlins placed RHP Edward Cabrera on the 15-day injured list "
    "retroactive to March 24, 2025. Right elbow inflammation.",
    "Seattle Mariners placed 3B Brendan Donovan on the 7-day injured list "
    "retroactive to August 3, 2026. Concussion.",
    "Los Angeles Dodgers placed RHP Edwin Díaz on the 15-day injured list. "
    "Right elbow loose bodies.",
    "Philadelphia Phillies placed RHP Pitcher A on the 60-day injured list. "
    "Right lat strain.",
    "Texas Rangers placed LHP Pitcher B on the 15-day injured list.",
]


def transactions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "transaction_id": list(range(1, len(DESCRIPTIONS) + 1)),
            "player_id": list(range(101, 101 + len(DESCRIPTIONS))),
            "transaction_date": [date(2025, 3, 27)] * len(DESCRIPTIONS),
            "description": DESCRIPTIONS,
        }
    )


def test_parse_placements_reads_length_start_and_injury() -> None:
    placements = parse_placements(transactions())

    assert placements["il_days"].to_list() == [15, 15, 7, 15, 60, 15]
    assert placements["il_start"].to_list() == [
        date(2025, 3, 27),
        date(2025, 3, 24),
        date(2026, 8, 3),
        date(2025, 3, 27),
        date(2025, 3, 27),
        date(2025, 3, 27),
    ]
    assert placements["injury"].to_list() == [
        "Knee surgery",
        "Right elbow inflammation",
        "Concussion",
        "Right elbow loose bodies",
        "Right lat strain",
        "",
    ]


def test_parse_placements_groups_arm_injuries() -> None:
    placements = parse_placements(transactions())

    assert placements["injury_group"].to_list() == [
        "lower_body",
        "arm",
        "other",
        "arm",
        "arm",
        "other",
    ]


def test_parse_placements_puts_each_injury_in_the_first_matching_group() -> None:
    injuries = [
        "Right thoracic outlet syndrome",
        "Low back strain",
        "Left oblique strain",
        "Right hamstring strain",
        "Left ankle sprain",
        "Illness",
        "Right middle finger blister",
    ]
    frame = pl.DataFrame(
        {
            "transaction_id": list(range(len(injuries))),
            "player_id": list(range(len(injuries))),
            "transaction_date": [date(2026, 5, 1)] * len(injuries),
            "description": [
                f"Team placed RHP X on the 15-day injured list. {injury}."
                for injury in injuries
            ],
        }
    )

    groups = parse_placements(frame)["injury_group"].to_list()

    assert groups == [
        "arm",
        "trunk",
        "trunk",
        "lower_body",
        "lower_body",
        "other",
        "other",
    ]


def test_il_placements_keeps_only_status_change_placements(output_dir: Path) -> None:
    with duckdb.connect(str(output_dir / "il.duckdb")) as connection:
        initialize_schema(connection)
        upsert_table(
            connection,
            "player_transaction",
            pl.DataFrame(
                {
                    "transaction_id": [1, 2, 3],
                    "player_id": [9, 9, 9],
                    "transaction_date": [date(2025, 6, 1)] * 3,
                    "effective_date": [date(2025, 6, 1)] * 3,
                    "type_code": ["SC", "SC", "CU"],
                    "type_desc": ["Status Change", "Status Change", "Recalled"],
                    "description": [
                        "Team placed RHP X on the 15-day injured list. "
                        "Right shoulder strain.",
                        "Team activated RHP X from the 15-day injured list.",
                        "Team recalled RHP X. RHP X placed on the 10-day injured list.",
                    ],
                }
            ),
        )

        placements = il_placements(connection)

    assert placements["transaction_id"].to_list() == [1]
    assert placements["injury_group"].to_list() == ["arm"]
