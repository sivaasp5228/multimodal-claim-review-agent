"""
output_generator.py — Generate the final output.csv with exact schema compliance.

Validates every row with Pydantic before writing, ensuring the output
matches the required column order and value constraints.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List

import pandas as pd

from schemas import OutputRow, OUTPUT_COLUMNS

logger = logging.getLogger(__name__)


def write_output_csv(rows: List[OutputRow], output_path: str) -> None:
    """
    Write output rows to CSV with exact schema compliance.

    Args:
        rows: List of validated OutputRow objects
        output_path: Path to write the CSV file
    """
    # Convert to list of dicts in correct column order
    records = []
    for row in rows:
        record = row.model_dump()
        # Ensure column order matches exactly
        ordered = {col: record[col] for col in OUTPUT_COLUMNS}
        records.append(ordered)

    # Write CSV
    df = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)

    logger.info(f"Wrote {len(rows)} rows to {output_path}")

    # Validate output
    _validate_output(output_path, len(rows))


def _validate_output(output_path: str, expected_rows: int) -> None:
    """Post-write validation of the output CSV."""
    df = pd.read_csv(output_path)

    # Check column count and names
    if list(df.columns) != OUTPUT_COLUMNS:
        logger.error(
            f"Column mismatch! Expected {OUTPUT_COLUMNS}, "
            f"got {list(df.columns)}"
        )
        raise ValueError("Output CSV column mismatch")

    # Check row count
    if len(df) != expected_rows:
        logger.error(
            f"Row count mismatch! Expected {expected_rows}, "
            f"got {len(df)}"
        )
        raise ValueError("Output CSV row count mismatch")

    # Check for required values
    valid_statuses = {"supported", "contradicted", "not_enough_information"}
    for idx, row in df.iterrows():
        status = row["claim_status"]
        if status not in valid_statuses:
            logger.error(f"Row {idx}: invalid claim_status '{status}'")

        severity = row["severity"]
        valid_severities = {"none", "low", "medium", "high", "unknown"}
        if severity not in valid_severities:
            logger.error(f"Row {idx}: invalid severity '{severity}'")

        evidence = str(row["evidence_standard_met"]).lower()
        if evidence not in ("true", "false"):
            logger.error(
                f"Row {idx}: invalid evidence_standard_met '{evidence}'"
            )

    logger.info("Output CSV validation passed")
