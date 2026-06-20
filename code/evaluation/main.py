"""
evaluation/main.py — Entry point for the evaluation workflow.

Usage:
    python code/evaluation/main.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure code/ is on the path
CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    from evaluation.evaluate import run_evaluation

    results = run_evaluation()

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for field in [
        "claim_status", "issue_type", "object_part",
        "severity", "evidence_standard_met", "valid_image",
    ]:
        data = results.get(field, {})
        acc = data.get("accuracy", 0)
        correct = data.get("correct", 0)
        total = data.get("total", 0)
        status = "[PASS]" if acc >= 0.8 else "[FAIL]"
        print(f"  {status} {field}: {acc:.1%} ({correct}/{total})")

    for field in ["risk_flags", "supporting_image_ids"]:
        data = results.get(field, {})
        f1 = data.get("f1", 0)
        status = "[PASS]" if f1 >= 0.7 else "[FAIL]"
        print(f"  {status} {field} F1: {f1:.1%}")

    print("=" * 60)
    print("Evaluation report: code/evaluation/evaluation_report.md")
    print("Sample predictions: sample_output.csv")


if __name__ == "__main__":
    main()
