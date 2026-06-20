"""
main.py — Main pipeline orchestrator for Multi-Modal Evidence Review.

Usage:
    python code/main.py                              # Process claims.csv → output.csv
    python code/main.py --input dataset/sample_claims.csv --output sample_output.csv
    python code/main.py --help

Reads claims, loads context (user history, evidence requirements),
runs the VLM + Rule Engine pipeline, and writes output CSV.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

# Ensure code/ is on the path for local imports
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dotenv import load_dotenv

from schemas import ClaimInput, OutputRow, OUTPUT_COLUMNS
from claim_extractor import extract_claim
from vision_analyzer import VisionAnalyzer
from evidence_checker import EvidenceChecker
from risk_engine import RiskEngine
from decision_engine import DecisionEngine
from output_generator import write_output_csv

# ── Logging setup ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orchestrator")


# ── Paths ───────────────────────────────────────────────────────────────────

REPO_ROOT = CODE_DIR.parent
DATASET_DIR = REPO_ROOT / "dataset"
DEFAULT_INPUT = DATASET_DIR / "claims.csv"
DEFAULT_OUTPUT = REPO_ROOT / "output.csv"
USER_HISTORY_PATH = DATASET_DIR / "user_history.csv"
EVIDENCE_REQUIREMENTS_PATH = DATASET_DIR / "evidence_requirements.csv"


def load_claims(input_path: str) -> List[ClaimInput]:
    """Load claims from CSV file."""
    claims = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            claims.append(ClaimInput(**row))
    logger.info(f"Loaded {len(claims)} claims from {input_path}")
    return claims


def resolve_image_paths(
    image_paths_str: str, dataset_dir: Path
) -> List[str]:
    """
    Resolve image paths from CSV to absolute filesystem paths.

    CSV contains paths like: images/test/case_001/img_1.jpg
    These are relative to the dataset/ directory.
    """
    paths = []
    for p in image_paths_str.split(";"):
        p = p.strip()
        if not p:
            continue
        full_path = dataset_dir / p
        if full_path.exists():
            paths.append(str(full_path))
        else:
            logger.warning(f"Image not found: {full_path}")
    return paths


def process_claim(
    claim: ClaimInput,
    vision_analyzer: VisionAnalyzer,
    evidence_checker: EvidenceChecker,
    risk_engine: RiskEngine,
    decision_engine: DecisionEngine,
    dataset_dir: Path,
    index: int,
    total: int,
) -> OutputRow:
    """Process a single claim through the full pipeline."""

    logger.info(
        f"[{index + 1}/{total}] Processing claim for {claim.user_id} "
        f"({claim.claim_object}) ..."
    )

    # ── Step 1: Extract structured claim info ───────────────────────────
    claim_extraction = extract_claim(claim.user_claim, claim.claim_object)
    logger.info(
        f"  Claim: {claim_extraction.primary_part} / "
        f"{claim_extraction.primary_issue} "
        f"(multi-part={claim_extraction.is_multi_part})"
    )

    # ── Step 2: Resolve image paths ─────────────────────────────────────
    image_paths = resolve_image_paths(claim.image_paths, dataset_dir)
    image_ids = claim.image_ids
    logger.info(f"  Images: {len(image_paths)} ({', '.join(image_ids)})")

    # ── Step 3: VLM analysis ────────────────────────────────────────────
    vlm_result = vision_analyzer.analyze_claim(
        image_paths=image_paths,
        image_ids=image_ids,
        claim_object=claim.claim_object,
        user_claim=claim.user_claim,
        claimed_parts=claim_extraction.claimed_parts,
        claimed_issues=claim_extraction.claimed_issues,
    )
    logger.info(
        f"  VLM verdict: {vlm_result.preliminary_verdict.claim_status} "
        f"(issue={vlm_result.preliminary_verdict.primary_issue_type}, "
        f"severity={vlm_result.preliminary_verdict.severity})"
    )

    # ── Step 4: Check evidence requirements ─────────────────────────────
    evidence_met, evidence_reason = evidence_checker.check_evidence(
        vlm_result=vlm_result,
        claim_object=claim.claim_object,
        claimed_issue=claim_extraction.primary_issue,
        claimed_part=claim_extraction.primary_part,
    )
    logger.info(f"  Evidence met: {evidence_met}")

    # ── Step 5: Aggregate risk flags ────────────────────────────────────
    risk_flags = risk_engine.aggregate_risk_flags(
        vlm_result=vlm_result,
        user_id=claim.user_id,
        has_adversarial_text_in_claim=claim_extraction.has_adversarial_text,
    )
    logger.info(f"  Risk flags: {risk_flags}")

    # ── Step 6: Final decision ──────────────────────────────────────────
    decision = decision_engine.decide(
        claim=claim,
        claim_extraction=claim_extraction,
        vlm_result=vlm_result,
        evidence_met=evidence_met,
        evidence_reason=evidence_reason,
        risk_flags=risk_flags,
    )
    logger.info(
        f"  FINAL: status={decision.claim_status}, "
        f"issue={decision.issue_type}, "
        f"part={decision.object_part}, "
        f"severity={decision.severity}"
    )

    # ── Step 7: Assemble output row ─────────────────────────────────────
    output_row = OutputRow.from_input_and_decision(claim, decision)

    return output_row


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Modal Evidence Review — Claim Verification Pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Input claims CSV path (default: dataset/claims.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output CSV path (default: output.csv)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.1-flash-lite",
        help="Gemini model name (default: gemini-3.1-flash-lite)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Delay between API calls in seconds (default: 5.0)",
    )
    args = parser.parse_args()

    # Load environment variables from .env
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded .env from {env_path}")

    # Check API key
    if not os.environ.get("GOOGLE_API_KEY"):
        logger.error(
            "GOOGLE_API_KEY not set! Set it as an environment variable "
            "or add it to a .env file in the repo root."
        )
        sys.exit(1)

    start_time = time.time()

    # ── Initialize engines ──────────────────────────────────────────────
    logger.info("Initializing pipeline engines...")

    vision_analyzer = VisionAnalyzer(
        model=args.model,
        request_delay=args.delay,
    )
    evidence_checker = EvidenceChecker(str(EVIDENCE_REQUIREMENTS_PATH))
    risk_engine = RiskEngine(str(USER_HISTORY_PATH))
    decision_engine = DecisionEngine()

    # ── Load claims ─────────────────────────────────────────────────────
    claims = load_claims(args.input)

    # ── Process all claims ──────────────────────────────────────────────
    output_rows: List[OutputRow] = []
    total = len(claims)

    for i, claim in enumerate(claims):
        try:
            row = process_claim(
                claim=claim,
                vision_analyzer=vision_analyzer,
                evidence_checker=evidence_checker,
                risk_engine=risk_engine,
                decision_engine=decision_engine,
                dataset_dir=DATASET_DIR,
                index=i,
                total=total,
            )
            output_rows.append(row)
        except Exception as e:
            logger.error(f"Failed to process claim {i + 1}: {e}")
            # Create a safe fallback row
            from schemas import DecisionResult
            fallback = DecisionResult(
                evidence_standard_met=False,
                evidence_standard_met_reason="Processing error; manual review required.",
                risk_flags="manual_review_required",
                issue_type="unknown",
                object_part="unknown",
                claim_status="not_enough_information",
                claim_status_justification="Automated processing failed for this claim.",
                supporting_image_ids="none",
                valid_image=True,
                severity="unknown",
            )
            output_rows.append(OutputRow.from_input_and_decision(claim, fallback))

    # ── Write output ────────────────────────────────────────────────────
    write_output_csv(output_rows, args.output)

    # ── Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    statuses = {}
    for row in output_rows:
        statuses[row.claim_status] = statuses.get(row.claim_status, 0) + 1

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Claims processed: {total}")
    logger.info(f"  VLM calls: {vision_analyzer.call_count}")
    logger.info(f"  Runtime: {elapsed:.1f}s")
    logger.info(f"  Status distribution: {statuses}")
    logger.info(f"  Output: {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
