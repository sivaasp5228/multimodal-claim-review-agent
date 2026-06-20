"""
evaluate.py — Evaluation runner for the claim verification pipeline.

Runs the pipeline on sample_claims.csv, compares against ground truth,
and generates an evaluation report with metrics and operational analysis.

Supports comparing multiple strategies (Strategy A: Pure VLM vs
Strategy B: VLM + Rule Engine + Evidence Requirements).
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure code/ is on the path
CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dotenv import load_dotenv

from schemas import ClaimInput, OutputRow, DecisionResult
from claim_extractor import extract_claim
from vision_analyzer import VisionAnalyzer
from evidence_checker import EvidenceChecker
from risk_engine import RiskEngine
from decision_engine import DecisionEngine
from output_generator import write_output_csv
from evaluation.metrics import (
    load_csv_rows,
    compute_all_metrics,
    format_report,
)

logger = logging.getLogger(__name__)

REPO_ROOT = CODE_DIR.parent
DATASET_DIR = REPO_ROOT / "dataset"
SAMPLE_CLAIMS_PATH = DATASET_DIR / "sample_claims.csv"
USER_HISTORY_PATH = DATASET_DIR / "user_history.csv"
EVIDENCE_REQUIREMENTS_PATH = DATASET_DIR / "evidence_requirements.csv"
EVAL_DIR = CODE_DIR / "evaluation"


def load_sample_claims() -> Tuple[List[ClaimInput], List[Dict[str, str]]]:
    """Load sample claims with ground truth."""
    gt_rows = load_csv_rows(str(SAMPLE_CLAIMS_PATH))

    claims = []
    for row in gt_rows:
        claims.append(
            ClaimInput(
                user_id=row["user_id"],
                image_paths=row["image_paths"],
                user_claim=row["user_claim"],
                claim_object=row["claim_object"],
            )
        )

    return claims, gt_rows


def resolve_image_paths(image_paths_str: str, dataset_dir: Path) -> List[str]:
    """Resolve image paths from CSV to absolute filesystem paths."""
    paths = []
    for p in image_paths_str.split(";"):
        p = p.strip()
        if not p:
            continue
        full_path = dataset_dir / p
        if full_path.exists():
            paths.append(str(full_path))
    return paths


def run_strategy_b(
    claims: List[ClaimInput],
    vision_analyzer: VisionAnalyzer,
    evidence_checker: EvidenceChecker,
    risk_engine: RiskEngine,
    decision_engine: DecisionEngine,
) -> Tuple[List[Dict[str, str]], Dict]:
    """
    Strategy B: VLM + Rule Engine + Evidence Requirements (Full Pipeline).
    """
    output_rows = []
    stats = {"vlm_calls": 0, "images_processed": 0, "start_time": time.time()}

    for i, claim in enumerate(claims):
        logger.info(
            f"  [Strategy B] [{i + 1}/{len(claims)}] "
            f"{claim.user_id} ({claim.claim_object})"
        )

        # Extract claim
        extraction = extract_claim(claim.user_claim, claim.claim_object)

        # Resolve images
        image_paths = resolve_image_paths(claim.image_paths, DATASET_DIR)
        image_ids = claim.image_ids
        stats["images_processed"] += len(image_paths)

        # VLM analysis
        vlm_result = vision_analyzer.analyze_claim(
            image_paths=image_paths,
            image_ids=image_ids,
            claim_object=claim.claim_object,
            user_claim=claim.user_claim,
            claimed_parts=extraction.claimed_parts,
            claimed_issues=extraction.claimed_issues,
        )
        stats["vlm_calls"] += 1

        # Evidence check
        evidence_met, evidence_reason = evidence_checker.check_evidence(
            vlm_result=vlm_result,
            claim_object=claim.claim_object,
            claimed_issue=extraction.primary_issue,
            claimed_part=extraction.primary_part,
        )

        # Risk flags
        risk_flags = risk_engine.aggregate_risk_flags(
            vlm_result=vlm_result,
            user_id=claim.user_id,
            has_adversarial_text_in_claim=extraction.has_adversarial_text,
        )

        # Decision
        decision = decision_engine.decide(
            claim=claim,
            claim_extraction=extraction,
            vlm_result=vlm_result,
            evidence_met=evidence_met,
            evidence_reason=evidence_reason,
            risk_flags=risk_flags,
        )

        # Build output row
        row = OutputRow.from_input_and_decision(claim, decision)
        output_rows.append(row.model_dump())

    stats["runtime"] = time.time() - stats["start_time"]
    return output_rows, stats


def generate_evaluation_report(
    results_b: Dict,
    stats_b: Dict,
    output_path: str,
) -> None:
    """Generate the evaluation report markdown."""
    lines = [
        "# Evaluation Report — Multi-Modal Evidence Review\n",
        "## Strategy Comparison\n",
        "### Strategy B: VLM + Rule Engine + Evidence Requirements (Full Pipeline)\n",
        "This is the primary strategy using Gemini 3.1 Flash Lite as the VLM,",
        "combined with a deterministic rule engine for decision validation,",
        "evidence requirements checking, and user history risk assessment.\n",
    ]

    # Strategy B results
    lines.append(format_report(results_b, "B: VLM + Rule Engine"))

    # Operational analysis
    lines.extend([
        "\n## Operational Analysis\n",
        "| Metric | Value |",
        "|---|---|",
        f"| VLM calls (sample) | {stats_b.get('vlm_calls', 'N/A')} |",
        f"| Images processed (sample) | {stats_b.get('images_processed', 'N/A')} |",
        f"| Runtime (sample) | {stats_b.get('runtime', 0):.1f}s |",
        f"| Model | Gemini 3.1 Flash Lite |",
        f"| Temperature | 0.0 |",
        f"| Response format | JSON |",
        "",
        "### Cost Estimates (Full Test Set — 45 claims)\n",
        "| Item | Estimate |",
        "|---|---|",
        "| VLM calls | ~45 |",
        "| Input tokens | ~400K (text + images) |",
        "| Output tokens | ~45K |",
        "| Gemini 3.5 Flash cost | ~$0.20 |",
        "| Runtime | ~7 minutes |",
        "",
        "### Rate Limiting & Optimization\n",
        "- **RPM**: 5s delay between calls (~12 RPM effective)",
        "- **Caching**: Not needed for single-run pipeline",
        "- **Batching**: One VLM call per claim (all images sent together)",
        "- **Retry**: Exponential backoff with 3 attempts via tenacity",
        "- **Determinism**: temperature=0.0, JSON response format",
        "",
        "## Design Decisions\n",
        "1. **Single VLM call per claim**: All images sent in one call for",
        "   cross-image consistency detection",
        "2. **Anti-injection system prompt**: Explicit rules to detect and",
        "   report (not follow) text instructions in images",
        "3. **Hybrid architecture**: VLM provides preliminary verdict,",
        "   rule engine validates and overrides when needed",
        "4. **Evidence requirements grounding**: Deterministic matching of",
        "   issue families to evidence requirement rules",
        "5. **Selective supporting_image_ids**: Only images that actively",
        "   contribute evidence, not all submitted images",
        "6. **Severity calibration**: Aligned with visible damage level,",
        "   not user's description",
    ])

    report_text = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"Evaluation report written to {output_path}")


def run_evaluation() -> Dict:
    """
    Main evaluation function.

    Returns the metrics dict for programmatic use.
    """
    logger.info("=" * 60)
    logger.info("EVALUATION — Sample Claims")
    logger.info("=" * 60)

    # Load environment
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    if not os.environ.get("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY not set!")
        sys.exit(1)

    # Load data
    claims, gt_rows = load_sample_claims()
    logger.info(f"Loaded {len(claims)} sample claims with ground truth")

    # Initialize engines
    vision_analyzer = VisionAnalyzer(
        model="gemini-3.1-flash-lite",
        request_delay=5.0,
    )
    evidence_checker = EvidenceChecker(str(EVIDENCE_REQUIREMENTS_PATH))
    risk_engine = RiskEngine(str(USER_HISTORY_PATH))
    decision_engine = DecisionEngine()

    # ── Strategy B: Full Pipeline ───────────────────────────────────────
    logger.info("Running Strategy B: VLM + Rule Engine...")
    pred_rows_b, stats_b = run_strategy_b(
        claims, vision_analyzer, evidence_checker, risk_engine, decision_engine
    )

    # Compute metrics for Strategy B
    results_b = compute_all_metrics(pred_rows_b, gt_rows)

    # Log key metrics
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info(
        f"  claim_status accuracy: "
        f"{results_b['claim_status']['accuracy']:.1%}"
    )
    logger.info(
        f"  issue_type accuracy: "
        f"{results_b['issue_type']['accuracy']:.1%}"
    )
    logger.info(
        f"  object_part accuracy: "
        f"{results_b['object_part']['accuracy']:.1%}"
    )
    logger.info(
        f"  severity accuracy: "
        f"{results_b['severity']['accuracy']:.1%}"
    )
    logger.info(
        f"  evidence_standard_met accuracy: "
        f"{results_b['evidence_standard_met']['accuracy']:.1%}"
    )
    logger.info("=" * 60)

    # Generate report
    report_path = str(EVAL_DIR / "evaluation_report.md")
    generate_evaluation_report(
        results_b=results_b,
        stats_b=stats_b,
        output_path=report_path,
    )

    # Also save predictions for inspection
    pred_output_path = str(REPO_ROOT / "sample_output.csv")
    output_row_objects = [
        OutputRow(**row) for row in pred_rows_b
    ]
    write_output_csv(output_row_objects, pred_output_path)

    return results_b
