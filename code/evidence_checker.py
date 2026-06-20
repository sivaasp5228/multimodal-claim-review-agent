"""
evidence_checker.py — Evaluate whether images meet minimum evidence requirements.

Loads evidence_requirements.csv and checks whether the VLM analysis result
satisfies the applicable requirements for the given claim.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from schemas import EvidenceRequirement, VLMAnalysisResult


# ── Issue family mapping ────────────────────────────────────────────────────
# Maps issue_type values to the "applies_to" families in evidence_requirements.csv

ISSUE_TO_FAMILY = {
    # Car body panel family
    "dent": "dent or scratch",
    "scratch": "dent or scratch",
    # Car glass/light/mirror family
    "crack": "crack, broken, or missing part",
    "glass_shatter": "crack, broken, or missing part",
    "broken_part": "crack, broken, or missing part",
    "missing_part": "crack, broken, or missing part",
    # Package exterior family
    "torn_packaging": "crushed, torn, or seal damage",
    "crushed_packaging": "crushed, torn, or seal damage",
    # Package label/stain family
    "water_damage": "water, stain, or label damage",
    "stain": "water, stain, or label damage",
}

# Additional family lookups for specific part-based routing
PART_TO_FAMILY = {
    "screen": "screen, keyboard, or trackpad",
    "keyboard": "screen, keyboard, or trackpad",
    "trackpad": "screen, keyboard, or trackpad",
    "hinge": "hinge, lid, corner, body, or port",
    "lid": "hinge, lid, corner, body, or port",
    "corner": "hinge, lid, corner, body, or port",
    "body": "hinge, lid, corner, body, or port",
    "base": "hinge, lid, corner, body, or port",
    "port": "hinge, lid, corner, body, or port",
    "contents": "contents or inner item",
    "item": "contents or inner item",
    "label": "water, stain, or label damage",
    "seal": "crushed, torn, or seal damage",
    "package_corner": "crushed, torn, or seal damage",
    "package_side": "water, stain, or label damage",
    "box": "crushed, torn, or seal damage",
    "windshield": "crack, broken, or missing part",
    "headlight": "crack, broken, or missing part",
    "taillight": "crack, broken, or missing part",
    "side_mirror": "crack, broken, or missing part",
}


class EvidenceChecker:
    """Check whether submitted images meet minimum evidence requirements."""

    def __init__(self, requirements_path: str):
        self.requirements: List[EvidenceRequirement] = []
        self._load_requirements(requirements_path)

    def _load_requirements(self, path: str) -> None:
        """Load evidence_requirements.csv."""
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.requirements.append(EvidenceRequirement(**row))

    def get_applicable_requirements(
        self, claim_object: str, issue_type: str, object_part: str
    ) -> List[EvidenceRequirement]:
        """Find all requirements that apply to this claim."""
        applicable = []

        for req in self.requirements:
            # General requirements apply to all
            if req.claim_object == "all":
                applicable.append(req)
                continue

            # Object-specific requirements
            if req.claim_object != claim_object:
                continue

            # Match by issue family
            issue_family = ISSUE_TO_FAMILY.get(issue_type, "")
            part_family = PART_TO_FAMILY.get(object_part, "")

            if issue_family and issue_family in req.applies_to:
                applicable.append(req)
            elif part_family and part_family in req.applies_to:
                applicable.append(req)
            elif "vehicle identity" in req.applies_to and claim_object == "car":
                applicable.append(req)

        return applicable

    def check_evidence(
        self,
        vlm_result: VLMAnalysisResult,
        claim_object: str,
        claimed_issue: str,
        claimed_part: str,
    ) -> Tuple[bool, str]:
        """
        Evaluate whether the VLM analysis result satisfies evidence requirements.

        Returns:
            (evidence_met: bool, reason: str)
        """
        # Use VLM's own assessment as primary signal
        vlm_evidence_met = vlm_result.preliminary_verdict.evidence_sufficient
        vlm_reason = vlm_result.preliminary_verdict.evidence_reason

        # Get applicable requirements for rule-based validation
        reqs = self.get_applicable_requirements(
            claim_object, claimed_issue, claimed_part
        )

        # Rule-based checks that can OVERRIDE the VLM
        issues: List[str] = []

        # Check 1: Is the claimed object visible at all?
        if not vlm_result.claim_comparison.claimed_object_matches:
            issues.append("claimed object not visible in images")

        # Check 2: Cross-image consistency (different objects = unreliable for cars)
        if not vlm_result.cross_image.all_images_same_object and claim_object == "car":
            issues.append(
                "images appear to show different objects, "
                "so evidence reliability cannot be confirmed"
            )

        # Check 3: Is the claimed part visible?
        if not vlm_result.claim_comparison.claimed_part_visible:
            issues.append("claimed part not visible in submitted images")

        # Check 4: All images appear to be non-original
        all_non_original = all(
            img.appears_screenshot_or_stock or not img.appears_original
            for img in vlm_result.images
        )
        if all_non_original and vlm_result.images:
            issues.append("submitted images do not appear to be original photos")

        # If any rule-based issue found, evidence is NOT met
        if issues:
            return False, "; ".join(issues)

        # Otherwise trust VLM assessment
        if vlm_reason:
            return vlm_evidence_met, vlm_reason

        # Fallback
        if vlm_result.claim_comparison.claimed_part_visible:
            return True, (
                f"The {claimed_part} is visible and can be evaluated "
                f"from the submitted image(s)."
            )

        return False, "Insufficient evidence to evaluate the claim."
