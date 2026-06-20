"""
schemas.py — Foundation types for the Multi-Modal Evidence Review pipeline.

All Pydantic models, enums, and validation logic live here. Every output
field is constrained to its allowed values so the CSV is always schema-compliant.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ── Allowed value enums ─────────────────────────────────────────────────────

class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_INFORMATION = "not_enough_information"


class IssueType(str, Enum):
    DENT = "dent"
    SCRATCH = "scratch"
    CRACK = "crack"
    GLASS_SHATTER = "glass_shatter"
    BROKEN_PART = "broken_part"
    MISSING_PART = "missing_part"
    TORN_PACKAGING = "torn_packaging"
    CRUSHED_PACKAGING = "crushed_packaging"
    WATER_DAMAGE = "water_damage"
    STAIN = "stain"
    NONE = "none"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class RiskFlag(str, Enum):
    NONE = "none"
    BLURRY_IMAGE = "blurry_image"
    CROPPED_OR_OBSTRUCTED = "cropped_or_obstructed"
    LOW_LIGHT_OR_GLARE = "low_light_or_glare"
    WRONG_ANGLE = "wrong_angle"
    WRONG_OBJECT = "wrong_object"
    WRONG_OBJECT_PART = "wrong_object_part"
    DAMAGE_NOT_VISIBLE = "damage_not_visible"
    CLAIM_MISMATCH = "claim_mismatch"
    POSSIBLE_MANIPULATION = "possible_manipulation"
    NON_ORIGINAL_IMAGE = "non_original_image"
    TEXT_INSTRUCTION_PRESENT = "text_instruction_present"
    USER_HISTORY_RISK = "user_history_risk"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"


class ClaimObject(str, Enum):
    CAR = "car"
    LAPTOP = "laptop"
    PACKAGE = "package"


# ── Object-specific parts ───────────────────────────────────────────────────

CAR_PARTS = [
    "front_bumper", "rear_bumper", "door", "hood", "windshield",
    "side_mirror", "headlight", "taillight", "fender", "quarter_panel",
    "body", "unknown",
]

LAPTOP_PARTS = [
    "screen", "keyboard", "trackpad", "hinge", "lid", "corner",
    "port", "base", "body", "unknown",
]

PACKAGE_PARTS = [
    "box", "package_corner", "package_side", "seal", "label",
    "contents", "item", "unknown",
]

ALL_PARTS = {
    "car": CAR_PARTS,
    "laptop": LAPTOP_PARTS,
    "package": PACKAGE_PARTS,
}


# ── Input models ────────────────────────────────────────────────────────────

class ClaimInput(BaseModel):
    """One row from claims.csv (input side only)."""
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str

    @property
    def image_path_list(self) -> List[str]:
        return [p.strip() for p in self.image_paths.split(";") if p.strip()]

    @property
    def image_ids(self) -> List[str]:
        """Extract image IDs (filename without extension) from paths."""
        import os
        return [
            os.path.splitext(os.path.basename(p))[0]
            for p in self.image_path_list
        ]


class UserHistoryRecord(BaseModel):
    """One row from user_history.csv."""
    user_id: str
    past_claim_count: int
    accept_claim: int
    manual_review_claim: int
    rejected_claim: int
    last_90_days_claim_count: int
    history_flags: str
    history_summary: str

    @property
    def flag_list(self) -> List[str]:
        return [f.strip() for f in self.history_flags.split(";") if f.strip()]

    @property
    def has_user_history_risk(self) -> bool:
        return "user_history_risk" in self.flag_list

    @property
    def has_manual_review_required(self) -> bool:
        return "manual_review_required" in self.flag_list

    @property
    def rejection_rate(self) -> float:
        if self.past_claim_count == 0:
            return 0.0
        return self.rejected_claim / self.past_claim_count


class EvidenceRequirement(BaseModel):
    """One row from evidence_requirements.csv."""
    requirement_id: str
    claim_object: str
    applies_to: str
    minimum_image_evidence: str


# ── VLM analysis models ────────────────────────────────────────────────────

class PerImageAnalysis(BaseModel):
    """VLM analysis result for a single image."""
    image_id: str = ""
    object_detected: str = "unknown"
    object_part_visible: str = "unknown"
    damage_detected: bool = False
    damage_type: str = "unknown"
    damage_severity: str = "unknown"
    quality_issues: List[str] = Field(default_factory=list)
    text_or_instructions_in_image: Optional[str] = None
    appears_original: bool = True
    appears_screenshot_or_stock: bool = False
    confidence_score: float = 0.5
    description: str = ""


class CrossImageAnalysis(BaseModel):
    """VLM cross-image consistency analysis."""
    all_images_same_object: bool = True
    object_identity_note: str = ""
    best_supporting_image_ids: List[str] = Field(default_factory=list)
    overall_object: str = "unknown"
    overall_part: str = "unknown"
    overall_damage_type: str = "unknown"
    overall_severity: str = "unknown"


class ClaimComparison(BaseModel):
    """VLM comparison of claim text vs visual evidence."""
    claimed_object_matches: bool = True
    claimed_part_visible: bool = False
    claimed_damage_visible: bool = False
    damage_matches_claim: bool = False
    claim_vs_visual_notes: str = ""


class PreliminaryVerdict(BaseModel):
    """VLM's preliminary verdict before rule-engine override."""
    claim_status: str = "not_enough_information"
    evidence_sufficient: bool = False
    evidence_reason: str = ""
    primary_issue_type: str = "unknown"
    primary_object_part: str = "unknown"
    severity: str = "unknown"
    justification: str = ""


class VLMAnalysisResult(BaseModel):
    """Complete VLM response for one claim."""
    images: List[PerImageAnalysis] = Field(default_factory=list)
    cross_image: CrossImageAnalysis = Field(default_factory=CrossImageAnalysis)
    claim_comparison: ClaimComparison = Field(default_factory=ClaimComparison)
    preliminary_verdict: PreliminaryVerdict = Field(default_factory=PreliminaryVerdict)


# ── Decision / output models ───────────────────────────────────────────────

class DecisionResult(BaseModel):
    """Final decision for one claim, ready for CSV output."""
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: str  # semicolon-separated
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str  # semicolon-separated or "none"
    valid_image: bool
    severity: str


class OutputRow(BaseModel):
    """Full output row matching the required CSV schema exactly."""
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    evidence_standard_met: str  # "true" / "false"
    evidence_standard_met_reason: str
    risk_flags: str
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str
    valid_image: str  # "true" / "false"
    severity: str

    @classmethod
    def from_input_and_decision(
        cls, claim: ClaimInput, decision: DecisionResult
    ) -> "OutputRow":
        return cls(
            user_id=claim.user_id,
            image_paths=claim.image_paths,
            user_claim=claim.user_claim,
            claim_object=claim.claim_object,
            evidence_standard_met=str(decision.evidence_standard_met).lower(),
            evidence_standard_met_reason=decision.evidence_standard_met_reason,
            risk_flags=decision.risk_flags,
            issue_type=decision.issue_type,
            object_part=decision.object_part,
            claim_status=decision.claim_status,
            claim_status_justification=decision.claim_status_justification,
            supporting_image_ids=decision.supporting_image_ids,
            valid_image=str(decision.valid_image).lower(),
            severity=decision.severity,
        )


# ── Column ordering for output CSV ──────────────────────────────────────────

OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]
