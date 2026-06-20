"""
risk_engine.py — Assess user history risk and propagate risk flags.

Loads user_history.csv and adds risk context to the claim evaluation.
NEVER overrides claim_status based on history alone — only adds flags.
"""

from __future__ import annotations

import csv
from typing import Dict, List, Optional, Set

from schemas import RiskFlag, UserHistoryRecord, VLMAnalysisResult


class RiskEngine:
    """Evaluate user history risk and aggregate all risk flags."""

    def __init__(self, history_path: str):
        self.history: Dict[str, UserHistoryRecord] = {}
        self._load_history(history_path)

    def _load_history(self, path: str) -> None:
        """Load user_history.csv into memory."""
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                record = UserHistoryRecord(
                    user_id=row["user_id"],
                    past_claim_count=int(row["past_claim_count"]),
                    accept_claim=int(row["accept_claim"]),
                    manual_review_claim=int(row["manual_review_claim"]),
                    rejected_claim=int(row["rejected_claim"]),
                    last_90_days_claim_count=int(row["last_90_days_claim_count"]),
                    history_flags=row["history_flags"],
                    history_summary=row["history_summary"],
                )
                self.history[record.user_id] = record

    def get_user_history(self, user_id: str) -> Optional[UserHistoryRecord]:
        """Look up a user's history record."""
        return self.history.get(user_id)

    def get_history_risk_flags(self, user_id: str) -> Set[str]:
        """
        Get risk flags from user history.

        Propagates flags from user_history.csv:
        - user_history_risk → adds to risk_flags
        - manual_review_required → adds to risk_flags
        """
        record = self.get_user_history(user_id)
        if record is None:
            return set()

        flags: Set[str] = set()

        # Direct flag propagation from history_flags column
        if record.has_user_history_risk:
            flags.add("user_history_risk")
        if record.has_manual_review_required:
            flags.add("manual_review_required")

        return flags

    def aggregate_risk_flags(
        self,
        vlm_result: VLMAnalysisResult,
        user_id: str,
        has_adversarial_text_in_claim: bool = False,
    ) -> Set[str]:
        """
        Aggregate all risk flags from VLM analysis and user history.

        Sources:
        1. Per-image quality issues from VLM
        2. Cross-image consistency issues
        3. Text/instruction detection in images
        4. Non-original image detection
        5. Claim comparison mismatches
        6. User history flags
        """
        flags: Set[str] = set()

        # ── 1. Per-image quality flags ──────────────────────────────────
        for img in vlm_result.images:
            for qi in img.quality_issues:
                qi_lower = qi.lower().replace(" ", "_")
                # Map VLM quality issue strings to valid risk flags
                flag_map = {
                    "blurry": "blurry_image",
                    "blurry_image": "blurry_image",
                    "blur": "blurry_image",
                    "cropped": "cropped_or_obstructed",
                    "cropped_or_obstructed": "cropped_or_obstructed",
                    "obstructed": "cropped_or_obstructed",
                    "low_light": "low_light_or_glare",
                    "low_light_or_glare": "low_light_or_glare",
                    "glare": "low_light_or_glare",
                    "dark": "low_light_or_glare",
                    "wrong_angle": "wrong_angle",
                }
                if qi_lower in flag_map:
                    flags.add(flag_map[qi_lower])

            # Text/instruction detection
            if img.text_or_instructions_in_image:
                flags.add("text_instruction_present")

            # Non-original image detection
            if img.appears_screenshot_or_stock or not img.appears_original:
                flags.add("non_original_image")

        # ── 2. Cross-image consistency ──────────────────────────────────
        if not vlm_result.cross_image.all_images_same_object:
            flags.add("wrong_object")

        # ── 3. Claim comparison mismatches ──────────────────────────────
        cc = vlm_result.claim_comparison
        if not cc.claimed_object_matches:
            flags.add("wrong_object")
            flags.add("claim_mismatch")

        if cc.claimed_part_visible and not cc.claimed_damage_visible:
            flags.add("damage_not_visible")

        if not cc.claimed_part_visible and cc.claimed_object_matches:
            flags.add("wrong_angle")

        if cc.claimed_damage_visible and not cc.damage_matches_claim:
            flags.add("claim_mismatch")

        # ── 4. User history ─────────────────────────────────────────────
        history_flags = self.get_history_risk_flags(user_id)
        flags.update(history_flags)

        # If user has history risk AND there are visual concerns, add manual review
        if "user_history_risk" in flags and (
            flags & {
                "claim_mismatch", "wrong_object", "damage_not_visible",
                "non_original_image", "text_instruction_present",
            }
        ):
            flags.add("manual_review_required")

        # ── 5. Adversarial text in claim ────────────────────────────────
        if has_adversarial_text_in_claim:
            flags.add("text_instruction_present")

        # Clean: remove "none" if other flags exist
        flags.discard("none")
        if not flags:
            flags.add("none")

        return flags

    @staticmethod
    def format_risk_flags(flags: Set[str]) -> str:
        """Format risk flags as semicolon-separated string."""
        if not flags or flags == {"none"}:
            return "none"
        clean = flags - {"none"}
        # Sort for determinism
        return ";".join(sorted(clean))
