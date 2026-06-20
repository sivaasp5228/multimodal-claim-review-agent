"""
claim_extractor.py — Extract structured claim information from conversation text.

Parses multi-language conversations to identify what the user is claiming,
which part is affected, and what issue type they describe. This provides
context for the VLM call but does NOT make the final determination —
the VLM decides what's *actually visible* in the images.
"""

from __future__ import annotations

import re
from typing import List, Optional

from schemas import ClaimObject, ALL_PARTS


# ── Issue keywords mapped to issue_type values ──────────────────────────────

ISSUE_KEYWORDS = {
    "dent": ["dent", "dented", "denting", "indent", "dab gaya"],
    "scratch": ["scratch", "scratched", "scrape", "scraped", "mark", "scratching"],
    "crack": ["crack", "cracked", "cracking", "crack lines"],
    "glass_shatter": ["shatter", "shattered", "shattering", "glass broke"],
    "broken_part": [
        "broken", "broke", "break", "missing mirror", "snapped",
        "toot gaya", "toot", "toota", "damaged", "not sitting",
    ],
    "missing_part": [
        "missing", "came off", "fell off", "gone", "faltan",
        "not inside", "not there",
    ],
    "torn_packaging": [
        "torn", "ripped", "opened", "phati", "torn-open",
        "open jaisa", "seal phati",
    ],
    "crushed_packaging": [
        "crushed", "crush", "dab gaya", "badly crushed",
        "crushed in transit", "box crushed",
    ],
    "water_damage": ["water", "wet", "moisture", "soaked", "water damaged"],
    "stain": ["stain", "stained", "mark", "oily", "oil stain", "discolor"],
}

# ── Part keywords per object ────────────────────────────────────────────────

CAR_PART_KEYWORDS = {
    "front_bumper": ["front bumper", "front side", "front area", "parachoques delantero", "parachoques de adelante"],
    "rear_bumper": ["rear bumper", "back bumper", "rear side", "back side", "tapped from behind", "parachoques trasero", "parachoques de atras"],
    "door": ["door", "door panel"],
    "hood": ["hood", "bonnet", "top panel"],
    "windshield": ["windshield", "front glass", "wind shield", "windscreen", "parabrisas"],
    "side_mirror": ["side mirror", "mirror", "wing mirror"],
    "headlight": ["headlight", "head light", "front light"],
    "taillight": ["taillight", "tail light", "back light", "rear light"],
    "fender": ["fender", "wheel arch"],
    "quarter_panel": ["quarter panel", "quarter"],
    "body": ["body", "body panel", "car body", "side panel"],
}

LAPTOP_PART_KEYWORDS = {
    "screen": ["screen", "display", "monitor", "pantalla", "lcd"],
    "keyboard": ["keyboard", "keys", "key", "teclas", "keycap"],
    "trackpad": ["trackpad", "touchpad", "track pad", "palm-rest"],
    "hinge": ["hinge", "hinge area"],
    "lid": ["lid", "outer lid", "top cover", "tapa"],
    "corner": ["corner", "outer corner"],
    "port": ["port", "usb", "charging port"],
    "base": ["base", "bottom", "underside"],
    "body": ["body", "body panel", "outer body", "chassis", "casing"],
}

PACKAGE_PART_KEYWORDS = {
    "box": ["box", "delivery box", "shipping box", "cardboard box", "outer box"],
    "package_corner": ["corner", "package corner", "box corner"],
    "package_side": ["side", "package side", "surface"],
    "seal": ["seal", "tape", "flap", "seal area", "seal side"],
    "label": ["label", "shipping label", "sticker"],
    "contents": ["contents", "inside", "product inside", "item inside", "andar"],
    "item": ["item", "product", "inner item"],
}

PART_KEYWORDS = {
    "car": CAR_PART_KEYWORDS,
    "laptop": LAPTOP_PART_KEYWORDS,
    "package": PACKAGE_PART_KEYWORDS,
}


class ClaimExtraction:
    """Structured extraction from a claim conversation."""

    def __init__(
        self,
        claim_object: str,
        claimed_parts: List[str],
        claimed_issues: List[str],
        is_multi_part: bool = False,
        has_adversarial_text: bool = False,
        raw_claim: str = "",
    ):
        self.claim_object = claim_object
        self.claimed_parts = claimed_parts
        self.claimed_issues = claimed_issues
        self.is_multi_part = is_multi_part
        self.has_adversarial_text = has_adversarial_text
        self.raw_claim = raw_claim

    @property
    def primary_part(self) -> str:
        return self.claimed_parts[0] if self.claimed_parts else "unknown"

    @property
    def primary_issue(self) -> str:
        return self.claimed_issues[0] if self.claimed_issues else "unknown"


def extract_claim(user_claim: str, claim_object: str) -> ClaimExtraction:
    """
    Parse the conversation text to extract structured claim info.

    This uses keyword matching as a best-effort extraction. The VLM
    will make the authoritative determination of what's actually visible.
    """
    text_lower = user_claim.lower()

    # Detect adversarial instructions in text
    adversarial_patterns = [
        r"approve.*claim",
        r"skip.*review",
        r"mark.*supported",
        r"ignore.*previous",
        r"accept.*immediately",
        r"follow.*note",
        r"approve.*kar",
        r"claim.*approve",
    ]
    has_adversarial = any(
        re.search(pat, text_lower) for pat in adversarial_patterns
    )

    # Extract parts
    part_kw = PART_KEYWORDS.get(claim_object, {})
    found_parts: List[str] = []
    for part_name, keywords in part_kw.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                if part_name not in found_parts:
                    found_parts.append(part_name)
                break

    if not found_parts:
        found_parts = ["unknown"]

    # Extract issues
    found_issues: List[str] = []
    for issue_name, keywords in ISSUE_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                if issue_name not in found_issues:
                    found_issues.append(issue_name)
                break

    if not found_issues:
        found_issues = ["unknown"]

    is_multi_part = len(found_parts) > 1

    return ClaimExtraction(
        claim_object=claim_object,
        claimed_parts=found_parts,
        claimed_issues=found_issues,
        is_multi_part=is_multi_part,
        has_adversarial_text=has_adversarial,
        raw_claim=user_claim,
    )
