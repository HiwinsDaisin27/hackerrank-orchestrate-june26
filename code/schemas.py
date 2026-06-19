"""Schema definitions and allowed-value enums for output validation."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


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


CAR_PARTS = {
    "front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
    "headlight", "taillight", "fender", "quarter_panel", "body", "unknown",
}
LAPTOP_PARTS = {
    "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port",
    "base", "body", "unknown",
}
PACKAGE_PARTS = {
    "box", "package_corner", "package_side", "seal", "label", "contents",
    "item", "unknown",
}


def parts_for_object(claim_object: str) -> set:
    mapping = {"car": CAR_PARTS, "laptop": LAPTOP_PARTS, "package": PACKAGE_PARTS}
    return mapping.get(claim_object, set())


OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
    "issue_type", "object_part", "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity",
]


class ClaimExtraction(BaseModel):
    claimed_issue_type: str
    claimed_object_part: str
    claim_summary: str = ""


class ImageAnalysis(BaseModel):
    image_id: str
    is_valid: bool
    is_corrupt: bool = False
    quality_flags: List[str] = Field(default_factory=list)
    detected_object: str = "unknown"
    object_matches_claim_type: bool = False
    issue_type: str = "unknown"
    object_part: str = "unknown"
    severity: str = "unknown"
    damage_visible: bool = False
    text_instruction_in_image: bool = False
    possible_manipulation: bool = False
    non_original_image: bool = False
    notes: str = ""


class VisionAnalysisResult(BaseModel):
    images: List[ImageAnalysis] = Field(default_factory=list)


class ClaimOutput(BaseModel):
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: List[str]
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: List[str]
    valid_image: bool
    severity: str

    def to_csv_row(self) -> dict:
        flags = self.risk_flags
        flag_str = "none" if not flags or flags == ["none"] else ";".join(flags)
        sup = self.supporting_image_ids
        sup_str = "none" if not sup else ";".join(sup)
        return {
            "user_id": self.user_id,
            "image_paths": self.image_paths,
            "user_claim": self.user_claim,
            "claim_object": self.claim_object,
            "evidence_standard_met": "true" if self.evidence_standard_met else "false",
            "evidence_standard_met_reason": self.evidence_standard_met_reason,
            "risk_flags": flag_str,
            "issue_type": self.issue_type,
            "object_part": self.object_part,
            "claim_status": self.claim_status,
            "claim_status_justification": self.claim_status_justification,
            "supporting_image_ids": sup_str,
            "valid_image": "true" if self.valid_image else "false",
            "severity": self.severity,
        }

    @field_validator("issue_type", "object_part", "claim_status", "severity", mode="before")
    @classmethod
    def strip_lower(cls, v: str) -> str:
        return str(v).strip().lower() if v is not None else ""
