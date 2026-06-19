"""Main orchestration pipeline."""

import logging
from typing import List

from data_loader import load_evidence_requirements, load_user_history, parse_image_paths
from models.client import ModelClient, get_client
from pipeline.claim_extraction import extract_claim
from pipeline.conflict_detection import detect_conflicts
from pipeline.decision_engine import decide
from pipeline.evidence_validator import validate_evidence
from pipeline.output_validator import validate_output
from pipeline.risk_assessment import assess_history_risk, merge_quality_flags
from pipeline.vision_analysis import analyze_images
from schemas import ClaimOutput

logger = logging.getLogger(__name__)

_client: ModelClient | None = None


def _get_client() -> ModelClient:
  global _client
  if _client is None:
    _client = get_client()
  return _client


def process_claim(
  row: dict,
  requirements: List[dict],
  history_map: dict,
  repo_root=None,
) -> ClaimOutput:
  from config import DATASET_DIR

  root = repo_root or DATASET_DIR.parent
  dataset_dir = root / "dataset" if (root / "dataset").exists() else DATASET_DIR
  client = _get_client()

  # Stage 1: Claim Extraction (text only)
  extraction = extract_claim(client, row["user_claim"], row["claim_object"])

  # Stage 2-3: Image paths + Vision Analysis (images primary)
  image_entries = parse_image_paths(row["image_paths"], dataset_dir)

  # Stage 4: Vision Analysis
  vision = analyze_images(client, row["claim_object"], image_entries)
  analyses = vision.images

  # Stage 5: Evidence Requirement Validation
  evidence_met, evidence_reason = validate_evidence(
    requirements, row["claim_object"], extraction, analyses
  )

  # Stage 6: Conflict Detection
  has_conflict, severity_mismatch, conflict_flags, _ = detect_conflicts(
    extraction, analyses, row["claim_object"]
  )
  # Drop claim_mismatch when issue and part align on best visible image
  best = max(
    [a for a in analyses if a.is_valid],
    key=lambda a: a.damage_visible,
    default=None,
  )
  if best and best.damage_visible:
    from pipeline.conflict_detection import _issues_compatible, _parts_compatible
    if (
      _issues_compatible(extraction.claimed_issue_type, best.issue_type)
      and _parts_compatible(extraction.claimed_object_part, best.object_part, row["claim_object"])
    ):
      conflict_flags.discard("claim_mismatch")
      conflict_flags.discard("wrong_object_part")
      has_conflict = bool(conflict_flags)

  # Stage 7: Risk Assessment
  history = history_map.get(row["user_id"])
  history_flags = assess_history_risk(history)
  quality_flags = merge_quality_flags(analyses)

  # Stage 8: Decision Engine (playbook-driven)
  output = decide(
    row,
    extraction,
    analyses,
    evidence_met,
    evidence_reason,
    conflict_flags,
    has_conflict,
    severity_mismatch,
    history_flags,
    quality_flags,
  )

  # Stage 9: Output Validation
  return validate_output(output)


def process_claims(rows: List[dict], repo_root=None) -> List[ClaimOutput]:
  requirements = load_evidence_requirements()
  history_map = load_user_history()
  results = []
  for i, row in enumerate(rows):
    logger.info("Processing claim %d/%d user=%s", i + 1, len(rows), row["user_id"])
    try:
      results.append(process_claim(row, requirements, history_map, repo_root))
    except Exception as e:
      logger.error("Failed claim %s: %s", row["user_id"], e)
      results.append(_fallback_output(row, str(e)))
  return results


def _fallback_output(row: dict, error: str) -> ClaimOutput:
  return ClaimOutput(
    user_id=row["user_id"],
    image_paths=row["image_paths"],
    user_claim=row["user_claim"],
    claim_object=row["claim_object"],
    evidence_standard_met=False,
    evidence_standard_met_reason=f"Processing error: {error}",
    risk_flags=["manual_review_required"],
    issue_type="unknown",
    object_part="unknown",
    claim_status="not_enough_information",
    claim_status_justification="Unable to evaluate due to processing error.",
    supporting_image_ids=[],
    valid_image=False,
    severity="unknown",
  )
