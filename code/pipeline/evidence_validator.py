"""Deterministic evidence requirement validation."""

from typing import List, Tuple

from schemas import ClaimExtraction, ImageAnalysis


def _normalize(s: str) -> str:
  return s.lower().strip().replace("_", " ").replace("-", " ")


def _part_matches(claimed: str, visible: str) -> bool:
  c, v = _normalize(claimed), _normalize(visible)
  if not c or c == "unknown" or v == "unknown":
    return True
  if c == v:
    return True
  if c in v or v in c:
    return True
  # bumper variants
  bumper_parts = {"front bumper", "rear bumper", "bumper"}
  if c in bumper_parts and v in bumper_parts:
    return c == v or "bumper" in c and "bumper" in v
  return False


def _issue_family_matches(requirement_applies: str, claimed_issue: str) -> bool:
  applies = _normalize(requirement_applies)
  issue = _normalize(claimed_issue)
  if applies == "general claim review":
    return True
  tokens = applies.split()
  return any(t in issue for t in tokens) or issue in applies


def select_requirements(
  requirements: List[dict],
  claim_object: str,
  claimed_issue: str,
) -> List[dict]:
  selected = []
  for req in requirements:
    obj = req["claim_object"]
    if obj != "all" and obj != claim_object:
      continue
    if _issue_family_matches(req["applies_to"], claimed_issue):
      selected.append(req)
  if not selected:
    for req in requirements:
      if req["claim_object"] == "all" and req["applies_to"] == "general claim review":
        selected.append(req)
  return selected


def validate_evidence(
  requirements: List[dict],
  claim_object: str,
  extraction: ClaimExtraction,
  analyses: List[ImageAnalysis],
) -> Tuple[bool, str]:
  """Return (evidence_standard_met, reason)."""
  claimed_issue = extraction.claimed_issue_type
  claimed_part = extraction.claimed_object_part
  applicable = select_requirements(requirements, claim_object, claimed_issue)

  usable = [a for a in analyses if a.is_valid and not a.is_corrupt]
  if not usable:
    return False, "No usable images in the submitted set."

  # Check object identity across images
  wrong_object_count = sum(1 for a in usable if not a.object_matches_claim_type)
  if wrong_object_count == len(usable):
    return False, (
      "Submitted images do not show the claimed object type clearly enough to evaluate."
    )

  # Multi-image identity check
  objects = {a.detected_object for a in usable if a.detected_object not in ("unknown", "other")}
  if len(objects) > 1 and claim_object == "car":
    return False, (
      "The image set appears to show different vehicles, so vehicle identity evidence is not satisfied."
    )

  # Part visibility for claimed part
  part_visible = any(
    _part_matches(claimed_part, a.object_part) or a.damage_visible
    for a in usable
  )
  if claimed_part != "unknown" and not part_visible:
    # Check if any image shows claimed part area
    has_part_view = any(
      _part_matches(claimed_part, a.object_part) for a in usable
    )
    if not has_part_view:
      for req in applicable:
        if "visible" in req["minimum_image_evidence"].lower():
          return False, (
            f"The claimed {claimed_part.replace('_', ' ')} is not visible clearly enough "
            "in the submitted images to evaluate the claim."
          )

  # Contents / missing item requirements
  if "contents" in _normalize(claimed_part) or "missing" in _normalize(claimed_issue):
    for req in applicable:
      if "contents" in req["applies_to"].lower():
        contents_visible = any(
          a.object_part in ("contents", "item") and a.is_valid for a in usable
        )
        if not contents_visible:
          return False, (
            "The images do not clearly show the opened package contents "
            "enough to verify missing or damaged items."
          )

  # General sufficiency: at least one image with clear view
  clear_images = [
    a for a in usable
    if a.damage_visible or a.object_part != "unknown" or a.issue_type != "unknown"
  ]
  if not clear_images:
    return False, "Submitted images do not provide enough visual evidence to evaluate the claim."

  # Build positive reason
  best = max(usable, key=lambda a: (a.damage_visible, a.is_valid))
  part_name = best.object_part.replace("_", " ")
  if best.damage_visible:
    return True, (
      f"The {part_name} is visible and the damage can be verified from the submitted image(s)."
    )
  return True, (
    f"The claimed area is visible enough in the submitted image(s) to evaluate the claim."
  )
