"""Playbook-driven decision engine."""

from typing import List, Set

from schemas import ClaimExtraction, ClaimOutput, ImageAnalysis


def _best_analysis(analyses: List[ImageAnalysis]) -> ImageAnalysis | None:
  usable = [a for a in analyses if a.is_valid and not a.is_corrupt]
  if not usable:
    return analyses[0] if analyses else None
  return max(
    usable,
    key=lambda a: (
      a.damage_visible,
      a.issue_type != "unknown",
      a.object_part != "unknown",
      len(a.notes),
    ),
  )


def _aggregate_issue(analyses: List[ImageAnalysis]) -> str:
  best = _best_analysis(analyses)
  if not best:
    return "unknown"
  if best.damage_visible:
    return best.issue_type if best.issue_type != "unknown" else "unknown"
  if best.object_part != "unknown":
    return "none"
  return best.issue_type


def _aggregate_part(analyses: List[ImageAnalysis]) -> str:
  best = _best_analysis(analyses)
  return best.object_part if best else "unknown"


def _aggregate_severity(analyses: List[ImageAnalysis]) -> str:
  best = _best_analysis(analyses)
  if not best:
    return "unknown"
  if not best.damage_visible:
    return "none"
  return best.severity if best.severity != "unknown" else "medium"


def _supporting_ids(analyses: List[ImageAnalysis], claim_status: str) -> List[str]:
  if claim_status == "not_enough_information":
    usable = [a for a in analyses if a.is_valid and a.damage_visible]
    if usable:
      return [a.image_id for a in usable]
    return []
  if claim_status == "contradicted":
    usable = [a for a in analyses if a.is_valid]
    if usable:
      return [usable[0].image_id]
    return []
  # supported
  ids = [a.image_id for a in analyses if a.is_valid and a.damage_visible]
  if ids:
    return ids
  usable = [a for a in analyses if a.is_valid]
  return [usable[0].image_id] if usable else []


def decide(
  row: dict,
  extraction: ClaimExtraction,
  analyses: List[ImageAnalysis],
  evidence_met: bool,
  evidence_reason: str,
  conflict_flags: Set[str],
  has_conflict: bool,
  severity_mismatch: bool,
  history_flags: Set[str],
  quality_flags: Set[str],
) -> ClaimOutput:
  all_flags: Set[str] = set()
  all_flags.update(conflict_flags)
  all_flags.update(history_flags)
  all_flags.update(quality_flags)

  issue_type = _aggregate_issue(analyses)
  object_part = _aggregate_part(analyses)
  severity = _aggregate_severity(analyses)
  valid_image = any(a.is_valid and not a.is_corrupt for a in analyses)

  claim_status = "not_enough_information"
  justification = ""

  if not evidence_met:
    claim_status = "not_enough_information"
    justification = evidence_reason
    if not valid_image:
      all_flags.add("damage_not_visible")
  elif has_conflict or "wrong_object" in all_flags:
    if severity_mismatch or "claim_mismatch" in all_flags:
      claim_status = "contradicted"
      best = _best_analysis(analyses)
      justification = (
        f"The images show {best.issue_type.replace('_', ' ')} on the "
        f"{best.object_part.replace('_', ' ')} rather than the claimed "
        f"{extraction.claimed_issue_type.replace('_', ' ')} on "
        f"{extraction.claimed_object_part.replace('_', ' ')}, so the claim is contradicted."
      )
    elif "wrong_object" in all_flags:
      claim_status = "contradicted"
      best = _best_analysis(analyses)
      justification = (
        f"The image shows a different object than the claimed {row['claim_object']}, "
        f"so it does not support the user's claim."
      )
    else:
      claim_status = "not_enough_information"
      justification = (
        "The submitted images do not reliably support the claim due to conflicting or insufficient evidence."
      )
      all_flags.add("manual_review_required")
  else:
    usable = [a for a in analyses if a.is_valid and not a.is_corrupt]
    claimed_damage = extraction.claimed_issue_type not in ("unknown", "none", "")
    any_damage = any(a.damage_visible for a in usable)
    part_match = any(
      extraction.claimed_object_part.replace("_", " ") in a.object_part.replace("_", " ")
      or a.object_part.replace("_", " ") in extraction.claimed_object_part.replace("_", " ")
      for a in usable
    )

    if claimed_damage and not any_damage and part_match:
      claim_status = "contradicted"
      justification = (
        f"The image shows the {object_part.replace('_', ' ')} area but does not show "
        f"clear {extraction.claimed_issue_type.replace('_', ' ')} damage, "
        "contradicting the physical damage claim."
      )
      all_flags.add("damage_not_visible")
    elif any_damage and part_match:
      claim_status = "supported"
      sup_ids = _supporting_ids(analyses, "supported")
      id_ref = sup_ids[0] if sup_ids else "image"
      justification = (
        f"The image{'s' if len(sup_ids) > 1 else ''} support the claim because "
        f"{issue_type.replace('_', ' ')} on the {object_part.replace('_', ' ')} "
        f"is visible in {id_ref}."
      )
    elif any_damage:
      claim_status = "supported"
      sup_ids = _supporting_ids(analyses, "supported")
      justification = (
        f"Visible {issue_type.replace('_', ' ')} damage supports the claim "
        f"in the submitted image(s)."
      )
    else:
      claim_status = "not_enough_information"
      justification = (
        "The submitted images do not provide enough evidence to verify the claimed damage."
      )
      all_flags.add("damage_not_visible")

  # Manual review triggers from playbook
  if len({a.issue_type for a in analyses if a.damage_visible}) > 1:
    all_flags.add("manual_review_required")
  if "manual_review_required" in history_flags:
    all_flags.add("manual_review_required")

  all_flags.discard("none")
  if not all_flags:
    all_flags.add("none")

  flags_list = sorted(all_flags)
  if flags_list == ["none"]:
    flags_list = ["none"]
  elif "none" in flags_list:
    flags_list = [f for f in flags_list if f != "none"]

  supporting = _supporting_ids(analyses, claim_status)

  return ClaimOutput(
    user_id=row["user_id"],
    image_paths=row["image_paths"],
    user_claim=row["user_claim"],
    claim_object=row["claim_object"],
    evidence_standard_met=evidence_met,
    evidence_standard_met_reason=evidence_reason,
    risk_flags=flags_list,
    issue_type=issue_type,
    object_part=object_part,
    claim_status=claim_status,
    claim_status_justification=justification,
    supporting_image_ids=supporting,
    valid_image=valid_image,
    severity=severity,
  )
