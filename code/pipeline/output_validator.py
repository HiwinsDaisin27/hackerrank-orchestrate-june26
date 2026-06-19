"""Final schema validation and allowed-value enforcement."""

from schemas import (
  CAR_PARTS,
  ClaimOutput,
  ClaimStatus,
  IssueType,
  LAPTOP_PARTS,
  PACKAGE_PARTS,
  RiskFlag,
  Severity,
  parts_for_object,
)


def _clamp_enum(value: str, allowed: set, default: str) -> str:
  v = value.lower().strip()
  if v in allowed:
    return v
  for a in allowed:
    if v.replace(" ", "_") == a or v in a:
      return a
  return default


def validate_output(output: ClaimOutput) -> ClaimOutput:
  parts = parts_for_object(output.claim_object)

  issue = _clamp_enum(output.issue_type, {e.value for e in IssueType}, "unknown")
  part = _clamp_enum(output.object_part, parts, "unknown")
  status = _clamp_enum(
    output.claim_status,
    {e.value for e in ClaimStatus},
    "not_enough_information",
  )
  severity = _clamp_enum(output.severity, {e.value for e in Severity}, "unknown")

  allowed_flags = {e.value for e in RiskFlag}
  flags = []
  for f in output.risk_flags:
    cf = _clamp_enum(f, allowed_flags, "")
    if cf and cf != "none":
      flags.append(cf)
  if not flags:
    flags = ["none"]

  # Consistency fixes
  if status == "supported" and issue == "none" and output.claim_object != "package":
    status = "contradicted"

  if not output.evidence_standard_met and status == "supported":
    status = "not_enough_information"

  return ClaimOutput(
    user_id=output.user_id,
    image_paths=output.image_paths,
    user_claim=output.user_claim,
    claim_object=output.claim_object,
    evidence_standard_met=output.evidence_standard_met,
    evidence_standard_met_reason=output.evidence_standard_met_reason,
    risk_flags=flags,
    issue_type=issue,
    object_part=part,
    claim_status=status,
    claim_status_justification=output.claim_status_justification,
    supporting_image_ids=output.supporting_image_ids,
    valid_image=output.valid_image,
    severity=severity,
  )
