"""Explicit conflict detection between claim text and visual findings."""

from typing import List, Set, Tuple

from schemas import ClaimExtraction, ImageAnalysis


def _norm(s: str) -> str:
  return s.lower().strip().replace("_", " ").replace("-", " ")


def _normalize_part(part: str) -> str:
  p = _norm(part)
  synonyms = {
    "door panel": "door",
    "front glass": "windshield",
    "back light": "taillight",
    "rear light": "taillight",
    "side mirror": "side mirror",
    "left headlight": "headlight",
    "right headlight": "headlight",
    "package corner": "package corner",
  }
  for k, v in synonyms.items():
    if k in p:
      return v
  return p


def _issues_compatible(claimed: str, visible: str) -> bool:
  c, v = _norm(claimed), _norm(visible)
  if v in ("unknown", "none", ""):
    return True
  if c == v:
    return True
  if c in v or v in c:
    return True
  families = {
    "dent": {"dent", "scratch"},
    "scratch": {"scratch", "dent"},
    "crack": {"crack", "glass shatter", "broken part", "glass_shatter", "broken_part"},
    "glass shatter": {"crack", "glass shatter", "broken part", "glass_shatter", "broken_part", "shatter"},
    "shatter": {"crack", "glass shatter", "glass_shatter", "broken_part"},
    "glass_shatter": {"crack", "glass shatter", "broken part", "glass_shatter", "broken_part", "shatter"},
    "broken part": {"broken part", "crack", "missing part", "broken_part", "missing_part"},
    "broken_part": {"broken part", "crack", "missing part", "broken_part", "missing_part"},
    "missing part": {"missing part", "broken part", "missing_part", "broken_part"},
    "torn packaging": {"torn packaging", "torn_packaging"},
    "crushed packaging": {"crushed packaging", "crushed_packaging", "dent"},
    "water damage": {"water damage", "stain", "water_damage", "stain"},
    "stain": {"stain", "water damage", "water_damage"},
  }
  cn = c.replace("_", " ")
  vn = v.replace("_", " ")
  for key, members in families.items():
    if cn == key or cn in members:
      if vn == key or vn in members or any(m in vn for m in members):
        return True
  return False


def _parts_compatible(claimed: str, visible: str, claim_object: str) -> bool:
  c, v = _normalize_part(claimed), _normalize_part(visible)
  if v in ("unknown", ""):
    return True
  if c == v:
    return True
  if c in v or v in c:
    return True
  if "front" in c and "rear" in v or "rear" in c and "front" in v:
    return False
  if "left" in c and "right" in v or "right" in c and "left" in v:
    return False
  # visible part may include extra context (e.g. headlight and fender)
  if c in v.split(" and ")[0] or any(c in seg for seg in v.split(" and ")):
    return True
  return False


def detect_conflicts(
  extraction: ClaimExtraction,
  analyses: List[ImageAnalysis],
  claim_object: str,
) -> Tuple[bool, bool, Set[str], str]:
  """
  Returns: (has_conflict, severity_mismatch, extra_flags, conflict_summary)
  """
  flags: Set[str] = set()
  usable = [a for a in analyses if a.is_valid and not a.is_corrupt]
  if not usable:
    return False, False, flags, ""

  claimed_issue = extraction.claimed_issue_type
  claimed_part = extraction.claimed_object_part

  part_conflicts = []
  issue_conflicts = []
  severity_mismatch = False

  for a in usable:
    if not a.object_matches_claim_type and a.detected_object not in ("unknown", claim_object):
      flags.add("wrong_object")

    if not _parts_compatible(claimed_part, a.object_part, claim_object):
      flags.add("wrong_object_part")
      flags.add("claim_mismatch")
      part_conflicts.append(f"{a.image_id}: shows {a.object_part}")

    if a.damage_visible and not _issues_compatible(claimed_issue, a.issue_type):
      # e.g. claimed severe dent but only scratch visible
      if claimed_issue in ("dent", "broken_part") and a.issue_type == "scratch":
        severity_mismatch = True
        flags.add("claim_mismatch")
        issue_conflicts.append(f"{a.image_id}: {a.issue_type} not {claimed_issue}")
      elif not _issues_compatible(claimed_issue, a.issue_type):
        flags.add("claim_mismatch")
        issue_conflicts.append(f"{a.image_id}: {a.issue_type} vs claimed {claimed_issue}")

    if not a.damage_visible and claimed_issue not in ("unknown", "none"):
      if a.object_part != "unknown" and _parts_compatible(claimed_part, a.object_part, claim_object):
        flags.add("damage_not_visible")

  has_conflict = bool(part_conflicts or issue_conflicts or "wrong_object" in flags)

  summary_parts = []
  if part_conflicts:
    summary_parts.append("part mismatch: " + "; ".join(part_conflicts))
  if issue_conflicts:
    summary_parts.append("issue mismatch: " + "; ".join(issue_conflicts))

  return has_conflict, severity_mismatch, flags, "; ".join(summary_parts)
