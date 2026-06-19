"""Risk assessment from user history — never overrides image evidence."""

from typing import List, Set


def assess_history_risk(history: dict | None) -> Set[str]:
  flags: Set[str] = set()
  if not history:
    return flags

  hf = history.get("history_flags", "none")
  if hf and hf != "none":
    for f in hf.split(";"):
      f = f.strip()
      if f:
        flags.add(f)

  rejected = int(history.get("rejected_claim", 0) or 0)
  manual = int(history.get("manual_review_claim", 0) or 0)
  last90 = int(history.get("last_90_days_claim_count", 0) or 0)

  if rejected >= 3 or (rejected >= 2 and last90 >= 4):
    flags.add("user_history_risk")

  if manual >= 2:
    flags.add("manual_review_required")

  return flags


def merge_quality_flags(analyses: list) -> Set[str]:
  flags: Set[str] = set()
  for a in analyses:
    for q in a.quality_flags:
      if q and q != "none":
        flags.add(q)
    if a.text_instruction_in_image:
      flags.add("text_instruction_present")
    if a.possible_manipulation:
      flags.add("possible_manipulation")
    if a.non_original_image:
      flags.add("non_original_image")
    if not a.object_matches_claim_type and a.detected_object not in ("unknown", ""):
      flags.add("wrong_object")
  return flags
