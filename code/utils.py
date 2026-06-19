"""JSON extraction from model responses."""

import json
import re
from typing import Any


def parse_json_response(raw: str) -> Any:
  text = raw.strip()
  if text.startswith("```"):
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
  try:
    return json.loads(text)
  except json.JSONDecodeError:
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
      return json.loads(match.group())
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
      return json.loads(match.group())
    raise
