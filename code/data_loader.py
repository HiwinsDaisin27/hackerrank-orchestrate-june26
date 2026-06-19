"""Data loading utilities."""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

from config import DATASET_DIR, EVIDENCE_REQUIREMENTS, USER_HISTORY


def load_csv(path: Path) -> List[dict]:
  with open(path, newline="", encoding="utf-8") as f:
    return list(csv.DictReader(f))


def load_user_history() -> Dict[str, dict]:
  rows = load_csv(USER_HISTORY)
  return {r["user_id"]: r for r in rows}


def load_evidence_requirements() -> List[dict]:
  return load_csv(EVIDENCE_REQUIREMENTS)


def parse_image_paths(image_paths: str, dataset_dir: Path = None) -> List[Tuple[str, Path]]:
  """Return list of (image_id, absolute_path). Paths in CSV are relative to dataset/."""
  from config import DATASET_DIR

  base = dataset_dir or DATASET_DIR
  result = []
  for raw in image_paths.split(";"):
    raw = raw.strip()
    if not raw:
      continue
    p = Path(raw)
    if not p.is_absolute():
      p = base / raw
    image_id = Path(raw).stem
    result.append((image_id, p))
  return result
