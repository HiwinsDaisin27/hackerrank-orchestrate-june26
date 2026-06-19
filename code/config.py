"""Configuration loaded from environment variables."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
CODE_DIR = Path(__file__).resolve().parent

# Load .env if present (never commit secrets)
_env_path = REPO_ROOT / ".env"
if _env_path.exists():
  for line in _env_path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
      k, _, v = line.partition("=")
      os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# API model preferences (free-tier friendly)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_TEXT_MODEL = os.environ.get("GEMINI_TEXT_MODEL", "gemini-3.1-flash-lite")
GEMINI_VISION_MODEL = os.environ.get("GEMINI_VISION_MODEL", GEMINI_MODEL)
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", OPENAI_MODEL)
OPENAI_VISION_MODEL = os.environ.get("OPENAI_VISION_MODEL", OPENAI_MODEL)

# Rate limiting / retries
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_DELAY_SEC = float(os.environ.get("RETRY_DELAY_SEC", "2.0"))

# Caching
CACHE_DIR = CODE_DIR / ".cache"
ENABLE_CACHE = os.environ.get("ENABLE_CACHE", "true").lower() in ("1", "true", "yes")

DEFAULT_OUTPUT = REPO_ROOT / "output.csv"
SAMPLE_CLAIMS = DATASET_DIR / "sample_claims.csv"
TEST_CLAIMS = DATASET_DIR / "claims.csv"
USER_HISTORY = DATASET_DIR / "user_history.csv"
EVIDENCE_REQUIREMENTS = DATASET_DIR / "evidence_requirements.csv"
