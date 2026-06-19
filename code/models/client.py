"""VLM/LLM client abstraction with API-only Gemini and OpenAI providers."""

import base64
import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
  CACHE_DIR,
  ENABLE_CACHE,
  GEMINI_TEXT_MODEL,
  GEMINI_VISION_MODEL,
  GOOGLE_API_KEY,
  MAX_RETRIES,
  OPENAI_API_KEY,
  OPENAI_TEXT_MODEL,
  OPENAI_VISION_MODEL,
  RETRY_DELAY_SEC,
)

logger = logging.getLogger(__name__)


class UsageTracker:
  """Track model calls, tokens, and images for operational analysis."""

  def __init__(self):
    self.model_calls = 0
    self.images_sent = 0
    self.input_tokens_est = 0
    self.output_tokens_est = 0
    self.errors = 0

  def record(self, images: int = 0, input_est: int = 0, output_est: int = 0):
    self.model_calls += 1
    self.images_sent += images
    self.input_tokens_est += input_est
    self.output_tokens_est += output_est

  def to_dict(self) -> dict:
    return {
      "model_calls": self.model_calls,
      "images_sent": self.images_sent,
      "input_tokens_est": self.input_tokens_est,
      "output_tokens_est": self.output_tokens_est,
      "errors": self.errors,
    }


USAGE = UsageTracker()


def _cache_key(payload: str) -> str:
  return hashlib.sha256(payload.encode()).hexdigest()


def _read_cache(key: str) -> Optional[str]:
  if not ENABLE_CACHE:
    return None
  path = CACHE_DIR / f"{key}.json"
  if path.exists():
    try:
      return json.loads(path.read_text(encoding="utf-8"))["response"]
    except (json.JSONDecodeError, KeyError):
      return None
  return None


def _write_cache(key: str, response: str):
  if not ENABLE_CACHE:
    return
  CACHE_DIR.mkdir(parents=True, exist_ok=True)
  path = CACHE_DIR / f"{key}.json"
  path.write_text(json.dumps({"response": response}), encoding="utf-8")


def _retry_delay(error: Exception, attempt: int) -> float:
  message = str(error)
  match = re.search(r"retry(?:Delay| in)?['\"]?\s*:?\s*['\"]?(\d+(?:\.\d+)?)s", message, re.IGNORECASE)
  if not match:
    match = re.search(r"Please retry in (\d+(?:\.\d+)?)s", message, re.IGNORECASE)
  if match:
    return max(float(match.group(1)) + 1.0, RETRY_DELAY_SEC)
  return RETRY_DELAY_SEC * (attempt + 1)


class ModelClient(ABC):
  @abstractmethod
  def complete_text(self, prompt: str, system: str = "") -> str:
    ...

  @abstractmethod
  def complete_vision(
    self, prompt: str, image_paths: List[Path], system: str = ""
  ) -> str:
    ...


class GeminiClient(ModelClient):
  def __init__(self):
    from google import genai
    from google.genai import types

    self._client = genai.Client(api_key=GOOGLE_API_KEY)
    self._types = types
    self.text_model = GEMINI_TEXT_MODEL
    self.vision_model = GEMINI_VISION_MODEL

  def _generate(self, contents, model: str, system: str = "") -> str:
    cfg = self._types.GenerateContentConfig(
      temperature=0.1,
      response_mime_type="application/json",
      system_instruction=system or None,
    )
    for attempt in range(MAX_RETRIES):
      try:
        resp = self._client.models.generate_content(
          model=model, contents=contents, config=cfg
        )
        text = resp.text or ""
        USAGE.record(
          images=sum(1 for c in contents if hasattr(c, "inline_data")),
          input_est=len(str(contents)) // 4,
          output_est=len(text) // 4,
        )
        return text
      except Exception as e:
        USAGE.errors += 1
        logger.warning("Gemini attempt %d failed: %s", attempt + 1, e)
        if attempt < MAX_RETRIES - 1:
          time.sleep(_retry_delay(e, attempt))
        else:
          raise

  def complete_text(self, prompt: str, system: str = "") -> str:
    key = _cache_key(f"gemini-text:{self.text_model}:{system}:{prompt}")
    cached = _read_cache(key)
    if cached:
      return cached
    result = self._generate(prompt, self.text_model, system)
    _write_cache(key, result)
    return result

  def complete_vision(
    self, prompt: str, image_paths: List[Path], system: str = ""
  ) -> str:
    parts: list = [prompt]
    for p in image_paths:
      data = p.read_bytes()
      mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
      parts.append(self._types.Part.from_bytes(data=data, mime_type=mime))
    key = _cache_key(
      f"gemini-vision:{self.vision_model}:{system}:{prompt}:"
      f"{','.join(str(p) for p in image_paths)}"
    )
    cached = _read_cache(key)
    if cached:
      return cached
    result = self._generate(parts, self.vision_model, system)
    _write_cache(key, result)
    return result


class OpenAIClient(ModelClient):
  def __init__(self):
    from openai import OpenAI

    self._client = OpenAI(api_key=OPENAI_API_KEY)
    self.text_model = OPENAI_TEXT_MODEL
    self.vision_model = OPENAI_VISION_MODEL

  def _chat(self, messages: List[Dict[str, Any]], model: str) -> str:
    for attempt in range(MAX_RETRIES):
      try:
        resp = self._client.chat.completions.create(
          model=model,
          messages=messages,
          temperature=0.1,
          response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or ""
        img_count = sum(
          1 for m in messages for c in m.get("content", [])
          if isinstance(c, dict) and c.get("type") == "image_url"
        )
        USAGE.record(
          images=img_count,
          input_est=resp.usage.prompt_tokens if resp.usage else 0,
          output_est=resp.usage.completion_tokens if resp.usage else 0,
        )
        return text
      except Exception as e:
        USAGE.errors += 1
        logger.warning("OpenAI attempt %d failed: %s", attempt + 1, e)
        if attempt < MAX_RETRIES - 1:
          time.sleep(_retry_delay(e, attempt))
        else:
          raise

  def complete_text(self, prompt: str, system: str = "") -> str:
    key = _cache_key(f"openai-text:{self.text_model}:{system}:{prompt}")
    cached = _read_cache(key)
    if cached:
      return cached
    messages = []
    if system:
      messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    result = self._chat(messages, self.text_model)
    _write_cache(key, result)
    return result

  def complete_vision(
    self, prompt: str, image_paths: List[Path], system: str = ""
  ) -> str:
    content: list = [{"type": "text", "text": prompt}]
    for p in image_paths:
      data = base64.b64encode(p.read_bytes()).decode()
      mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
      content.append({
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{data}"},
      })
    key = _cache_key(
      f"openai-vision:{self.vision_model}:{system}:{prompt}:"
      f"{','.join(str(p) for p in image_paths)}"
    )
    cached = _read_cache(key)
    if cached:
      return cached
    messages = []
    if system:
      messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    result = self._chat(messages, self.vision_model)
    _write_cache(key, result)
    return result


def get_client() -> ModelClient:
  if GOOGLE_API_KEY:
    return GeminiClient()
  if OPENAI_API_KEY:
    return OpenAIClient()
  raise RuntimeError(
    "No API credentials found. Set GOOGLE_API_KEY for Gemini, or "
    "OPENAI_API_KEY for the OpenAI fallback."
  )
