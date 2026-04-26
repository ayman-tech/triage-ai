"""Parse JSON objects from LLM text (markdown fences, leading prose)."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_llm_json(content: str | None) -> dict[str, Any]:
    if content is None or not str(content).strip():
        raise ValueError("LLM returned empty content; expected a JSON object.")

    text = str(content).strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)

    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in LLM output: {text[:300]!r}")

    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError(f"LLM JSON was not an object: {type(obj).__name__}")
    return obj
