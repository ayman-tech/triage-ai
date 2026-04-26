"""Redact or truncate text before logs / audit JSON snapshots."""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any

_NARRATIVE_MAX = int(os.getenv("TRACE_NARRATIVE_MAX_CHARS", "240"))


def redact_narrative(text: str | None, *, max_chars: int | None = None) -> str:
    if not text:
        return ""
    limit = max_chars if max_chars is not None else _NARRATIVE_MAX
    t = str(text).strip()
    t = re.sub(r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b", "[SSN]", t)
    t = re.sub(r"\b\d{13,19}\b", "[PAN]", t)
    if len(t) <= limit:
        return t
    return t[:limit] + f"...[truncated,len={len(t)}]"


def prompt_fingerprint(text: str | None) -> str:
    """Short stable hash for correlating prompt versions without storing full text."""
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def json_safe(obj: Any) -> Any:
    """Best-effort JSON-serializable structure for snapshots."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, "model_dump"):
        return json_safe(obj.model_dump())
    if hasattr(obj, "value") and type(obj).__name__ != "dict":
        try:
            return getattr(obj, "value", str(obj))
        except Exception:
            return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    return str(obj)
