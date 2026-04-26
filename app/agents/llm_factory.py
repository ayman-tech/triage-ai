"""Gemini client factory.

Reads GEMINI_API_KEY and GEMINI_CHAT_MODEL from the environment and returns
a configured google.genai.Client instance.  A module-level singleton is used
so the same client is reused across all agent calls in a process.
"""

from __future__ import annotations

import os

import google.genai as genai

_DEFAULT_MODEL = "gemini-2.5-flash"

_client: genai.Client | None = None


def get_gemini_client() -> genai.Client:
    """Return the shared Gemini client (created once per process)."""
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable is not set."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def default_model_name() -> str:
    """Return the configured Gemini model name."""
    return os.getenv("GEMINI_CHAT_MODEL", _DEFAULT_MODEL)


def fallback_model_names() -> list[str]:
    """Return fallback Gemini models to try after quota/model errors."""
    raw = os.getenv("GEMINI_FALLBACK_MODELS", "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def candidate_model_names(model_name: str | None = None) -> list[str]:
    """Return primary model plus configured fallbacks without duplicates."""
    candidates: list[str] = []
    for name in [model_name or default_model_name(), *fallback_model_names()]:
        if name and name not in candidates:
            candidates.append(name)
    return candidates


def get_provider() -> str:
    """Return the configured LLM provider label for observability."""
    return os.getenv("LLM_PROVIDER", "google")
