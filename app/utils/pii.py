"""Shared PII redaction utilities."""

from __future__ import annotations

import re


def redact_pii(text: str) -> str:
    """Replace common PII patterns with redaction tokens.

    Currently handles:
    - SSNs (e.g. 123-45-6789, 123.45.6789, 123456789)
    - 16-digit card numbers
    - Email addresses
    - Phone numbers (US formats)
    """
    # SSNs
    text = re.sub(r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b", "[SSN_REDACTED]", text)
    # 16-digit card numbers (with optional separators)
    text = re.sub(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[CARD_REDACTED]", text)
    # Email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_REDACTED]", text)
    # US phone numbers (various formats)
    text = re.sub(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE_REDACTED]", text)
    return text
