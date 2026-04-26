"""Minimal Document dataclass used by retrieval indexes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """A retrieved text chunk with associated metadata."""

    page_content: str
    metadata: dict = field(default_factory=dict)
