from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """
    Minimal evidence trace schema.

    This is designed to be JSON-serializable and stored alongside the case
    for explainability and auditability.
    """

    evidence_type: str = Field(..., description="e.g. precedent_complaint, policy_snippet, taxonomy_candidate")
    summary: str = Field(..., description="Human-readable evidence summary")
    source_ref: Optional[str] = Field(None, description="Optional ID (complaint_id, policy_id, etc.)")
    score: Optional[float] = Field(None, description="Optional relevance/score in [0,1] or distance")
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceTrace(BaseModel):
    items: list[EvidenceItem] = Field(default_factory=list)

