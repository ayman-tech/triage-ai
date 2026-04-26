from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class RootCauseHypothesis(BaseModel):
    """
    First-class output representing the likely operational/control root cause.

    This is intentionally company-aware: it should be grounded in retrieved
    control knowledge and complaint evidence.
    """

    root_cause_category: str = Field(..., description="Internal root cause category")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Brief explanation grounded in evidence")
    controls_to_check: list[str] = Field(default_factory=list)
    notes: Optional[str] = None

