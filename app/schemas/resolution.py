"""Pydantic models for resolution recommendation output."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ResolutionAction(str, Enum):
    MONETARY_RELIEF = "monetary_relief"
    NON_MONETARY_RELIEF = "non_monetary_relief"
    EXPLANATION = "explanation"
    CORRECTION = "correction"
    REFERRAL = "referral"
    NO_ACTION = "no_action"


class ResolutionRecommendation(BaseModel):
    """Structured output produced by the resolution agent."""

    recommended_action: ResolutionAction
    description: str = Field(
        ..., description="Detailed description of the recommended resolution"
    )
    similar_case_ids: list[str] = Field(
        default_factory=list,
        description="IDs of historically similar cases used as precedent",
    )
    estimated_resolution_days: int = Field(
        ..., ge=1, description="Expected number of days to resolve"
    )
    monetary_amount: Optional[float] = Field(
        None, description="Monetary relief amount, if applicable"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence in recommendation"
    )
    reasoning: str = Field(
        ..., description="Explanation for the recommendation"
    )
