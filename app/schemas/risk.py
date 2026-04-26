"""Pydantic models for risk‑assessment output."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskFactor(BaseModel):
    name: str
    description: str
    weight: float = Field(..., ge=0.0, le=1.0)


class RiskAssessment(BaseModel):
    """Structured output produced by the risk‑assessment agent."""

    risk_level: RiskLevel
    risk_score: float = Field(
        ..., ge=0.0, le=100.0, description="Numeric risk score (0‑100)"
    )
    factors: list[RiskFactor] = Field(
        default_factory=list, description="Contributing risk factors"
    )
    regulatory_risk: bool = Field(
        False, description="Whether the complaint poses regulatory exposure"
    )
    financial_impact_estimate: Optional[float] = Field(
        None, description="Estimated financial impact in USD"
    )
    escalation_required: bool = False
    reasoning: str = Field(
        ..., description="Explanation of the risk assessment"
    )
