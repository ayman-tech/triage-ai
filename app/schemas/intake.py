"""Schemas for channel-agnostic complaint intake (chat + voice)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class IntakeIntent(str, Enum):
    COMPLAINT = "complaint"
    DISPUTE = "dispute"
    SUPPORT_QUERY = "support_query"
    FRAUD_REPORT = "fraud_report"
    OTHER = "other"


class InformationSufficiency(str, Enum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class RecommendedHandoff(str, Enum):
    SUPERVISOR = "supervisor"
    HUMAN_ESCALATION = "human_escalation"
    UNSUPPORTED = "unsupported"


class IntakePacket(BaseModel):
    """Structured snapshot of what the intake agent has gathered so far.

    This is intentionally channel-agnostic so both chat and voice can share it.
    """

    model_config = ConfigDict(extra="forbid")

    # High-level intent / eligibility
    intent: IntakeIntent = IntakeIntent.COMPLAINT
    is_financial_complaint: bool = True
    supported_by_platform: bool = True

    # Context
    channel: Literal["web_chat", "voice", "unknown"] = "web_chat"
    customer_summary: str = ""

    # Rough product / issue hints (not the final taxonomy)
    product_hint: Optional[str] = None
    issue_hint: Optional[str] = None
    sub_issue_hint: Optional[str] = None

    # Key facts
    date_of_incident: Optional[str] = None
    amount: Optional[str] = None
    currency: Optional[str] = None
    merchant_or_counterparty: Optional[str] = None
    account_or_reference_available: Optional[bool] = None
    has_supporting_docs: Optional[bool] = None

    prior_contact_attempted: Optional[bool] = None
    desired_resolution: Optional[str] = None

    # Sentiment / urgency
    sentiment: Literal["calm", "frustrated", "angry", "distressed", "unknown"] = "unknown"
    urgency: Literal["low", "medium", "high"] = "medium"
    escalation_reasons: list[str] = Field(default_factory=list)

    # Sufficiency and outcome (filled deterministically by backend)
    missing_fields: list[str] = Field(default_factory=list)
    information_sufficiency: InformationSufficiency = InformationSufficiency.INSUFFICIENT
    recommended_handoff: RecommendedHandoff = RecommendedHandoff.SUPERVISOR

    # Transcript-style narrative that will feed CaseCreate.consumer_narrative
    narrative_for_case: str = ""

    # Case payload that can be used to construct CaseCreate
    intake_case: dict[str, Any] = Field(
        default_factory=dict,
        description="Snapshot of the CaseCreate-compatible payload the engine plans to submit.",
    )


class IntakeSessionState(BaseModel):
    """State for a single multi-turn intake conversation."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    channel: Literal["web_chat", "voice", "unknown"] = "web_chat"

    turn_index: int = 0
    packet: IntakePacket

    last_agent_message: str = ""
    last_user_message: str = ""
    conversation_history: list[dict[str, str]] = Field(default_factory=list)

    completed: bool = False
    handoff_triggered: bool = False
