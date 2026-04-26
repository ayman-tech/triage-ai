"""Pydantic models for a consumer‑complaint case."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ── Enums ────────────────────────────────────────────────────────────────────


class CaseStatus(str, Enum):
    RECEIVED = "received"
    INTAKE_COMPLETE = "intake_complete"
    CLASSIFIED = "classified"
    RISK_ASSESSED = "risk_assessed"
    RESOLUTION_PROPOSED = "resolution_proposed"
    COMPLIANCE_CHECKED = "compliance_checked"
    REVIEWED = "reviewed"
    ROUTED = "routed"
    RESOLVED = "resolved"


class Channel(str, Enum):
    WEB = "web"
    PHONE = "phone"
    EMAIL = "email"
    FAX = "fax"
    POSTAL = "postal"
    REFERRAL = "referral"


# ── Core schema ──────────────────────────────────────────────────────────────


class CaseCreate(BaseModel):
    """Payload accepted from the API to open a new case.

    CFPB-style data: ``cfpb_*`` fields are **consumer selections from the portal**
    (noisy priors, not ground truth). ``consumer_narrative`` is optional when
    enough structured portal fields are present.
    """

    model_config = ConfigDict(extra="ignore")

    case_id: Optional[str] = Field(
        None,
        description="Optional preassigned case identifier used when the UI registers a complaint before backend enrichment finishes.",
    )

    consumer_narrative: Optional[str] = Field(
        None,
        description="Free-text complaint narrative (may be absent if CFPB portal fields suffice).",
    )

    product: Optional[str] = Field(
        None,
        description="Financial product or service (legacy / may mirror CFPB product label)",
    )
    sub_product: Optional[str] = None
    company: Optional[str] = None
    state: Optional[str] = Field(None, max_length=2)
    zip_code: Optional[str] = Field(None, max_length=5)
    channel: Channel = Channel.WEB
    submitted_at: Optional[datetime] = None

    # Explicit CFPB portal selections (public complaint database columns).
    cfpb_product: Optional[str] = Field(
        None, description="Consumer-selected CFPB product (portal)"
    )
    cfpb_sub_product: Optional[str] = Field(
        None, description="Consumer-selected CFPB sub-product (portal)"
    )
    cfpb_issue: Optional[str] = Field(
        None, description="Consumer-selected CFPB issue (portal)"
    )
    cfpb_sub_issue: Optional[str] = Field(
        None, description="Consumer-selected CFPB sub-issue (portal)"
    )

    # Legacy external labels (still supported for CSV/API compatibility).
    external_product_category: Optional[str] = Field(
        None, description="Optional externally provided product category"
    )
    external_issue_type: Optional[str] = Field(
        None, description="Optional externally provided issue type"
    )
    requested_resolution: Optional[str] = Field(
        None, description="Optional externally requested resolution"
    )
    intake_prior_contact_attempted: Optional[bool] = Field(
        None, description="Whether the customer says they already reported this issue to the bank."
    )
    intake_intent: Optional[str] = Field(
        None, description="Optional intake-stage intent label from the interactive intake agent"
    )
    intake_urgency: Optional[str] = Field(
        None, description="Optional intake-stage urgency level from the interactive intake agent"
    )
    intake_recommended_handoff: Optional[str] = Field(
        None, description="Optional intake-stage handoff recommendation"
    )
    intake_escalation_reasons: Optional[list[str]] = Field(
        None, description="Optional intake-stage escalation reason codes"
    )
    intake_customer_summary: Optional[str] = Field(
        None, description="Optional intake-stage summary captured during interactive intake"
    )

    @model_validator(mode="after")
    def require_narrative_or_structured_path(self) -> CaseCreate:
        """At least one of: rich narrative, or minimal CFPB/legacy structured path."""
        nar = (self.consumer_narrative or "").strip()
        has_rich_narrative = len(nar) >= 10

        has_cfpb_core = bool(self.cfpb_product and self.cfpb_issue)
        has_legacy_structured = bool(
            self.product and (self.cfpb_issue or self.external_issue_type)
        )
        has_structured = has_cfpb_core or has_legacy_structured

        if not has_rich_narrative and not has_structured:
            raise ValueError(
                "Provide either consumer_narrative (at least 10 characters) or "
                "structured portal fields: (cfpb_product AND cfpb_issue), OR "
                "(product AND (cfpb_issue OR external_issue_type))."
            )
        return self


class CaseRead(BaseModel):
    """Full case representation returned by the API."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    public_case_id: Optional[str] = None
    status: CaseStatus = CaseStatus.RECEIVED
    consumer_narrative: str = ""
    product: Optional[str] = None
    sub_product: Optional[str] = None
    company: Optional[str] = None
    user_id: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    channel: Channel = Channel.WEB
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # CFPB portal fields (copied from intake payload for agent access).
    cfpb_product: Optional[str] = None
    cfpb_sub_product: Optional[str] = None
    cfpb_issue: Optional[str] = None
    cfpb_sub_issue: Optional[str] = None

    # Downstream enrichment (populated by agents)
    classification: Optional[dict] = None
    classification_audit: Optional[dict] = None
    risk_assessment: Optional[dict] = None
    proposed_resolution: Optional[dict] = None
    compliance_flags: Optional[list[str]] = None
    review_notes: Optional[str] = None
    routed_to: Optional[str] = None

    # New company-aware fields (stored as JSON/dicts for flexibility).
    external_schema: Optional[dict] = None
    operational_mapping: Optional[dict] = None
    evidence_trace: Optional[dict] = None
    severity_class: Optional[str] = None
    team_assignment: Optional[str] = None
    sla_class: Optional[str] = None
    root_cause_hypothesis: Optional[dict] = None
    case_documents: list[dict[str, Any]] = Field(default_factory=list)
    case_document_summary: Optional[dict[str, Any]] = None
    document_gate_result: Optional[dict[str, Any]] = None
    document_consistency: Optional[dict[str, Any]] = None

    # Jira integration (populated after routing)
    jira_issue_key: Optional[str] = None   # e.g. "KAN-7"
    jira_issue_url: Optional[str] = None   # direct browser URL

    class Config:
        from_attributes = True
