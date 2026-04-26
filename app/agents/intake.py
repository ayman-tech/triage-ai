"""Intake agent – validates, normalises and enriches the raw complaint."""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime

from app.schemas.case import CaseCreate, CaseRead, CaseStatus
from app.utils.pii import redact_pii

logger = logging.getLogger(__name__)


def _normalise_text(text: str) -> str:
    """Redact PII, lower-case, and collapse whitespace."""
    text = redact_pii(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def run_intake(payload: CaseCreate) -> CaseRead:
    """Execute the intake step and return an enriched case object.

    Responsibilities
    ────────────────
    • Validate required fields (Pydantic handles most of this).
    • Redact obvious PII from the narrative when present.
    • Normalise whitespace and casing.
    • Stamp ``submitted_at`` if missing.
    • Set status → ``intake_complete``.
    • Preserve CFPB portal fields and legacy external labels on the case.
    """
    logger.info("Intake agent processing new complaint")

    raw_narrative = (payload.consumer_narrative or "").strip()
    clean_narrative = _normalise_text(raw_narrative) if raw_narrative else ""

    case = CaseRead(
        id=payload.case_id or uuid.uuid4().hex,
        consumer_narrative=clean_narrative,
        product=payload.product,
        sub_product=payload.sub_product,
        company=payload.company,
        state=payload.state,
        zip_code=payload.zip_code,
        channel=payload.channel,
        submitted_at=payload.submitted_at or datetime.utcnow(),
        status=CaseStatus.INTAKE_COMPLETE,
        cfpb_product=payload.cfpb_product,
        cfpb_sub_product=payload.cfpb_sub_product,
        cfpb_issue=payload.cfpb_issue,
        cfpb_sub_issue=payload.cfpb_sub_issue,
    )

    case.external_schema = {
        "external_product_category": payload.external_product_category,
        "external_issue_type": payload.external_issue_type,
        "requested_resolution": payload.requested_resolution,
        "intake_prior_contact_attempted": payload.intake_prior_contact_attempted,
        "intake_intent": payload.intake_intent,
        "intake_urgency": payload.intake_urgency,
        "intake_recommended_handoff": payload.intake_recommended_handoff,
        "intake_escalation_reasons": payload.intake_escalation_reasons,
        "intake_customer_summary": payload.intake_customer_summary,
        "cfpb_product": payload.cfpb_product,
        "cfpb_sub_product": payload.cfpb_sub_product,
        "cfpb_issue": payload.cfpb_issue,
        "cfpb_sub_issue": payload.cfpb_sub_issue,
        "narrative_absent_or_short": len(clean_narrative) < 10,
    }

    logger.info("Intake complete – case %s", case.id)
    return case
