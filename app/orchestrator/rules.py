"""Business rules and conditional edges for the workflow graph."""

from __future__ import annotations

import logging

from app.observability.events import log_workflow_event
from app.orchestrator.state import WorkflowState

logger = logging.getLogger(__name__)

# ── Maximum retries before the workflow gives up ─────────────────────────────
MAX_RETRIES = 2


def should_escalate(state: WorkflowState) -> bool:
    """Return *True* if the risk level demands immediate escalation."""
    risk = state.get("risk_assessment")
    if risk is None:
        return False
    return risk.risk_level.value == "critical"


def needs_compliance_review(state: WorkflowState) -> bool:
    """Return *True* if the case must pass through compliance."""
    risk = state.get("risk_assessment")
    if risk is None:
        return True  # err on the side of caution
    return risk.regulatory_risk or risk.risk_level.value in ("high", "critical")


def review_decision_router(state: WorkflowState) -> str:
    """Return the next node name based on the review agent's decision.

    Possible returns
    ────────────────
    * ``"route"``   – proceed to routing (approved).
    * ``"revise"``  – loop back to the resolution agent.
    * ``"escalate"``– jump to routing with escalation flag.
    """
    review = state.get("review", {})
    decision = review.get("decision", "approve")

    retry_count = state.get("retry_count", 0)
    if decision == "revise" and retry_count < MAX_RETRIES:
        logger.info("Review requested revision (attempt %d)", retry_count + 1)
        log_workflow_event(
            "retry_triggered",
            node_name="review_gate",
            trigger_reason="review_revise",
            retry_number=retry_count + 1,
        )
        return "revise"
    if decision == "escalate":
        logger.warning("Review requested escalation")
        log_workflow_event(
            "retry_triggered",
            node_name="review_gate",
            trigger_reason="review_escalate",
            retry_number=retry_count,
        )
        return "escalate"

    return "route"


def low_confidence_gate(state: WorkflowState) -> str:
    """If classification confidence is below threshold, re‑classify.

    Returns ``"continue"`` or ``"reclassify"``.
    """
    classification = state.get("classification")
    if classification is None:
        return "continue"

    if classification.confidence < 0.6:
        retry = state.get("retry_count", 0)
        if retry < MAX_RETRIES:
            logger.info("Low confidence (%.2f) – reclassifying", classification.confidence)
            log_workflow_event(
                "retry_triggered",
                node_name="classify",
                trigger_reason="low_confidence",
                retry_number=retry + 1,
            )
            return "reclassify"

    return "continue"
