"""Routing agent – determines which team or department handles the case."""

from __future__ import annotations

import logging

from app.schemas.case import CaseRead, CaseStatus
from app.schemas.classification import ClassificationResult, ProductCategory
from app.schemas.risk import RiskAssessment, RiskLevel

logger = logging.getLogger(__name__)

# ── Routing rules ────────────────────────────────────────────────────────────

_PRODUCT_TO_TEAM: dict[ProductCategory, str] = {
    ProductCategory.CREDIT_REPORTING: "credit_reporting_team",
    ProductCategory.DEBT_COLLECTION: "debt_collection_team",
    ProductCategory.MORTGAGE: "mortgage_team",
    ProductCategory.CREDIT_CARD: "credit_card_team",
    ProductCategory.CHECKING_SAVINGS: "banking_team",
    ProductCategory.STUDENT_LOAN: "student_loan_team",
    ProductCategory.VEHICLE_LOAN: "auto_loan_team",
    ProductCategory.PAYDAY_LOAN: "consumer_lending_team",
    ProductCategory.MONEY_TRANSFER: "payments_team",
    ProductCategory.PREPAID_CARD: "payments_team",
    ProductCategory.OTHER: "general_complaints_team",
}


def run_routing(
    case: CaseRead,
    classification: ClassificationResult | None,
    risk: RiskAssessment | None,
    root_cause_hypothesis: object | None = None,
    review_decision: str = "approve",
    company_context: dict | None = None,
) -> str:
    """Determine the destination team for the case.

    Logic
    ─────
    • If the review decision is ``escalate`` → management_escalation_team.
    • If risk_level is critical → executive_complaints_team.
    • Otherwise → map by product_category.

    Notes
    ─────
    Both ``classification`` and ``risk`` may be *None* when the supervisor
    force-routes after repeated upstream agent failures.  Safe fallbacks are
    applied in those cases.
    """
    logger.info("Routing agent running for case %s", case.id)

    routing_candidates = company_context.get("routing_candidates", {}) if company_context else {}
    team_by_product = routing_candidates.get("team_by_product_category", _PRODUCT_TO_TEAM)
    executive_team = routing_candidates.get("executive_team", "executive_complaints_team")
    management_team = routing_candidates.get(
        "management_escalation_team", "management_escalation_team"
    )

    if review_decision == "escalate":
        destination = management_team
    elif risk is not None and risk.risk_level == RiskLevel.CRITICAL:
        destination = executive_team
    elif classification is not None:
        key_str = (
            classification.product_category.value
            if hasattr(classification.product_category, "value")
            else classification.product_category
        )
        destination = team_by_product.get(key_str)
        if destination is None:
            # Fallback for legacy/default mapping where keys are enums.
            destination = team_by_product.get(classification.product_category)  # type: ignore[arg-type]
        if destination is None:
            destination = "general_complaints_team"
    else:
        # classification is None (e.g. supervisor force-routed after repeated failures)
        logger.warning("No classification available for case %s; routing to general team", case.id)
        destination = "general_complaints_team"

    logger.info("Case %s routed to → %s", case.id, destination)
    return destination
