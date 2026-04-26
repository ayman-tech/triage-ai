"""Resolution agent – recommends a resolution based on precedent."""

from __future__ import annotations

import logging
from pathlib import Path

from app.agents.narrative_context import narrative_for_agent_prompt
from app.agents.tool_loop import run_agent_with_tools
from app.agents.tools import (
    get_case_document_facts,
    lookup_routing_rules,
    lookup_severity_rubric,
    search_case_documents,
    search_similar_resolutions,
)
from app.schemas.case import CaseRead
from app.schemas.classification import ClassificationResult
from app.schemas.resolution import ResolutionRecommendation
from app.schemas.risk import RiskAssessment

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "resolution.md"


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def run_resolution(
    *,
    classification: ClassificationResult,
    risk: RiskAssessment,
    root_cause_hypothesis: object | None = None,
    case: CaseRead | None = None,
    narrative: str = "",
    instructions: str = "",
    model_name: str | None = None,
    temperature: float = 0.0,
) -> ResolutionRecommendation:
    """Propose a resolution for the complaint."""
    logger.info("Resolution agent running")

    system_prompt = _load_prompt()
    narrative_text = narrative_for_agent_prompt(case) if case is not None else narrative
    review_hint = ""
    if classification.review_recommended:
        review_hint = (
            "\nNote: Classification has review_recommended=true; "
            "prefer conservative, well-documented resolution steps.\n"
        )

    user_message = (
        f"Narrative / case text:\n{narrative_text}\n"
        f"{review_hint}"
        f"Classification: {classification.model_dump_json()}\n"
        f"Risk Assessment: {risk.model_dump_json()}\n"
    )
    if case is not None and getattr(case, "id", None):
        user_message += f"Case ID: {case.id}\n"
    if root_cause_hypothesis is not None:
        user_message += f"Root-cause hypothesis (grounding context): {root_cause_hypothesis}\n"
    if instructions:
        user_message += f"\nSupervisor instructions: {instructions}\n"

    user_message += (
        "\nYou have tools available to search for similar resolutions and look up "
        "severity rubrics, policies, routing rules, and uploaded-document evidence. "
        "Use document facts/chunks whenever documents are attached to the case. "
        "When done, respond with the resolution JSON."
    )

    tools = [search_similar_resolutions, lookup_severity_rubric, lookup_routing_rules, get_case_document_facts, search_case_documents]
    result_data = run_agent_with_tools(system_prompt, user_message, tools)

    _ACTION_ALIASES: dict[str, str] = {
        "investigation": "correction",
        "fraud_investigation": "correction",
        "fraud_investigation_and_remediation": "correction",
        "remediation": "correction",
        "escalation": "referral",
        "escalate": "referral",
        "apology": "non_monetary_relief",
        "refund": "monetary_relief",
        "credit": "monetary_relief",
        "fee_waiver": "monetary_relief",
    }
    _VALID_ACTIONS = {a.value for a in ResolutionRecommendation.model_fields["recommended_action"].annotation}
    raw_action = result_data.get("recommended_action", "")
    if raw_action not in _VALID_ACTIONS:
        mapped = _ACTION_ALIASES.get(raw_action, "correction")
        logger.warning(
            "LLM returned invalid recommended_action %r; mapping to %r",
            raw_action,
            mapped,
        )
        result_data["recommended_action"] = mapped

    result = ResolutionRecommendation(**result_data)

    logger.info(
        "Resolution complete – action=%s, confidence=%.2f",
        result.recommended_action,
        result.confidence,
    )
    return result
