"""Risk-assessment agent – evaluates complaint risk level."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.agents.adk_runner import run_adk_json_agent
from app.agents.narrative_context import narrative_for_agent_prompt
from app.agents.tools import (
    get_case_document_facts,
    lookup_severity_rubric,
    search_case_documents,
    search_similar_complaints,
)
from app.schemas.case import CaseRead
from app.schemas.classification import ClassificationResult
from app.schemas.risk import RiskAssessment

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "risk.md"


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _normalize_factor_weights(result_data: dict[str, Any]) -> dict[str, Any]:
    """Accept percentage-style LLM weights while preserving schema semantics."""
    factors = result_data.get("factors")
    if not isinstance(factors, list):
        return result_data

    for factor in factors:
        if not isinstance(factor, dict):
            continue
        raw_weight = factor.get("weight")
        if isinstance(raw_weight, str):
            raw_weight = raw_weight.strip().removesuffix("%").strip()
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            continue
        if 1.0 < weight <= 100.0:
            factor["weight"] = weight / 100.0

    return result_data


def run_risk_assessment(
    *,
    classification: ClassificationResult,
    case: CaseRead | None = None,
    narrative: str = "",
    instructions: str = "",
    model_name: str | None = None,
    temperature: float = 0.0,
) -> RiskAssessment:
    """Assess the risk posed by the complaint."""
    logger.info("Risk agent running")

    system_prompt = _load_prompt()
    narrative_text = narrative_for_agent_prompt(case) if case is not None else narrative
    review_hint = ""
    if classification.review_recommended:
        review_hint = (
            "\nNote: Classification has review_recommended=true; "
            f"reason_codes={classification.reason_codes}. Treat regulatory sensitivity carefully.\n"
        )

    user_message = (
        f"Narrative / case text:\n{narrative_text}\n"
        f"{review_hint}"
        f"Classification: {classification.model_dump_json()}\n"
    )
    if case is not None and getattr(case, "id", None):
        user_message += f"Case ID: {case.id}\n"
    if instructions:
        user_message += f"\nSupervisor instructions: {instructions}\n"

    user_message += (
        "\nYou have tools available to search for similar complaints and look up "
        "severity rubrics, document facts, and uploaded case-document content. "
        "If documents are attached, use them to ground your risk assessment. "
        "When done, respond with the risk assessment JSON."
    )

    tools = [search_similar_complaints, lookup_severity_rubric, get_case_document_facts, search_case_documents]
    result_data = run_adk_json_agent(
        name="risk_agent",
        description="Assesses complaint severity, regulatory exposure, escalation need, and financial impact.",
        instruction=system_prompt,
        user_message=user_message,
        tools=tools,
        model_name=model_name,
        temperature=temperature,
    )
    result = RiskAssessment(**_normalize_factor_weights(result_data))

    logger.info(
        "Risk assessment complete – level=%s, score=%.1f",
        result.risk_level,
        result.risk_score,
    )
    return result
