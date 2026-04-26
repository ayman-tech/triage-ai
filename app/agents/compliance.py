"""Compliance agent – checks for regulatory and policy violations."""

from __future__ import annotations

import logging

from app.agents.adk_runner import run_adk_json_agent
from app.agents.narrative_context import narrative_for_agent_prompt
from app.agents.tools import lookup_severity_rubric
from app.schemas.case import CaseRead
from app.schemas.classification import ClassificationResult
from app.schemas.resolution import ResolutionRecommendation
from app.schemas.risk import RiskAssessment

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a compliance officer reviewing a consumer complaint case that has
already been classified, risk-assessed, and assigned a proposed resolution.

Your job is to flag any **regulatory or policy compliance concerns**.

You have a tool available to look up the company's severity rubric and policy
snippets. Use it to ground your compliance review.

You may consider well-known consumer-protection themes (non-exhaustive examples:
fair credit reporting, debt collection communications, disclosures/notice
adequacy, unfair/deceptive/abusive acts), but do not invent company-specific
rules not present in the retrieved guidance.

Return a JSON object:
{
  "flags": ["<flag_1>", "<flag_2>", ...],
  "passed": true/false,
  "notes": "<optional free-text note>"
}

If no concerns exist, return {"flags": [], "passed": true, "notes": null}.
"""


def run_compliance_check(
    *,
    classification: ClassificationResult,
    risk: RiskAssessment,
    resolution: ResolutionRecommendation | None = None,
    case: CaseRead | None = None,
    narrative: str = "",
    instructions: str = "",
    model_name: str | None = None,
    temperature: float = 0.0,
) -> dict:
    """Run the compliance check and return flags."""
    logger.info("Compliance agent running")

    narrative_text = narrative_for_agent_prompt(case) if case is not None else narrative
    review_hint = ""
    if classification.review_recommended:
        review_hint = (
            "\nNote: Classification has review_recommended=true — apply heightened scrutiny.\n"
        )

    resolution_text = resolution.model_dump_json() if resolution is not None else "Not yet determined"
    user_message = (
        f"Narrative / case text:\n{narrative_text}\n"
        f"{review_hint}"
        f"Classification: {classification.model_dump_json()}\n"
        f"Risk Assessment: {risk.model_dump_json()}\n"
        f"Proposed Resolution: {resolution_text}\n"
    )
    if instructions:
        user_message += f"\nSupervisor instructions: {instructions}\n"

    user_message += (
        "\nUse your tool to look up severity rubrics and policy snippets. "
        "When done, respond with the compliance check JSON."
    )

    tools = [lookup_severity_rubric]
    result = run_adk_json_agent(
        name="compliance_agent",
        description="Reviews classified complaint cases for regulatory and policy compliance concerns.",
        instruction=_SYSTEM_PROMPT,
        user_message=user_message,
        tools=tools,
        model_name=model_name,
        temperature=temperature,
    )

    logger.info(
        "Compliance check complete – passed=%s, flags=%d",
        result.get("passed"),
        len(result.get("flags", [])),
    )
    return result
