"""Root-cause analysis agent – hypothesises the root cause."""

from __future__ import annotations

import logging

from app.agents.adk_runner import run_adk_json_agent
from app.agents.narrative_context import narrative_for_agent_prompt
from app.agents.tools import (
    get_case_document_facts,
    lookup_root_cause_controls,
    search_case_documents,
    search_similar_complaints,
)
from app.schemas.case import CaseRead
from app.schemas.classification import ClassificationResult
from app.schemas.risk import RiskAssessment
from app.schemas.root_cause import RootCauseHypothesis

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a root-cause analyst for a consumer-complaint operations pipeline.

You will be given:
- the complaint narrative and any extracted complaint facts
- the company-aware operational classification
- the company-aware risk assessment

You have tools available to retrieve company control knowledge and search
for similar past complaints. Use them to ground your analysis.

Return a JSON object matching the RootCauseHypothesis schema:
{
  "root_cause_category": "<string>",
  "confidence": <0.0..1.0>,
  "reasoning": "<brief explanation grounded in evidence>",
  "controls_to_check": ["<control_1>", ...],
  "notes": "<optional notes>"
}

Rules:
- Prefer grounding in provided control knowledge and evidence.
- Use business-readable root cause category labels from the retrieved control knowledge,
  not low-level snake_case identifiers.
- If uncertainty remains, use lower confidence and suggest controls_to_check
  that can validate the hypothesis.
"""


def run_root_cause_hypothesis(
    *,
    classification: ClassificationResult,
    risk: RiskAssessment,
    case: CaseRead | None = None,
    narrative: str = "",
    instructions: str = "",
    model_name: str | None = None,
    temperature: float = 0.0,
) -> RootCauseHypothesis:
    """Generate a root cause hypothesis with tool access."""
    logger.info("Root-cause agent running")

    narrative_text = narrative_for_agent_prompt(case) if case is not None else narrative

    user_message = (
        f"Narrative / case text:\n{narrative_text}\n"
        f"Operational classification: {classification.model_dump_json()}\n"
        f"Risk assessment: {risk.model_dump_json()}\n"
    )
    if case is not None and getattr(case, "id", None):
        user_message += f"Case ID: {case.id}\n"
    if instructions:
        user_message += f"\nSupervisor instructions: {instructions}\n"

    user_message += (
        "\nUse your tools to look up root cause controls, extracted case-document facts, "
        "and relevant uploaded-document chunks before finalizing your hypothesis. "
        "When done, respond with the root cause hypothesis JSON."
    )

    tools = [lookup_root_cause_controls, search_similar_complaints, get_case_document_facts, search_case_documents]
    result_data = run_adk_json_agent(
        name="root_cause_agent",
        description="Finds likely complaint root causes using control knowledge and complaint evidence.",
        instruction=_SYSTEM_PROMPT,
        user_message=user_message,
        tools=tools,
        model_name=model_name,
        temperature=temperature,
    )
    result = RootCauseHypothesis(**result_data)

    logger.info(
        "Root-cause complete – category=%s, confidence=%.2f",
        result.root_cause_category,
        result.confidence,
    )
    return result
