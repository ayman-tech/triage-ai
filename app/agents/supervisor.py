"""Supervisor agent — decides which specialist to invoke next.

The supervisor reads the current WorkflowState, reasons about what has been
accomplished and what remains, and returns a SupervisorDecision indicating
the next specialist node to run (or FINISH to end the workflow).
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import google.genai as genai
from google.genai import types as genai_types
from pydantic import BaseModel, ValidationError

from app.agents.llm_factory import default_model_name, get_gemini_client
from app.agents.llm_json import parse_llm_json
from app.agents.tool_loop import record_llm_usage

if TYPE_CHECKING:
    from app.orchestrator.state import WorkflowState

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "supervisor.md"

_VALID_AGENTS = frozenset(
    {"classify", "risk", "root_cause", "resolve", "check_compliance", "qa_review", "route", "FINISH"}
)

DEFAULT_MAX_STEPS = 15
MAX_AGENT_INVOCATIONS = 3


# ── Supervisor decision schema ───────────────────────────────────────────────

class SupervisorDecision(BaseModel):
    next_agent: Literal[
        "classify", "risk", "root_cause", "resolve",
        "check_compliance", "qa_review", "route", "FINISH",
    ]
    reasoning: str
    instructions: str = ""


# ── State summariser ─────────────────────────────────────────────────────────

def _build_state_summary(state: dict[str, Any]) -> str:
    parts: list[str] = []

    case = state.get("case")
    if case:
        narrative = getattr(case, "consumer_narrative", None) or ""
        parts.append(f"Case narrative (first 300 chars): {narrative[:300]}")
        if len(narrative.strip()) < 10:
            parts.append(
                "Consumer narrative is absent or short — specialists use portal fields "
                "and classification; do not assume rich free-text detail."
            )
        parts.append(f"Product hint: {getattr(case, 'product', None) or 'N/A'}")
    else:
        parts.append("Case: not yet ingested")

    completed = state.get("completed_steps", [])
    parts.append(f"Completed steps: {completed if completed else 'none'}")
    parts.append(f"Step count: {state.get('step_count', 0)}/{state.get('max_steps', DEFAULT_MAX_STEPS)}")

    cls = state.get("classification")
    if cls:
        cat = getattr(cls, "product_category", None)
        cat_val = getattr(cat, "value", str(cat)) if cat else "N/A"
        issue = getattr(cls, "issue_type", None)
        issue_val = getattr(issue, "value", str(issue)) if issue else "N/A"
        conf = getattr(cls, "confidence", None)
        parts.append(
            f"Classification: product={cat_val}, issue={issue_val}, "
            f"confidence={conf:.2f}" if conf is not None else f"Classification: product={cat_val}, issue={issue_val}"
        )
        rr = getattr(cls, "review_recommended", False)
        if rr:
            rc = getattr(cls, "reason_codes", []) or []
            parts.append(f"Classification review_recommended=true; reason_codes={list(rc)}")

    risk = state.get("risk_assessment")
    if risk:
        level = getattr(risk, "risk_level", None)
        level_val = getattr(level, "value", str(level)) if level else "N/A"
        score = getattr(risk, "risk_score", None)
        reg = getattr(risk, "regulatory_risk", None)
        parts.append(f"Risk: level={level_val}, score={score}, regulatory_risk={reg}")

    rc_hyp = state.get("root_cause_hypothesis")
    if rc_hyp:
        cat = getattr(rc_hyp, "root_cause_category", None)
        conf = getattr(rc_hyp, "confidence", None)
        parts.append(f"Root cause: {cat} (confidence={conf})")

    res = state.get("resolution")
    if res:
        action = getattr(res, "recommended_action", None)
        action_val = getattr(action, "value", str(action)) if action else "N/A"
        conf = getattr(res, "confidence", None)
        parts.append(f"Resolution: action={action_val}, confidence={conf}")

    comp = state.get("compliance")
    if comp:
        parts.append(f"Compliance: passed={comp.get('passed')}, flags={comp.get('flags', [])}")

    rev = state.get("review")
    if rev:
        parts.append(f"Review: decision={rev.get('decision')}, notes={rev.get('notes', '')}")

    feedback = state.get("review_feedback")
    if feedback:
        parts.append(f"Review feedback: {feedback}")

    routed = state.get("routed_to")
    if routed:
        parts.append(f"Routed to: {routed}")

    return "\n".join(parts)


# ── Safe fallback ─────────────────────────────────────────────────────────────

def _fallback_decision(
    state: dict[str, Any],
    error: Exception,
    step_count: int,
) -> SupervisorDecision:
    completed = state.get("completed_steps", [])
    next_agent = "route" if "route" not in completed else "FINISH"
    logger.error(
        "Supervisor failed to parse LLM response (%s: %s); falling back to %s",
        type(error).__name__,
        str(error)[:300],
        next_agent,
    )
    return SupervisorDecision(
        next_agent=next_agent,  # type: ignore[arg-type]
        reasoning=f"Fallback: LLM response parsing failed ({type(error).__name__})",
        instructions="",
    )


# ── Supervisor runner ─────────────────────────────────────────────────────────

def run_supervisor(state: dict[str, Any]) -> SupervisorDecision:
    """Decide the next specialist agent to invoke.

    Returns a SupervisorDecision with next_agent set to the chosen agent
    name (or 'FINISH') plus reasoning and instructions for the orchestrator.
    """
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", DEFAULT_MAX_STEPS)
    completed_steps = list(state.get("completed_steps", []))

    # Safety: force finish if we've hit the step limit
    if step_count >= max_steps:
        logger.warning("Supervisor hit max_steps (%d), forcing route/FINISH", max_steps)
        next_agent = "route" if "route" not in completed_steps else "FINISH"
        return SupervisorDecision(
            next_agent=next_agent,  # type: ignore[arg-type]
            reasoning="Max steps reached.",
            instructions="",
        )

    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
    state_summary = _build_state_summary(state)

    client = get_gemini_client()
    model = default_model_name()

    contents = [
        genai_types.Content(role="user", parts=[genai_types.Part(text=state_summary)])
    ]
    config = genai_types.GenerateContentConfig(system_instruction=system_prompt)

    started_at = datetime.utcnow()
    response = client.models.generate_content(model=model, contents=contents, config=config)
    ended_at = datetime.utcnow()
    record_llm_usage(response, model_name=model, started_at=started_at, ended_at=ended_at)

    try:
        result = parse_llm_json(response.text or "")
        decision = SupervisorDecision(**result)
    except (ValueError, TypeError, KeyError, ValidationError) as exc:
        return _fallback_decision(state, exc, step_count)

    logger.info(
        "Supervisor decision: next=%s, reasoning=%s",
        decision.next_agent,
        decision.reasoning,
    )

    # Validate the decision
    if decision.next_agent not in _VALID_AGENTS:
        logger.error("Supervisor returned invalid agent: %s, defaulting to route", decision.next_agent)
        decision = SupervisorDecision(
            next_agent="route",
            reasoning=decision.reasoning,
            instructions=decision.instructions,
        )

    # Enforce max invocations per agent (prevent infinite loops)
    if decision.next_agent != "FINISH":
        agent_counts = Counter(completed_steps)
        if agent_counts[decision.next_agent] >= MAX_AGENT_INVOCATIONS:
            logger.warning(
                "Agent %s already invoked %d times (max %d), forcing route/FINISH",
                decision.next_agent,
                agent_counts[decision.next_agent],
                MAX_AGENT_INVOCATIONS,
            )
            next_forced = "route" if "route" not in completed_steps else "FINISH"
            decision = SupervisorDecision(
                next_agent=next_forced,  # type: ignore[arg-type]
                reasoning=f"Agent {decision.next_agent} hit max invocations.",
                instructions="",
            )

    return decision
