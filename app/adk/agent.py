"""Google ADK root agent and sub-agent graph.

This module is intentionally separate from the FastAPI workflow.  The web app
continues to use ``app.orchestrator.workflow.process_complaint`` while this
module exposes an ADK-native ``root_agent`` for ADK tooling and Vertex AI Agent
Engine deployment.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.agents.adk_runner import make_llm_agent
from app.agents.tools import (
    get_case_document_facts,
    lookup_company_taxonomy,
    lookup_root_cause_controls,
    lookup_routing_rules,
    lookup_severity_rubric,
    search_case_documents,
    search_similar_complaints,
    search_similar_resolutions,
)
from app.agents.compliance import _SYSTEM_PROMPT as COMPLIANCE_PROMPT
from app.agents.root_cause import _SYSTEM_PROMPT as ROOT_CAUSE_PROMPT
from app.agents.review import _SYSTEM_PROMPT as REVIEW_PROMPT
from app.agents.google_tools import search_google_policy_knowledge


_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _prompt(name: str) -> str:
    return (_PROMPT_DIR / name).read_text(encoding="utf-8")


def _workflow_agent_imports():
    from google.adk.agents import LoopAgent, ParallelAgent, SequentialAgent

    return LoopAgent, ParallelAgent, SequentialAgent


def _google_search_tools_enabled() -> bool:
    return os.getenv("ENABLE_VERTEX_AI_SEARCH_TOOLS", "").lower() in ("1", "true", "yes")


def _policy_tools() -> list:
    if _google_search_tools_enabled():
        return [lookup_severity_rubric, search_google_policy_knowledge]
    return [lookup_severity_rubric]


intake_agent = make_llm_agent(
    name="intake_agent",
    description="Conversational complaint intake agent for gathering safe complaint facts.",
    instruction=_prompt("intake_chat.md"),
    output_key="intake_packet_json",
)

classification_agent = make_llm_agent(
    name="classification_agent",
    description="Classifies complaint product, issue, confidence, and review signals.",
    instruction=_prompt("classification.md"),
    tools=[
        search_similar_complaints,
        lookup_company_taxonomy,
        get_case_document_facts,
        search_case_documents,
    ],
    output_key="classification_json",
)

risk_agent = make_llm_agent(
    name="risk_agent",
    description="Assesses complaint severity, regulatory exposure, and escalation need.",
    instruction=_prompt("risk.md"),
    tools=[
        search_similar_complaints,
        *_policy_tools(),
        get_case_document_facts,
        search_case_documents,
    ],
    output_key="risk_json",
)

root_cause_agent = make_llm_agent(
    name="root_cause_agent",
    description="Hypothesizes operational root cause using control and complaint evidence.",
    instruction=ROOT_CAUSE_PROMPT,
    tools=[
        lookup_root_cause_controls,
        search_similar_complaints,
        get_case_document_facts,
        search_case_documents,
    ],
    output_key="root_cause_json",
)

resolution_agent = make_llm_agent(
    name="resolution_agent",
    description="Recommends complaint resolution grounded in policy and precedent.",
    instruction=_prompt("resolution.md"),
    tools=[
        search_similar_resolutions,
        *_policy_tools(),
        lookup_routing_rules,
        get_case_document_facts,
        search_case_documents,
    ],
    output_key="resolution_json",
)

compliance_agent = make_llm_agent(
    name="compliance_agent",
    description="Reviews regulatory and internal policy concerns.",
    instruction=COMPLIANCE_PROMPT,
    tools=_policy_tools(),
    output_key="compliance_json",
)

review_agent = make_llm_agent(
    name="review_agent",
    description="Performs final QA review and requests revisions when needed.",
    instruction=REVIEW_PROMPT,
    output_key="review_json",
)

LoopAgent, ParallelAgent, SequentialAgent = _workflow_agent_imports()

risk_root_cause_parallel_agent = ParallelAgent(
    name="risk_root_cause_parallel",
    description="Runs independent risk and root-cause specialist analysis concurrently.",
    sub_agents=[risk_agent, root_cause_agent],
)

review_revision_loop_agent = LoopAgent(
    name="review_revision_loop",
    description=(
        "Iterates resolution, compliance, and review to demonstrate the ADK "
        "revision loop used by the production supervisor."
    ),
    sub_agents=[resolution_agent, compliance_agent, review_agent],
    max_iterations=2,
)

complaint_pipeline_agent = SequentialAgent(
    name="complaint_pipeline_agent",
    description="ADK-native sequential complaint processing pipeline.",
    sub_agents=[
        classification_agent,
        risk_root_cause_parallel_agent,
        review_revision_loop_agent,
    ],
)

root_agent = make_llm_agent(
    name="complaint_supervisor_agent",
    description="Root supervisor for the TriageAI multi-agent complaint operating model.",
    instruction=(
        "You are the root supervisor for a regulated complaint management system. "
        "Coordinate intake, classification, risk, root-cause, resolution, compliance, "
        "and review specialists. Prefer grounded tool use and structured JSON outputs. "
        "For end-to-end complaint processing, delegate to complaint_pipeline_agent."
    ),
    sub_agents=[
        intake_agent,
        complaint_pipeline_agent,
    ],
    output_key="root_supervisor_response",
)
