"""Agentic workflow — supervisor-driven complaint processing.

Hub-and-spoke architecture: the supervisor decides which specialist agent to
invoke next.  The workflow is a plain Python loop (no LangGraph); the
supervisor returns a SupervisorDecision and the orchestrator dispatches
accordingly.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Callable

from opentelemetry.trace import Status, StatusCode

from app.agents.classification import run_classification
from app.agents.narrative_context import narrative_for_agent_prompt
from app.agents.compliance import run_compliance_check
from app.agents.intake import run_intake
from app.agents.resolution import run_resolution
from app.agents.review import run_review
from app.agents.risk import run_risk_assessment
from app.agents.root_cause import run_root_cause_hypothesis
from app.agents.routing import run_routing
from app.agents.supervisor import run_supervisor
from app.integrations.jira_client import create_complaint_ticket
from app.documents.service import (
    build_case_document_summary,
    compare_case_to_documents,
    list_case_documents,
    wait_for_case_documents,
)
from app.knowledge.mock_company_pack import deployment_label
from app.observability.context import (
    ActiveRun,
    reset_active_run,
    set_active_run,
    set_trace_id,
)
from app.observability.cost import TokenCostAccumulator
from app.observability.events import log_workflow_event
from app.observability.instrumentation import wrap_node, wrap_supervisor_node
from app.observability.persistence import (
    derive_run_outcome,
    finalize_workflow_run,
    insert_workflow_run,
)
from app.observability.tracing import get_workflow_tracer, setup_tracing, trace_id_hex_from_span
from app.observability.versions import workflow_version
from app.orchestrator.state import WorkflowState
from app.schemas.case import CaseCreate, CaseStatus

logger = logging.getLogger(__name__)


# ── Node functions ────────────────────────────────────────────────────────────

def intake_node(state: WorkflowState) -> WorkflowState:
    """Deterministic intake: PII redaction, validation, normalisation."""
    payload = CaseCreate(**state["raw_payload"])
    case = run_intake(payload)
    return {
        **state,
        "case": case,
        "document_gate_result": {"required": False, "status": "not_run"},
        "document_consistency": {"status": "not_run", "conflicts": [], "verified_facts": {}},
        "completed_steps": [],
        "step_count": 0,
        "max_steps": state.get("max_steps", 15),
    }


def document_gate_node(state: WorkflowState) -> WorkflowState:
    """Wait for attached documents to finish background processing."""
    case = state["case"]
    if not case.id:
        return state

    gate_result = wait_for_case_documents(case.id)
    case.case_documents = [doc.model_dump() for doc in list_case_documents(case.id)] if gate_result.get("required") else []
    case.case_document_summary = build_case_document_summary(case.id).model_dump()
    case.document_gate_result = gate_result
    return {**state, "case": case, "document_gate_result": gate_result}


def document_consistency_node(state: WorkflowState) -> WorkflowState:
    """Deterministically compare claimant narrative with extracted document facts."""
    case = state["case"]
    doc_summary = case.case_document_summary or {}
    consistency = compare_case_to_documents(
        narrative_text=case.consumer_narrative or "",
        document_summary=doc_summary,
    )
    case.document_consistency = consistency
    return {**state, "case": case, "document_consistency": consistency}


def classify_node(state: WorkflowState) -> WorkflowState:
    """Classification specialist with tool access."""
    case = state["case"]
    instructions = state.get("supervisor_instructions", "")

    pipeline_out = run_classification(case=case, instructions=instructions)
    result = pipeline_out.result

    case.classification = result.model_dump()
    case.classification_audit = pipeline_out.audit.model_dump(mode="json")
    case.status = CaseStatus.CLASSIFIED
    case.operational_mapping = {
        "product_category": result.product_category.value,
        "sub_product": result.sub_product,
        "issue_type": result.issue_type.value,
        "sub_issue": result.sub_issue,
    }

    completed = list(state.get("completed_steps", []))
    completed.append("classify")
    return {**state, "case": case, "classification": result, "completed_steps": completed}


def risk_node(state: WorkflowState) -> WorkflowState:
    """Risk assessment specialist with tool access."""
    case = state["case"]
    instructions = state.get("supervisor_instructions", "")

    result = run_risk_assessment(
        case=case,
        classification=state["classification"],
        instructions=instructions,
    )

    case.risk_assessment = result.model_dump()
    case.severity_class = result.risk_level.value
    case.status = CaseStatus.RISK_ASSESSED

    completed = list(state.get("completed_steps", []))
    completed.append("risk")
    return {**state, "case": case, "risk_assessment": result, "completed_steps": completed}


def root_cause_node(state: WorkflowState) -> WorkflowState:
    """Root cause hypothesis specialist with tool access."""
    case = state["case"]
    instructions = state.get("supervisor_instructions", "")

    result = run_root_cause_hypothesis(
        case=case,
        classification=state["classification"],
        risk=state["risk_assessment"],
        instructions=instructions,
    )

    case.root_cause_hypothesis = result.model_dump()

    completed = list(state.get("completed_steps", []))
    completed.append("root_cause")
    return {**state, "case": case, "root_cause_hypothesis": result, "completed_steps": completed}


def resolution_node(state: WorkflowState) -> WorkflowState:
    """Resolution specialist with tool access."""
    case = state["case"]
    instructions = state.get("supervisor_instructions", "")

    result = run_resolution(
        case=case,
        classification=state["classification"],
        risk=state["risk_assessment"],
        root_cause_hypothesis=state.get("root_cause_hypothesis"),
        instructions=instructions,
    )

    case.proposed_resolution = result.model_dump()
    case.status = CaseStatus.RESOLUTION_PROPOSED

    completed = list(state.get("completed_steps", []))
    completed.append("resolve")
    return {**state, "case": case, "resolution": result, "completed_steps": completed}


def compliance_node(state: WorkflowState) -> WorkflowState:
    """Compliance specialist with tool access."""
    case = state["case"]
    instructions = state.get("supervisor_instructions", "")

    result = run_compliance_check(
        case=case,
        classification=state["classification"],
        risk=state["risk_assessment"],
        resolution=state.get("resolution"),
        instructions=instructions,
    )

    case.compliance_flags = result.get("flags", [])
    case.status = CaseStatus.COMPLIANCE_CHECKED

    completed = list(state.get("completed_steps", []))
    completed.append("check_compliance")
    return {**state, "case": case, "compliance": result, "completed_steps": completed}


def review_node(state: WorkflowState) -> WorkflowState:
    """Review specialist — QA pass with structured feedback."""
    case = state["case"]
    instructions = state.get("supervisor_instructions", "")

    result = run_review(
        narrative=narrative_for_agent_prompt(case),
        classification_json=json.dumps(state["classification"].model_dump()),
        risk_json=json.dumps(state["risk_assessment"].model_dump()),
        resolution_json=json.dumps(state["resolution"].model_dump()),
        compliance_json=json.dumps(state.get("compliance", {})),
        instructions=instructions,
    )

    case.review_notes = result.get("notes", "")
    case.status = CaseStatus.REVIEWED

    completed = list(state.get("completed_steps", []))
    completed.append("qa_review")

    update: dict = {**state, "case": case, "review": result, "completed_steps": completed}

    if result.get("decision") == "revise" and result.get("review_feedback"):
        update["review_feedback"] = result["review_feedback"]

    return update


def routing_node(state: WorkflowState) -> WorkflowState:
    """Deterministic routing based on company knowledge pack rules."""
    from app.agents.tools import _company_knowledge_service

    case = state["case"]

    company_context = None
    try:
        svc = _company_knowledge_service()
        ctx = svc.build_company_context("")
        company_context = {"routing_candidates": ctx.routing_candidates}
    except Exception:
        logger.warning("Could not load routing rules from knowledge pack; using defaults")

    destination = run_routing(
        case=case,
        classification=state.get("classification"),
        risk=state.get("risk_assessment"),
        root_cause_hypothesis=state.get("root_cause_hypothesis"),
        review_decision=state.get("review", {}).get("decision", "approve"),
        company_context=company_context,
    )

    case.routed_to = destination
    case.team_assignment = destination
    case.status = CaseStatus.ROUTED

    completed = list(state.get("completed_steps", []))
    completed.append("route")

    # ── Jira integration ─────────────────────────────────────────────────────
    jira_ticket: dict = {}
    try:
        classification = state.get("classification")
        risk = state.get("risk_assessment")
        resolution = state.get("resolution")
        root_cause = state.get("root_cause_hypothesis")
        compliance = state.get("compliance", {})

        product_category = (
            classification.product_category.value
            if classification and hasattr(classification, "product_category") else None
        )
        issue_type = (
            classification.issue_type.value
            if classification and hasattr(classification, "issue_type") else None
        )
        classification_reasoning = getattr(classification, "reasoning", None) if classification else None

        risk_level = (
            risk.risk_level.value if risk and hasattr(risk, "risk_level") else None
        )
        risk_score = getattr(risk, "risk_score", None) if risk else None
        risk_reasoning = getattr(risk, "reasoning", None) if risk else None
        regulatory_risk = getattr(risk, "regulatory_risk", False) if risk else False
        financial_impact = getattr(risk, "financial_impact_estimate", None) if risk else None

        resolution_action = (
            resolution.recommended_action.value
            if resolution and hasattr(resolution, "recommended_action") else None
        )
        resolution_description = getattr(resolution, "description", None) if resolution else None
        resolution_reasoning = getattr(resolution, "reasoning", None) if resolution else None
        estimated_days = getattr(resolution, "estimated_resolution_days", None) if resolution else None
        monetary_amount = getattr(resolution, "monetary_amount", None) if resolution else None

        root_cause_category = getattr(root_cause, "root_cause_category", None) if root_cause else None
        root_cause_reasoning = getattr(root_cause, "reasoning", None) if root_cause else None
        controls_to_check = getattr(root_cause, "controls_to_check", None) if root_cause else None

        compliance_flags = compliance.get("flags", []) if isinstance(compliance, dict) else []

        jira_ticket = create_complaint_ticket(
            case_id=case.id,
            team=destination,
            product_category=product_category,
            issue_type=issue_type,
            risk_level=risk_level,
            risk_score=risk_score,
            risk_reasoning=risk_reasoning,
            regulatory_risk=regulatory_risk,
            financial_impact=financial_impact,
            channel=case.channel.value if case.channel else None,
            consumer_narrative=case.consumer_narrative,
            resolution_action=resolution_action,
            resolution_description=resolution_description,
            resolution_reasoning=resolution_reasoning,
            estimated_resolution_days=estimated_days,
            monetary_amount=monetary_amount,
            root_cause_category=root_cause_category,
            root_cause_reasoning=root_cause_reasoning,
            controls_to_check=controls_to_check,
            compliance_flags=compliance_flags if compliance_flags else None,
            classification_reasoning=classification_reasoning,
            company=case.company,
            state=case.state,
        )
        case.jira_issue_key = jira_ticket.get("key")
        case.jira_issue_url = jira_ticket.get("url")
        logger.info(
            "Jira ticket %s created for case %s → team %s",
            jira_ticket.get("key"),
            case.id,
            destination,
        )
    except Exception as exc:
        logger.warning(
            "Jira ticket creation failed for case %s (team=%s): %s",
            case.id,
            destination,
            exc,
        )

    return {
        **state,
        "case": case,
        "routed_to": destination,
        "completed_steps": completed,
        "jira_ticket": jira_ticket,
    }


# ── Agent dispatch map ────────────────────────────────────────────────────────

_AGENT_MAP: dict[str, Callable[[WorkflowState], WorkflowState]] = {
    "classify": wrap_node("classify", classify_node),
    "risk": wrap_node("risk", risk_node),
    "root_cause": wrap_node("root_cause", root_cause_node),
    "resolve": wrap_node("resolve", resolution_node),
    "check_compliance": wrap_node("check_compliance", compliance_node),
    "qa_review": wrap_node("qa_review", review_node),
    "route": wrap_node("route", routing_node),
}

_wrapped_intake = wrap_node("intake", intake_node)
_wrapped_doc_gate = wrap_node("document_gate", document_gate_node)
_wrapped_doc_consistency = wrap_node("check_document_consistency", document_consistency_node)
_wrapped_supervisor = wrap_supervisor_node(lambda state: run_supervisor(state))


# ── Main entry point ──────────────────────────────────────────────────────────

def process_complaint(payload: dict) -> WorkflowState:
    """Run the full agentic complaint pipeline and return the final state."""
    setup_tracing()

    run_id = uuid.uuid4().hex
    log_label = deployment_label()
    ar = ActiveRun(run_id=run_id, company_id=log_label)
    ctx_token = set_active_run(ar)
    tracer = get_workflow_tracer()
    cost_acc = TokenCostAccumulator()

    state: WorkflowState = {"raw_payload": payload, "retry_count": 0}

    try:
        with tracer.start_as_current_span("process_complaint") as root:
            tid = trace_id_hex_from_span(root)
            if tid:
                set_trace_id(tid)
            root.set_attribute("complaint.run_id", run_id)
            root.set_attribute("complaint.deployment", log_label)

            log_workflow_event(
                "workflow_started",
                run_id=run_id,
                company_id=log_label,
                trace_id=tid or "",
            )
            insert_workflow_run(run_id, tid or None)

            try:
                # ── Deterministic entry sequence ──────────────────────────
                state = _wrapped_intake(state)
                state = _wrapped_doc_gate(state)
                state = _wrapped_doc_consistency(state)

                # ── Supervisor loop ───────────────────────────────────────
                while True:
                    decision = _wrapped_supervisor(state)
                    state["step_count"] = state.get("step_count", 0) + 1
                    state["supervisor_reasoning"] = decision.reasoning
                    state["supervisor_instructions"] = decision.instructions

                    if decision.next_agent == "FINISH":
                        logger.info("Supervisor decided FINISH — workflow complete")
                        break

                    node_fn = _AGENT_MAP.get(decision.next_agent)
                    if node_fn is None:
                        logger.error(
                            "Unknown agent '%s' from supervisor; ending workflow",
                            decision.next_agent,
                        )
                        break

                    state = node_fn(state)

            except Exception as exc:
                root.record_exception(exc)
                root.set_status(Status(StatusCode.ERROR, str(exc)))
                log_workflow_event(
                    "workflow_failed",
                    run_id=run_id,
                    node_name="process_complaint",
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:500],
                )
                finalize_workflow_run(
                    run_id,
                    run_status="failed",
                    final_route=None,
                    final_severity=None,
                    manual_review_required=False,
                    retry_count_total=int(state.get("retry_count") or 0),
                    token_total=cost_acc.total_tokens or None,
                    cost_estimate_usd=None,
                )
                raise

            root.set_status(Status(StatusCode.OK))

        run_status, route, sev, manual, retries = derive_run_outcome(state)
        finalize_workflow_run(
            run_id,
            run_status=run_status,
            final_route=route,
            final_severity=sev,
            manual_review_required=manual,
            retry_count_total=retries,
            token_total=cost_acc.total_tokens or None,
            cost_estimate_usd=None,
        )
        log_workflow_event(
            "workflow_completed",
            run_id=run_id,
            final_route=route,
            run_status=run_status,
            total_retry_count=retries,
        )
        logger.info("Workflow complete – routed to %s", state.get("routed_to"))
        return state
    finally:
        reset_active_run(ctx_token)
