"""Channel-agnostic intake engine used by chat (and later voice).

The engine maintains an IntakeSessionState and, on each turn:
  * calls an LLM with the existing IntakePacket and latest user message
  * parses a JSON response with `assistant_message` and updated `intake_packet`
  * applies deterministic sufficiency rules and builds a CaseCreate-compatible payload
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple
from uuid import uuid4

from app.agents.adk_runner import reset_adk_session, run_adk_json_agent, set_adk_session
from app.db.models import IntakeSessionRecord, WorkflowRun
from app.db.session import SessionLocal
from app.knowledge.company_knowledge import CompanyKnowledgeService
from app.knowledge.mock_company_pack import deployment_label
from app.observability.context import (
    ActiveRun,
    ActiveStep,
    get_active_run,
    reset_active_run,
    reset_active_step,
    set_active_run,
    set_active_step,
)
from app.observability.persistence import (
    finalize_workflow_run,
    insert_workflow_run,
    insert_workflow_step,
    update_workflow_run_case_id,
)
from app.schemas.case import CaseCreate
from app.schemas.intake import (
    InformationSufficiency,
    IntakePacket,
    IntakeSessionState,
    RecommendedHandoff,
)
from app.utils.pii import redact_pii

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "intake_chat.md"

# In-memory session store for the demo. For production, back this with Redis or DB.
_SESSIONS: Dict[str, IntakeSessionState] = {}
_MIN_DESCRIPTION_CHARS = 10
_DB_SESSION_STORE_AVAILABLE = True
_HUMAN_ESCALATION_REASONS = {
    "fraud_suspected",
    "identity_theft",
    "threat_of_harm",
    "self_harm",
    "elder_abuse",
    "legal_threat",
    "regulatory_threat",
}
_company_knowledge: CompanyKnowledgeService | None = None


def _company_knowledge_service() -> CompanyKnowledgeService:
    global _company_knowledge
    if _company_knowledge is None:
        _company_knowledge = CompanyKnowledgeService()
    return _company_knowledge


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").lower() in ("1", "true", "yes", "on")


def _trace_intake_to_langsmith_enabled() -> bool:
    """Opt-in only: do not send raw intake chat history by default."""
    return _truthy_env("TRACE_INTAKE_TO_LANGSMITH")


def _intake_run_id(session_id: str) -> str:
    return f"intake-{session_id}"


def _ensure_intake_tracking_run(session_id: str) -> str:
    run_id = _intake_run_id(session_id)
    try:
        session = SessionLocal()
        try:
            exists = session.get(WorkflowRun, run_id) is not None
        finally:
            session.close()
        if not exists:
            insert_workflow_run(run_id, trace_id=None)
    except Exception:
        insert_workflow_run(run_id, trace_id=None)
    return run_id


def link_intake_costs_to_case(session_id: str, case_id: str) -> None:
    """Attach intake chat LLM costs to the complaint after case creation."""
    run_id = _intake_run_id(session_id)
    update_workflow_run_case_id(run_id, case_id)
    finalize_workflow_run(
        run_id,
        run_status="intake_completed",
        final_route=None,
        final_severity=None,
        manual_review_required=False,
        retry_count_total=0,
    )


class _temp_disable_langsmith_tracing:
    """Disable LangSmith tracing for intake turns without mutating env vars."""

    def __init__(self) -> None:
        self._ctx = None

    def __enter__(self):
        try:
            from langsmith.run_helpers import tracing_context  # type: ignore

            self._ctx = tracing_context(enabled=False)
            self._ctx.__enter__()
        except Exception:
            self._ctx = None

    def __exit__(self, exc_type, exc, tb):
        if self._ctx is not None:
            self._ctx.__exit__(exc_type, exc, tb)


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _build_company_intake_context() -> dict[str, Any]:
    try:
        return _company_knowledge_service().build_intake_brief()
    except Exception:
        logger.warning("Unable to load company intake context", exc_info=True)
        return {
            "company_profile": {},
            "policy_candidates": [],
            "routing_candidates": {},
            "severity_rubric": [],
        }


def _render_company_intake_context() -> str:
    brief = _build_company_intake_context()
    profile = brief.get("company_profile") or {}
    display_name = profile.get("display_name") or "the bank"
    supported_products = ", ".join(profile.get("supported_products") or []) or "financial products"
    routing_guidance = "\n".join(f"- {item}" for item in profile.get("intake_routing_guidance") or [])
    policy_lines = "\n".join(
        f"- {item.get('policy_id')}: {item.get('description')}"
        for item in brief.get("policy_candidates") or []
    )
    return (
        "## Company Intake Context\n"
        f"- Company name: {display_name}\n"
        f"- Institution type: {profile.get('customer_identity') or 'financial institution'}\n"
        f"- Supported products: {supported_products}\n"
        f"- Role: {profile.get('intake_operator_style') or 'You are the internal complaints intake operator.'}\n"
        f"- Safe reference guidance: {profile.get('safe_reference_guidance') or 'Ask only for safe non-sensitive locators.'}\n"
        "### Internal routing guidance\n"
        f"{routing_guidance or '- Escalate urgent harm or fraud internally.'}\n"
        "### Relevant policy priorities\n"
        f"{policy_lines or '- Document the complaint clearly and route it to the right internal team.'}\n"
    )


def _build_intake_system_prompt() -> str:
    return f"{_load_prompt().rstrip()}\n\n{_render_company_intake_context()}"


def _persist_session_state(state: IntakeSessionState) -> None:
    global _DB_SESSION_STORE_AVAILABLE
    _SESSIONS[state.session_id] = state
    if not _DB_SESSION_STORE_AVAILABLE:
        return
    try:
        session = SessionLocal()
        try:
            record = session.get(IntakeSessionRecord, state.session_id) or IntakeSessionRecord(
                session_id=state.session_id
            )
            record.channel = state.channel
            record.company_id = None
            record.turn_index = state.turn_index
            record.packet_json = state.packet.model_dump_json()
            record.last_agent_message = state.last_agent_message
            record.last_user_message = state.last_user_message
            record.conversation_history_json = json.dumps(state.conversation_history, ensure_ascii=False)
            record.completed = state.completed
            record.handoff_triggered = state.handoff_triggered
            session.merge(record)
            session.commit()
        finally:
            session.close()
    except Exception:
        _DB_SESSION_STORE_AVAILABLE = False
        logger.warning("Unable to persist intake session; using in-memory fallback.")


def _load_session_state(session_id: str) -> IntakeSessionState | None:
    global _DB_SESSION_STORE_AVAILABLE
    cached = _SESSIONS.get(session_id)
    if cached is not None:
        return cached
    if not _DB_SESSION_STORE_AVAILABLE:
        return None

    try:
        session = SessionLocal()
        try:
            record = session.get(IntakeSessionRecord, session_id)
        finally:
            session.close()
    except Exception:
        _DB_SESSION_STORE_AVAILABLE = False
        logger.warning("Unable to load intake session from database; using in-memory fallback.")
        return None

    if record is None:
        return None

    state = IntakeSessionState(
        session_id=record.session_id,
        channel=record.channel,
        turn_index=record.turn_index,
        packet=IntakePacket.model_validate_json(record.packet_json),
        last_agent_message=record.last_agent_message or "",
        last_user_message=record.last_user_message or "",
        conversation_history=json.loads(record.conversation_history_json or "[]"),
        completed=bool(record.completed),
        handoff_triggered=bool(record.handoff_triggered),
    )
    _SESSIONS[session_id] = state
    return state


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return redact_pii(str(value)).strip()


def _coerce_optional_llm_bool(value: Any) -> bool | None:
    """Map messy LLM output to bool | None. Models sometimes put digits or text in bool slots."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        if not s:
            return None
        if s in ("true", "yes", "1", "y", "on"):
            return True
        if s in ("false", "no", "0", "n", "off"):
            return False
        # Common mistake: last-4 / reference stuffed into a boolean field — treat as "reference exists"
        if s.isdigit():
            return True
    return None


def _coerce_required_llm_bool(value: Any, *, default: bool) -> bool:
    coerced = _coerce_optional_llm_bool(value)
    return default if coerced is None else coerced


def _sanitize_packet_data(packet_data: dict[str, Any]) -> dict[str, Any]:
    """Normalize and redact free-text fields coming back from the model."""
    text_fields = {
        "customer_summary",
        "product_hint",
        "issue_hint",
        "sub_issue_hint",
        "date_of_incident",
        "amount",
        "currency",
        "merchant_or_counterparty",
        "desired_resolution",
        "narrative_for_case",
    }
    cleaned = dict(packet_data)
    for key in text_fields:
        if key in cleaned and cleaned[key] is not None:
            cleaned[key] = _clean_text(cleaned[key]) or None

    for key in ("customer_summary", "narrative_for_case"):
        if cleaned.get(key) is None:
            cleaned[key] = ""

    for key in ("escalation_reasons", "missing_fields"):
        if key in cleaned and cleaned[key] is not None:
            raw_value = cleaned[key]
            if isinstance(raw_value, (list, tuple)):
                cleaned[key] = [str(item).strip() for item in raw_value if str(item).strip()]
            else:
                single_value = str(raw_value).strip()
                cleaned[key] = [single_value] if single_value else []

    for key in ("is_financial_complaint", "supported_by_platform"):
        if key in cleaned and cleaned[key] is not None:
            cleaned[key] = _coerce_required_llm_bool(cleaned[key], default=True)

    for key in (
        "account_or_reference_available",
        "has_supporting_docs",
        "prior_contact_attempted",
    ):
        if key in cleaned and cleaned[key] is not None:
            cleaned[key] = _coerce_optional_llm_bool(cleaned[key])

    return cleaned


def _infer_currency_from_amount_packet(packet: IntakePacket) -> IntakePacket:
    """If currency is unset but the amount string contains a symbol, set currency (e.g. $ → USD)."""
    if (packet.currency or "").strip():
        return packet
    amt = (packet.amount or "").strip()
    if not amt:
        return packet
    inferred: str | None = None
    if amt.startswith("€") or amt.startswith("EUR"):
        inferred = "EUR"
    elif amt.startswith("£") or amt.startswith("GBP"):
        inferred = "GBP"
    elif amt.startswith("$") or amt.startswith("US$") or amt.startswith("USD"):
        inferred = "USD"
    elif "€" in amt:
        inferred = "EUR"
    elif "£" in amt:
        inferred = "GBP"
    elif "$" in amt:
        inferred = "USD"
    if inferred:
        return packet.model_copy(update={"currency": inferred})
    return packet


def _build_issue_label(packet: IntakePacket) -> str | None:
    parts = [packet.issue_hint, packet.sub_issue_hint]
    label = " / ".join(part.strip() for part in parts if part and part.strip())
    return label or None


def _channel_to_case_channel(channel: str) -> str:
    return "phone" if channel == "voice" else "web"


def _needs_human_escalation(packet: IntakePacket) -> bool:
    reasons = {reason.strip().lower() for reason in packet.escalation_reasons if reason.strip()}
    return (
        packet.intent.value == "fraud_report"
        or packet.urgency == "high"
        or bool(reasons & _HUMAN_ESCALATION_REASONS)
    )


def _compute_sufficiency(packet: IntakePacket) -> IntakePacket:
    """Deterministic intake completion rules for interactive chat/voice intake.

    Chat intake is intentionally stricter than bare ``CaseCreate`` validation:
      * unsupported / non-financial requests must not be filed as complaints
      * the operator should capture a usable complaint description
      * the operator should identify the product or issue before handoff
    """
    missing: list[str] = []
    narrative = (packet.narrative_for_case or "").strip()
    summary = (packet.customer_summary or "").strip()
    has_description = len(narrative) >= _MIN_DESCRIPTION_CHARS or len(summary) >= _MIN_DESCRIPTION_CHARS
    has_product_or_issue = bool(
        (packet.product_hint and packet.product_hint.strip())
        or (packet.issue_hint and packet.issue_hint.strip())
        or (packet.sub_issue_hint and packet.sub_issue_hint.strip())
    )

    if not packet.is_financial_complaint or not packet.supported_by_platform:
        packet.information_sufficiency = InformationSufficiency.INSUFFICIENT
        packet.recommended_handoff = RecommendedHandoff.UNSUPPORTED
        packet.missing_fields = ["financial_domain"]
        return packet

    if not has_description:
        missing.append("complaint_description")
    if not has_product_or_issue:
        missing.append("product_or_issue")
    if packet.prior_contact_attempted is None:
        missing.append("reported_to_bank")

    if missing:
        packet.information_sufficiency = (
            InformationSufficiency.PARTIAL
            if has_description or has_product_or_issue
            else InformationSufficiency.INSUFFICIENT
        )
    else:
        packet.information_sufficiency = InformationSufficiency.SUFFICIENT

    if _needs_human_escalation(packet):
        packet.recommended_handoff = RecommendedHandoff.HUMAN_ESCALATION
    else:
        packet.recommended_handoff = RecommendedHandoff.SUPERVISOR
    packet.missing_fields = missing
    return packet


def _build_case_payload(packet: IntakePacket) -> dict:
    """Construct a CaseCreate-compatible payload from an IntakePacket."""
    narrative = (packet.narrative_for_case or "").strip() or (packet.customer_summary or "").strip()
    data = {
        "consumer_narrative": narrative or None,
        # Product hints are useful to downstream classifiers; issue hints stay in external labels.
        "product": packet.product_hint,
        "sub_product": None,
        "company": None,
        "state": None,
        "zip_code": None,
        "channel": _channel_to_case_channel(packet.channel),
        "submitted_at": None,
        "external_product_category": packet.product_hint,
        "external_issue_type": _build_issue_label(packet),
        "requested_resolution": packet.desired_resolution,
        "intake_prior_contact_attempted": packet.prior_contact_attempted,
        "intake_intent": packet.intent.value,
        "intake_urgency": packet.urgency,
        "intake_recommended_handoff": packet.recommended_handoff.value,
        "intake_escalation_reasons": packet.escalation_reasons,
        "intake_customer_summary": packet.customer_summary or None,
        # CFPB portal fields are optional in this path.
        "cfpb_product": None,
        "cfpb_sub_product": None,
        "cfpb_issue": None,
        "cfpb_sub_issue": None,
    }
    return data


def _submission_offer_message(packet: IntakePacket) -> str:
    """Short line — the lodge UI shows a persistent in-chat summary card with full details."""
    urgency_note = ""
    if packet.recommended_handoff is RecommendedHandoff.HUMAN_ESCALATION:
        urgency_note = " I'm also flagging this for urgent internal review."
    return (
        "I have the minimum information needed to document your complaint."
        f"{urgency_note} Review the summary card below — submit when you're ready, or keep chatting to add details."
    )


def _needs_bank_registration_question(packet: IntakePacket) -> bool:
    """Ask whether the issue has already been reported to the bank before completion."""
    has_description = bool((packet.narrative_for_case or "").strip() or (packet.customer_summary or "").strip())
    has_product_or_issue = bool(
        (packet.product_hint and packet.product_hint.strip())
        or (packet.issue_hint and packet.issue_hint.strip())
        or (packet.sub_issue_hint and packet.sub_issue_hint.strip())
    )
    return packet.prior_contact_attempted is None and (has_description or has_product_or_issue)


def _bank_registration_follow_up(packet: IntakePacket) -> str:
    if packet.intent.value == "fraud_report":
        return (
            "Before I finalize this report, have you already reported this to Mock Bank "
            "or spoken with a bank representative about the fraud?"
        )
    return (
        "Before I finalize the complaint, have you already reported this issue to Mock Bank "
        "or spoken with the bank about it?"
    )


def start_intake_session(channel: str = "web_chat") -> Tuple[str, IntakeSessionState]:
    """Create a new intake session with an initial greeting from the agent."""
    session_id = uuid4().hex
    packet = IntakePacket(channel=channel if channel in ("web_chat", "voice") else "web_chat")
    state = IntakeSessionState(
        session_id=session_id,
        channel=packet.channel,
        packet=packet,
        turn_index=0,
        last_agent_message="",
        last_user_message="",
        conversation_history=[],
        completed=False,
        handoff_triggered=False,
    )
    greeting = (
        "Thanks for reaching out. I'm here to help document your complaint. "
        "Please briefly describe what happened, which financial product or service it relates to, "
        "and whether you've already reported it to Mock Bank. "
        "Do not include full card numbers, bank account numbers, or your Social Security number."
    )
    state.last_agent_message = greeting
    state.conversation_history.append({"role": "assistant", "message": greeting})
    _persist_session_state(state)
    return session_id, state


def get_intake_session(session_id: str) -> IntakeSessionState | None:
    return _load_session_state(session_id)


def process_intake_message(session_id: str, user_message: str, model_name: str | None = None) -> IntakeSessionState:
    """Process one user turn and update the intake session state."""
    state = _load_session_state(session_id)
    if state is None:
        raise KeyError(f"Unknown intake session_id={session_id}")

    was_completed = state.completed
    sanitized_user_message = _clean_text(user_message)
    state.turn_index += 1
    state.last_user_message = sanitized_user_message
    state.conversation_history.append({"role": "user", "message": sanitized_user_message})

    run_id = _ensure_intake_tracking_run(session_id)
    active_run_already_set = get_active_run() is not None
    run_ctx_token = None
    if not active_run_already_set:
        run_ctx_token = set_active_run(
            ActiveRun(
                run_id=run_id,
                company_id=deployment_label(),
            )
        )
    step_ctx_token = set_active_step(
        ActiveStep(
            node_name="intake_chat",
            sequence_number=state.turn_index,
            retry_number=0,
        )
    )
    adk_ctx_token = set_adk_session("intake", f"intake-{session_id}")
    started_at = datetime.utcnow()
    status = "success"
    error_type = None
    error_message = None
    try:
        system_prompt = _build_intake_system_prompt()
        # Send a transcript window plus the structured packet so the model can
        # preserve user corrections and maintain continuity across turns.
        payload = {
            "current_intake_packet": json.loads(state.packet.model_dump_json()),
            "conversation_history": state.conversation_history[-6:],
            "last_user_message": sanitized_user_message,
        }
        user_content = json.dumps(payload, ensure_ascii=False)

        # Privacy default: do not emit intake turns to LangSmith unless explicitly enabled.
        if _trace_intake_to_langsmith_enabled():
            data = run_adk_json_agent(
                name="intake_agent",
                description="Conversational complaint intake agent that gathers enough safe facts to open a case.",
                instruction=system_prompt,
                user_message=user_content,
                tools=[],
                model_name=model_name,
            )
        else:
            with _temp_disable_langsmith_tracing():
                data = run_adk_json_agent(
                    name="intake_agent",
                    description="Conversational complaint intake agent that gathers enough safe facts to open a case.",
                    instruction=system_prompt,
                    user_message=user_content,
                    tools=[],
                    model_name=model_name,
                )

        if set(data.keys()) != {"assistant_message", "intake_packet"}:
            raise ValueError("LLM response must contain exactly assistant_message and intake_packet.")
        packet_data = data.get("intake_packet")
        if not isinstance(packet_data, dict):
            raise ValueError("LLM response intake_packet must be an object.")

        merged_data = state.packet.model_dump(mode="python")
        merged_data.update(_sanitize_packet_data(packet_data))
        merged = IntakePacket.model_validate(merged_data)
        merged = _infer_currency_from_amount_packet(merged)
        merged = _compute_sufficiency(merged)
        merged.intake_case = _build_case_payload(merged)

        assistant_message = _clean_text(data.get("assistant_message")) or (
            "Please tell me what happened, which product or service it relates to, "
            "and any date or amount involved if you know it."
        )

        if _needs_bank_registration_question(merged):
            assistant_message = _bank_registration_follow_up(merged)

        state.packet = merged
        state.completed = merged.information_sufficiency is InformationSufficiency.SUFFICIENT
        if state.completed and not was_completed:
            state.last_agent_message = _submission_offer_message(merged)
        else:
            state.last_agent_message = assistant_message
    except Exception:
        status = "failure"
        error_type = "IntakeTurnProcessingError"
        error_message = "Intake turn processing failed; safe fallback returned."
        logger.exception("Intake turn processing failed; returning safe fallback")
        state.packet = _infer_currency_from_amount_packet(state.packet)
        state.packet = _compute_sufficiency(state.packet)
        state.packet.intake_case = _build_case_payload(state.packet)
        state.completed = state.packet.information_sufficiency is InformationSufficiency.SUFFICIENT
        state.last_agent_message = (
            "I'm sorry, I couldn't process that cleanly. Please restate what happened, "
            "which financial product or service it relates to, and any date or amount if known."
        )
    finally:
        ended_at = datetime.utcnow()
        latency_ms = max((ended_at - started_at).total_seconds() * 1000.0, 0.0)
        insert_workflow_step(
            run_id=run_id,
            node_name="intake_chat",
            sequence_number=state.turn_index,
            started_at=started_at,
            ended_at=ended_at,
            latency_ms=latency_ms,
            status=status,
            retry_number=0,
            model_name=model_name,
            input_snapshot_json=None,
            output_snapshot_json=None,
            state_diff_json=None,
            confidence=None,
            error_type=error_type,
            error_message=error_message,
        )
        finalize_workflow_run(
            run_id,
            run_status="intake_active",
            final_route=None,
            final_severity=None,
            manual_review_required=False,
            retry_count_total=0,
        )
        reset_adk_session(adk_ctx_token)
        reset_active_step(step_ctx_token)
        if run_ctx_token is not None:
            reset_active_run(run_ctx_token)

    state.conversation_history.append({"role": "assistant", "message": state.last_agent_message})
    _persist_session_state(state)
    return state


def finalize_intake_session(session_id: str) -> Tuple[CaseCreate, IntakeSessionState]:
    """Build a CaseCreate from the intake session; caller is responsible for running the workflow.

    Raises:
        ValueError if information is not sufficient.
    """
    state = _load_session_state(session_id)
    if state is None:
        raise KeyError(f"Unknown intake session_id={session_id}")

    packet = state.packet
    packet = _compute_sufficiency(packet)
    if packet.information_sufficiency is not InformationSufficiency.SUFFICIENT:
        raise ValueError(
            f"Intake information is not sufficient to open a case; missing={packet.missing_fields}"
        )

    payload = _build_case_payload(packet)
    case_create = CaseCreate(**payload)
    state.packet.intake_case = payload
    state.completed = True
    state.handoff_triggered = True
    _persist_session_state(state)
    return case_create, state
