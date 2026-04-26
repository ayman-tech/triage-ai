"""LangGraph node wrappers: OTel spans, JSON events, audit rows."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from opentelemetry.trace import Status, StatusCode

from app.observability.context import (
    ActiveStep,
    get_active_run,
    reset_active_step,
    set_active_step,
    set_case_id,
    set_trace_id,
)
from app.observability.events import log_workflow_event, monotonic_ms
from app.observability.persistence import insert_workflow_step, update_workflow_run_case_id
from app.observability.state_summary import diff_summaries, dumps_compact, summarize_workflow_state
from app.observability.tracing import get_workflow_tracer, trace_id_hex_from_span
from app.observability.versions import default_chat_model
from app.orchestrator.state import WorkflowState

logger = logging.getLogger(__name__)

_LLM_NODES = frozenset(
    {
        "classify",
        "risk",
        "root_cause",
        "resolve",
        "check_compliance",
        "qa_review",
        "supervisor",
    }
)


def _confidence_after(node_name: str, state: WorkflowState) -> float | None:
    try:
        if node_name == "classify":
            c = state.get("classification")
            if c is None:
                return None
            v = getattr(c, "confidence", None)
            if v is None and isinstance(c, dict):
                v = c.get("confidence")
            return float(v) if v is not None else None
        if node_name == "risk":
            return None
        if node_name == "root_cause":
            rc = state.get("root_cause_hypothesis")
            if rc is None:
                return None
            v = getattr(rc, "confidence", None)
            if v is None and isinstance(rc, dict):
                v = rc.get("confidence")
            return float(v) if v is not None else None
        if node_name == "resolve":
            res = state.get("resolution")
            if res is None:
                return None
            v = getattr(res, "confidence", None)
            if v is None and isinstance(res, dict):
                v = res.get("confidence")
            return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
    return None


def wrap_node(
    node_name: str,
    fn: Callable[[WorkflowState], WorkflowState],
) -> Callable[[WorkflowState], WorkflowState]:
    """Decorate a LangGraph node with tracing + audit."""

    def _wrapped(state: WorkflowState) -> WorkflowState:
        ar = get_active_run()
        if ar is None:
            return fn(state)

        seq = ar.next_sequence()
        step_ctx_token = set_active_step(
            ActiveStep(
                node_name=node_name,
                sequence_number=seq,
                retry_number=int(state.get("retry_count") or 0),
            )
        )
        before = summarize_workflow_state(state)
        t0 = monotonic_ms()
        started_at = datetime.utcnow()
        tracer = get_workflow_tracer()
        retry_after = 0
        # retry_count after node completes (classify/resolution bump inside fn)
        model_name = default_chat_model() if node_name in _LLM_NODES else None

        log_workflow_event(
            "node_started",
            node_name=node_name,
            sequence_number=seq,
        )

        try:
            with tracer.start_as_current_span(f"node.{node_name}") as span:
                span.set_attribute("workflow.node", node_name)
                span.set_attribute("complaint.run_id", ar.run_id)
                span.set_attribute("complaint.company_id", ar.company_id)
                if ar.case_id:
                    span.set_attribute("complaint.case_id", ar.case_id)
                tid = trace_id_hex_from_span(span)
                if tid and not ar.trace_id:
                    set_trace_id(tid)

                try:
                    out = fn(state)
                except Exception as exc:
                    ended_at = datetime.utcnow()
                    latency_ms = monotonic_ms() - t0
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    log_workflow_event(
                        "workflow_failed",
                        node_name=node_name,
                        sequence_number=seq,
                        error_type=type(exc).__name__,
                        error_message=str(exc)[:500],
                        latency_ms=round(latency_ms, 2),
                    )
                    insert_workflow_step(
                        run_id=ar.run_id,
                        node_name=node_name,
                        sequence_number=seq,
                        started_at=started_at,
                        ended_at=ended_at,
                        latency_ms=latency_ms,
                        status="failure",
                        retry_number=int(state.get("retry_count") or 0),
                        model_name=model_name,
                        input_snapshot_json=dumps_compact(before),
                        output_snapshot_json=None,
                        state_diff_json=None,
                        confidence=None,
                        error_type=type(exc).__name__,
                        error_message=str(exc)[:2000],
                    )
                    raise

                ended_at = datetime.utcnow()
                latency_ms = monotonic_ms() - t0
                span.set_status(Status(StatusCode.OK))

                after = summarize_workflow_state(out)
                diff = diff_summaries(before, after)
                conf = _confidence_after(node_name, out)
                retry_after = int(out.get("retry_count") or 0)

                if node_name == "intake" and out.get("case") is not None:
                    case = out["case"]
                    cid = getattr(case, "id", None) or (
                        case.get("id") if isinstance(case, dict) else None
                    )
                    if cid:
                        set_case_id(str(cid))
                        update_workflow_run_case_id(ar.run_id, str(cid))

                log_workflow_event(
                    "node_completed",
                    node_name=node_name,
                    sequence_number=seq,
                    status="success",
                    latency_ms=round(latency_ms, 2),
                    confidence=conf,
                    retry_number=retry_after,
                )

                insert_workflow_step(
                    run_id=ar.run_id,
                    node_name=node_name,
                    sequence_number=seq,
                    started_at=started_at,
                    ended_at=ended_at,
                    latency_ms=latency_ms,
                    status="success",
                    retry_number=retry_after,
                    model_name=model_name,
                    input_snapshot_json=dumps_compact(before),
                    output_snapshot_json=dumps_compact(after),
                    state_diff_json=dumps_compact(diff),
                    confidence=conf,
                )

                return out
        finally:
            reset_active_step(step_ctx_token)

    return _wrapped


def wrap_supervisor_node(
    fn: Callable[[WorkflowState], Any],
) -> Callable[[WorkflowState], Any]:
    """Decorate the supervisor node with tracing + audit.

    Unlike ``wrap_node``, this handles the supervisor's ``Command`` return
    type — it records the routing decision without trying to summarize the
    output as a WorkflowState.
    """

    node_name = "supervisor"

    def _wrapped(state: WorkflowState) -> Any:
        ar = get_active_run()
        if ar is None:
            return fn(state)

        seq = ar.next_sequence()
        step_ctx_token = set_active_step(
            ActiveStep(
                node_name=node_name,
                sequence_number=seq,
                retry_number=int(state.get("retry_count") or 0),
            )
        )
        t0 = monotonic_ms()
        started_at = datetime.utcnow()
        tracer = get_workflow_tracer()
        model_name = default_chat_model()

        log_workflow_event(
            "node_started",
            node_name=node_name,
            sequence_number=seq,
        )

        try:
            with tracer.start_as_current_span(f"node.{node_name}") as span:
                span.set_attribute("workflow.node", node_name)
                span.set_attribute("complaint.run_id", ar.run_id)
                span.set_attribute("complaint.company_id", ar.company_id)
                if ar.case_id:
                    span.set_attribute("complaint.case_id", ar.case_id)
                tid = trace_id_hex_from_span(span)
                if tid and not ar.trace_id:
                    set_trace_id(tid)

                try:
                    command = fn(state)
                except Exception as exc:
                    ended_at = datetime.utcnow()
                    latency_ms = monotonic_ms() - t0
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    log_workflow_event(
                        "workflow_failed",
                        node_name=node_name,
                        sequence_number=seq,
                        error_type=type(exc).__name__,
                        error_message=str(exc)[:500],
                        latency_ms=round(latency_ms, 2),
                    )
                    insert_workflow_step(
                        run_id=ar.run_id,
                        node_name=node_name,
                        sequence_number=seq,
                        started_at=started_at,
                        ended_at=ended_at,
                        latency_ms=latency_ms,
                        status="failure",
                        retry_number=0,
                        model_name=model_name,
                        input_snapshot_json=None,
                        output_snapshot_json=None,
                        state_diff_json=None,
                        confidence=None,
                        error_type=type(exc).__name__,
                        error_message=str(exc)[:2000],
                    )
                    raise

                ended_at = datetime.utcnow()
                latency_ms = monotonic_ms() - t0
                span.set_status(Status(StatusCode.OK))

                goto = getattr(command, "goto", None)
                update = getattr(command, "update", {}) or {}
                reasoning = update.get("supervisor_reasoning", "")

                span.set_attribute("supervisor.goto", str(goto))
                if reasoning:
                    span.set_attribute("supervisor.reasoning", reasoning[:300])

                output_snapshot = dumps_compact({
                    "goto": str(goto),
                    "reasoning": reasoning,
                    "instructions": update.get("supervisor_instructions", ""),
                })

                log_workflow_event(
                    "node_completed",
                    node_name=node_name,
                    sequence_number=seq,
                    status="success",
                    latency_ms=round(latency_ms, 2),
                )

                insert_workflow_step(
                    run_id=ar.run_id,
                    node_name=node_name,
                    sequence_number=seq,
                    started_at=started_at,
                    ended_at=ended_at,
                    latency_ms=latency_ms,
                    status="success",
                    retry_number=0,
                    model_name=model_name,
                    input_snapshot_json=None,
                    output_snapshot_json=output_snapshot,
                    state_diff_json=None,
                    confidence=None,
                )

                return command
        finally:
            reset_active_step(step_ctx_token)

    return _wrapped
