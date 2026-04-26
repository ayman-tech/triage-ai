"""Persist workflow_runs / workflow_steps to Postgres (best-effort if DB up)."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from sqlalchemy import case, func
from sqlalchemy.exc import SQLAlchemyError

from app.db.models import ComplaintCase, LLMCallCost, WorkflowRun, WorkflowStep
from app.db.session import SessionLocal
from app.knowledge.mock_company_pack import deployment_label
from app.observability.versions import (
    default_chat_model,
    knowledge_pack_version,
    prompt_bundle_version,
    workflow_version,
)

logger = logging.getLogger(__name__)


def _atomic_call_count_expr():
    return func.coalesce(
        func.sum(
            case(
                (LLMCallCost.status == "backfilled_aggregate", 0),
                else_=1,
            )
        ),
        0,
    )


def _rollup_costs_for_run(
    session,
    run_id: str,
    *,
    sequence_number: int | None = None,
) -> dict[str, float | int]:
    query = session.query(
        _atomic_call_count_expr(),
        func.coalesce(func.sum(LLMCallCost.prompt_tokens), 0),
        func.coalesce(func.sum(LLMCallCost.completion_tokens), 0),
        func.coalesce(func.sum(LLMCallCost.total_tokens), 0),
        func.coalesce(func.sum(LLMCallCost.total_cost_usd), 0.0),
    ).filter(LLMCallCost.run_id == run_id)
    if sequence_number is not None:
        query = query.filter(LLMCallCost.sequence_number == sequence_number)
    row = query.one()
    return {
        "llm_call_count": int(row[0] or 0),
        "prompt_tokens": int(row[1] or 0),
        "completion_tokens": int(row[2] or 0),
        "token_total": int(row[3] or 0),
        "cost_estimate_usd": float(row[4] or 0.0),
    }


def insert_llm_call_cost(
    *,
    run_id: str,
    case_id: str | None,
    sequence_number: int | None,
    agent_name: str | None,
    langsmith_run_id: str | None,
    provider: str | None,
    model_name: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    input_cost_usd: float,
    output_cost_usd: float,
    total_cost_usd: float,
    latency_ms: float | None,
    status: str,
    retry_number: int,
    started_at: datetime | None,
    ended_at: datetime | None,
    metadata: dict[str, Any] | None = None,
) -> None:
    try:
        session = SessionLocal()
        try:
            row = LLMCallCost(
                run_id=run_id,
                case_id=case_id,
                sequence_number=sequence_number,
                agent_name=agent_name,
                langsmith_run_id=langsmith_run_id,
                provider=provider,
                model_name=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                input_cost_usd=input_cost_usd,
                output_cost_usd=output_cost_usd,
                total_cost_usd=total_cost_usd,
                latency_ms=latency_ms,
                status=status,
                retry_number=retry_number,
                started_at=started_at or datetime.utcnow(),
                ended_at=ended_at,
                metadata_json=json.dumps(metadata, separators=(",", ":")) if metadata else None,
            )
            session.add(row)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    except SQLAlchemyError as e:
        logger.warning("llm_call_costs insert skipped: %s", e)


def insert_workflow_run(
    run_id: str,
    trace_id: str | None,
) -> bool:
    try:
        session = SessionLocal()
        try:
            label = deployment_label()
            row = WorkflowRun(
                run_id=run_id,
                company_id=label,
                trace_id=trace_id,
                started_at=datetime.utcnow(),
                run_status="running",
                llm_call_count=0,
                workflow_version=workflow_version(),
                prompt_version=prompt_bundle_version(),
                knowledge_pack_version=knowledge_pack_version(),
                model_version=default_chat_model(),
            )
            session.add(row)
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    except SQLAlchemyError as e:
        logger.warning("workflow_runs insert skipped: %s", e)
        return False


def update_workflow_run_case_id(run_id: str, case_id: str) -> None:
    try:
        session = SessionLocal()
        try:
            row = session.get(WorkflowRun, run_id)
            if row:
                row.case_id = case_id
                session.query(LLMCallCost).filter(
                    LLMCallCost.run_id == run_id,
                    LLMCallCost.case_id.is_(None),
                ).update({"case_id": case_id}, synchronize_session=False)
                session.commit()
        finally:
            session.close()
    except SQLAlchemyError as e:
        logger.warning("workflow_runs case_id update skipped: %s", e)


def insert_workflow_step(
    *,
    run_id: str,
    node_name: str,
    sequence_number: int,
    started_at: datetime,
    ended_at: datetime,
    latency_ms: float,
    status: str,
    retry_number: int,
    model_name: str | None,
    input_snapshot_json: str | None,
    output_snapshot_json: str | None,
    state_diff_json: str | None,
    confidence: float | None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    try:
        session = SessionLocal()
        try:
            step = WorkflowStep(
                run_id=run_id,
                node_name=node_name,
                sequence_number=sequence_number,
                started_at=started_at,
                ended_at=ended_at,
                latency_ms=latency_ms,
                status=status,
                retry_number=retry_number,
                model_name=model_name,
                input_snapshot_json=input_snapshot_json,
                output_snapshot_json=output_snapshot_json,
                state_diff_json=state_diff_json,
                confidence=confidence,
                error_type=error_type,
                error_message=error_message,
            )
            rollup = _rollup_costs_for_run(session, run_id, sequence_number=sequence_number)
            step.llm_call_count = int(rollup["llm_call_count"])
            step.prompt_tokens = int(rollup["prompt_tokens"])
            step.completion_tokens = int(rollup["completion_tokens"])
            step.token_total = int(rollup["token_total"])
            step.cost_estimate_usd = float(rollup["cost_estimate_usd"])
            session.add(step)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    except SQLAlchemyError as e:
        logger.warning("workflow_steps insert skipped: %s", e)


def finalize_workflow_run(
    run_id: str,
    *,
    run_status: str,
    final_route: str | None,
    final_severity: str | None,
    manual_review_required: bool,
    retry_count_total: int,
    token_total: int | None = None,
    cost_estimate_usd: float | None = None,
) -> None:
    try:
        session = SessionLocal()
        try:
            row = session.get(WorkflowRun, run_id)
            if not row:
                return
            rollup = _rollup_costs_for_run(session, run_id)
            row.ended_at = datetime.utcnow()
            row.run_status = run_status
            row.final_route = final_route
            row.final_severity = final_severity
            row.manual_review_required = manual_review_required
            row.retry_count_total = retry_count_total
            row.llm_call_count = int(rollup["llm_call_count"])
            row.token_total = token_total if token_total is not None else int(rollup["token_total"])
            row.cost_estimate_total = (
                cost_estimate_usd if cost_estimate_usd is not None else float(rollup["cost_estimate_usd"])
            )
            # Read case_id before commit so it's available after session state changes
            linked_case_id = row.case_id
            session.commit()

            # Propagate cost to the linked complaint case
            if linked_case_id:
                _update_case_cost(session, linked_case_id, row.token_total, row.cost_estimate_total)
        finally:
            session.close()
    except SQLAlchemyError as e:
        logger.warning("workflow_runs finalize skipped: %s", e)


def _update_case_cost(
    session,
    case_id: str,
    token_total: int | None,
    cost_estimate_usd: float | None,
) -> None:
    """Write token/cost totals back to the complaint_cases row."""
    try:
        case = session.get(ComplaintCase, case_id)
        if not case:
            return
        if token_total is not None:
            case.token_total = token_total
        if cost_estimate_usd is not None:
            case.cost_estimate_usd = cost_estimate_usd
        session.commit()
    except SQLAlchemyError as e:
        logger.warning("complaint_cases cost update skipped (case=%s): %s", case_id, e)


def derive_run_outcome(final_state: dict[str, Any]) -> tuple[str, str | None, str | None, bool, int]:
    """run_status, final_route, final_severity, manual_review_required, retry_count."""
    retry_count = int(final_state.get("retry_count") or 0)
    routed = final_state.get("routed_to")
    review = final_state.get("review") or {}
    decision = review.get("decision") if isinstance(review, dict) else None

    risk = final_state.get("risk_assessment")
    sev = None
    if risk is not None:
        rd = risk.model_dump() if hasattr(risk, "model_dump") else dict(risk)
        sev = str(rd.get("risk_level", ""))

    manual = decision in ("escalate", "revise")
    if decision == "escalate":
        status = "escalated"
    elif decision == "revise":
        status = "needs_follow_up"
    else:
        status = "completed"

    return status, routed if isinstance(routed, str) else None, sev, manual, retry_count
