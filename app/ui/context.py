"""Template context builders — query DB and format data for Jinja2 templates."""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timedelta

from sqlalchemy import case, func
from sqlalchemy.orm import Session

from app.documents.service import build_case_document_summary
from app.evals.service import (
    build_production_evaluation_case_detail as _build_production_evaluation_case_detail,
    build_evaluation_case_detail as _build_evaluation_case_detail,
    build_evaluation_dashboard_data as _build_evaluation_dashboard_data,
)
from app.knowledge.mock_company_pack import format_root_cause_category
from app.db.models import (
    ComplaintCase,
    ClassificationRecord,
    LLMCallCost,
    RiskRecord,
    WorkflowRun,
)

logger = logging.getLogger(__name__)

_REPORTING_RANGES: dict[str, tuple[str, timedelta | None]] = {
    "24h": ("24 Hours", timedelta(hours=24)),
    "7d": ("7 Days", timedelta(days=7)),
    "30d": ("Monthly", timedelta(days=30)),
    "all": ("All Time", None),
}


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


def _atomic_cost_sum_expr():
    return func.coalesce(
        func.sum(
            case(
                (LLMCallCost.status == "backfilled_aggregate", 0.0),
                else_=LLMCallCost.total_cost_usd,
            )
        ),
        0.0,
    )


def _safe_json_load(value: str | None) -> dict | list | None:
    if not value:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def _case_cost_snapshot(db_case: ComplaintCase) -> dict:
    return {
        "token_total": int(getattr(db_case, "token_total", 0) or 0),
        "cost_estimate_usd": float(getattr(db_case, "cost_estimate_usd", 0.0) or 0.0),
    }


def _run_cost_snapshot(run: WorkflowRun | None) -> dict:
    if run is None:
        return {
            "llm_call_count": None,
            "llm_call_count_available": False,
            "token_total": 0,
            "cost_estimate_total": 0.0,
        }
    llm_call_count = getattr(run, "llm_call_count", None)
    return {
        "llm_call_count": int(llm_call_count) if llm_call_count is not None else None,
        "llm_call_count_available": llm_call_count is not None,
        "token_total": int(getattr(run, "token_total", 0) or 0),
        "cost_estimate_total": float(getattr(run, "cost_estimate_total", 0.0) or 0.0),
    }


def _run_cost_snapshot_with_ledger(run: WorkflowRun | None, db: Session | None) -> dict:
    snapshot = _run_cost_snapshot(run)
    if run is None or db is None:
        return snapshot

    ledger_totals = (
        db.query(
            func.count(LLMCallCost.id),
            _atomic_call_count_expr(),
            func.coalesce(func.sum(LLMCallCost.total_tokens), 0),
            func.coalesce(func.sum(LLMCallCost.total_cost_usd), 0.0),
        )
        .filter(LLMCallCost.run_id == run.run_id)
        .one()
    )
    ledger_row_count = int(ledger_totals[0] or 0)
    ledger_call_count = int(ledger_totals[1] or 0)
    ledger_token_total = int(ledger_totals[2] or 0)
    ledger_cost_total = float(ledger_totals[3] or 0.0)

    if ledger_row_count > 0:
        if ledger_call_count > 0:
            snapshot["llm_call_count"] = ledger_call_count
            snapshot["llm_call_count_available"] = True
        else:
            snapshot["llm_call_count"] = None
            snapshot["llm_call_count_available"] = False
        snapshot["token_total"] = ledger_token_total
        snapshot["cost_estimate_total"] = ledger_cost_total

    return snapshot


def _extract_supporting_documents(transcript: dict | None) -> list[dict]:
    if not isinstance(transcript, dict):
        return []

    candidates: list[object] = []
    final_packet = transcript.get("final_packet")
    if isinstance(final_packet, dict):
        candidates.extend([
            final_packet.get("documents"),
            final_packet.get("attachments"),
            final_packet.get("uploaded_files"),
            final_packet.get("files"),
        ])
        intake_case = final_packet.get("intake_case")
        if isinstance(intake_case, dict):
            candidates.extend([
                intake_case.get("documents"),
                intake_case.get("attachments"),
                intake_case.get("uploaded_files"),
                intake_case.get("files"),
            ])

    documents: list[dict] = []
    for candidate in candidates:
        if not isinstance(candidate, list):
            continue
        for item in candidate:
            if isinstance(item, dict):
                documents.append(item)
            elif isinstance(item, str):
                documents.append({"name": item})
    return documents


def _document_rows_from_case(db_case: ComplaintCase) -> list[dict]:
    docs = []
    for row in getattr(db_case, "documents", []) or []:
        artifact = getattr(row, "artifact", None)
        extracted = _safe_json_load(getattr(artifact, "extracted_json", None)) if artifact else {}
        docs.append({
            "id": row.id,
            "name": row.original_filename,
            "mime_type": row.mime_type,
            "size_bytes": row.size_bytes,
            "upload_status": row.upload_status,
            "parser_status": row.parser_status,
            "extraction_status": row.extraction_status,
            "document_type": row.document_type,
            "processing_error": row.processing_error,
            "facts": extracted if isinstance(extracted, dict) else {},
        })
    return docs


def build_case_summary(db_case: ComplaintCase) -> dict:
    """Build a lightweight summary dict for the dashboard table."""
    cls = db_case.classification
    risk = db_case.risk_assessment

    narrative = db_case.consumer_narrative or ""
    subject = narrative[:200] + "..." if len(narrative) > 200 else narrative

    res = db_case.resolution
    transcript = _safe_json_load(getattr(db_case, "intake_session_transcript_json", None))
    final_packet = transcript.get("final_packet") if isinstance(transcript, dict) else None
    conversation_history = (
        transcript.get("conversation_history")
        if isinstance(transcript, dict) and isinstance(transcript.get("conversation_history"), list)
        else []
    )
    supporting_documents = _document_rows_from_case(db_case) or _extract_supporting_documents(transcript)
    document_summary = build_case_document_summary(db_case.id) if db_case.id else None

    created = db_case.created_at
    updated = db_case.updated_at
    cost_snapshot = _case_cost_snapshot(db_case)
    if created and updated and updated > created:
        delta = updated - created
        total_seconds = int(delta.total_seconds())
        if total_seconds < 3600:
            total_time_str = f"{total_seconds // 60}m"
        elif total_seconds < 86400:
            h, m = divmod(total_seconds // 60, 60)
            total_time_str = f"{h}h {m}m" if m else f"{h}h"
        else:
            days = total_seconds // 86400
            h = (total_seconds % 86400) // 3600
            total_time_str = f"{days}d {h}h" if h else f"{days}d"
    else:
        total_time_str = None

    return {
        "id": db_case.id,
        "public_case_id": getattr(db_case, "public_case_id", None) or db_case.id[:12].upper(),
        "subject": subject,
        "status": db_case.status or "unknown",
        "product": cls.product_category if cls else None,
        "issue_type": cls.issue_type if cls else None,
        "sub_issue": cls.sub_issue if cls else None,
        "sub_product": db_case.sub_product,
        "risk_level": risk.risk_level if risk else None,
        "risk_score": risk.risk_score if risk else None,
        "confidence": cls.confidence if cls else None,
        "routed_to": db_case.routed_to,
        "team_assignment": db_case.team_assignment,
        "created_at": db_case.created_at,
        "severity_class": db_case.severity_class,
        "total_time_str": total_time_str,
        "estimated_resolution_days": res.estimated_resolution_days if res else None,
        "monetary_amount": res.monetary_amount if res else None,
        "intake_session_transcript": transcript if isinstance(transcript, dict) else None,
        "conversation_history": conversation_history,
        "intake_payload": final_packet if isinstance(final_packet, dict) else None,
        "supporting_documents": supporting_documents,
        "has_supporting_docs": (
            bool(supporting_documents) or bool(final_packet.get("has_supporting_docs"))
            if isinstance(final_packet, dict)
            else bool(supporting_documents)
        ),
        "document_summary": document_summary.model_dump() if document_summary else {},
        "token_total": cost_snapshot["token_total"],
        "cost_estimate_usd": cost_snapshot["cost_estimate_usd"],
    }


_TERMINAL_STATUSES = frozenset({"resolved", "closed", "dismissed"})


def build_admin_overview_data(db: Session) -> dict:
    """KPIs and drilldown cards for the admin control-tower overview."""
    return build_admin_overview_data_for_range(db)


def build_admin_overview_data_for_range(db: Session, range_key: str = "24h") -> dict:
    """KPIs and recent rows for the admin operational intelligence overview."""
    selected_range_key = range_key if range_key in _REPORTING_RANGES else "24h"
    selected_range_label, delta = _REPORTING_RANGES[selected_range_key]
    cutoff = datetime.utcnow() - delta if delta is not None else None

    total = db.query(ComplaintCase).count()
    resolved_count = (
        db.query(ComplaintCase)
        .filter(ComplaintCase.status.in_(list(_TERMINAL_STATUSES)))
        .count()
    )
    active_count = max(0, total - resolved_count)

    critical_count = db.query(RiskRecord).filter(RiskRecord.risk_level == "critical").count()

    recent = (
        db.query(ComplaintCase)
        .order_by(ComplaintCase.created_at.desc())
        .limit(8)
        .all()
    )
    recent_queue = [build_case_summary(row) for row in recent]

    resolution_rate = round(100.0 * resolved_count / total, 1) if total else 0.0
    llm_cost_query = db.query(LLMCallCost)
    run_query = db.query(WorkflowRun)
    complaint_query = db.query(ComplaintCase)
    if cutoff is not None:
        llm_cost_query = llm_cost_query.filter(LLMCallCost.started_at >= cutoff)
        run_query = run_query.filter(WorkflowRun.started_at >= cutoff)
        complaint_query = complaint_query.filter(ComplaintCase.created_at >= cutoff)

    tracked_spend = (
        llm_cost_query.with_entities(func.coalesce(func.sum(LLMCallCost.total_cost_usd), 0.0)).scalar()
        or 0.0
    )
    atomic_spend = (
        llm_cost_query.with_entities(_atomic_cost_sum_expr()).scalar()
        or 0.0
    )
    atomic_call_count = (
        llm_cost_query.with_entities(_atomic_call_count_expr()).scalar()
        or 0
    )
    tracked_tokens = (
        llm_cost_query.with_entities(func.coalesce(func.sum(LLMCallCost.total_tokens), 0)).scalar()
        or 0
    )
    tracked_case_count = (
        run_query.with_entities(func.count(func.distinct(WorkflowRun.case_id)))
        .filter(WorkflowRun.case_id.isnot(None), WorkflowRun.cost_estimate_total.isnot(None))
        .scalar()
        or 0
    )
    intake_count = complaint_query.count()
    completed_runs = (
        run_query
        .filter(WorkflowRun.run_status == "completed")
        .count()
    )
    avg_cost_per_complaint = (
        float(tracked_spend) / tracked_case_count if tracked_case_count else 0.0
    )
    tracking_coverage_pct = (
        round((tracked_case_count / intake_count) * 100.0, 1)
        if intake_count else 0.0
    )
    top_agent_row = (
        llm_cost_query.with_entities(
            LLMCallCost.agent_name,
            func.coalesce(func.sum(LLMCallCost.total_cost_usd), 0.0),
            _atomic_call_count_expr(),
        )
        .filter(LLMCallCost.agent_name.isnot(None), LLMCallCost.status != "backfilled_aggregate")
        .group_by(LLMCallCost.agent_name)
        .order_by(func.sum(LLMCallCost.total_cost_usd).desc())
        .first()
    )
    top_agent = None
    if top_agent_row is not None:
        top_agent = {
            "agent_name": top_agent_row[0],
            "total_cost_usd": round(float(top_agent_row[1] or 0.0), 4),
            "call_count": int(top_agent_row[2] or 0),
            "share_pct": round(((float(top_agent_row[1] or 0.0) / float(atomic_spend)) * 100.0), 1) if atomic_spend else 0.0,
        }

    highest_cost_cases_rows = (
        db.query(ComplaintCase)
        .filter(ComplaintCase.cost_estimate_usd.isnot(None))
        .order_by(ComplaintCase.cost_estimate_usd.desc(), ComplaintCase.updated_at.desc())
        .limit(5)
        .all()
    )
    highest_cost_cases = [build_case_summary(row) for row in highest_cost_cases_rows]

    return {
        "total": total,
        "active_count": active_count,
        "resolved_count": resolved_count,
        "critical_count": critical_count,
        "resolution_rate": resolution_rate,
        "recent_queue": recent_queue,
        "selected_range_key": selected_range_key,
        "selected_range_label": selected_range_label,
        "range_options": [
            {"key": key, "label": label}
            for key, (label, _delta) in _REPORTING_RANGES.items()
            if key != "all"
        ],
        "window_complaint_count": int(intake_count),
        "tracked_spend_usd": round(float(tracked_spend), 4),
        "atomic_spend_usd": round(float(atomic_spend), 4),
        "tracked_tokens": int(tracked_tokens or 0),
        "completed_runs": int(completed_runs),
        "tracked_case_count": int(tracked_case_count),
        "avg_cost_per_complaint_usd": round(avg_cost_per_complaint, 4),
        "tracking_coverage_pct": tracking_coverage_pct,
        "atomic_call_count": int(atomic_call_count),
        "top_agent": top_agent,
        "highest_cost_cases": highest_cost_cases,
    }


def build_case_detail(db_case: ComplaintCase, db: Session | None = None) -> dict:
    """Build a full detail dict for the complaint detail view."""
    cls = db_case.classification
    risk = db_case.risk_assessment
    res = db_case.resolution

    classification = None
    if cls:
        classification = {
            "product_category": cls.product_category,
            "issue_type": cls.issue_type,
            "sub_issue": cls.sub_issue,
            "confidence": cls.confidence,
            "reasoning": cls.reasoning,
        }

    risk_assessment = None
    if risk:
        risk_assessment = {
            "risk_level": risk.risk_level,
            "risk_score": risk.risk_score,
            "regulatory_risk": risk.regulatory_risk,
            "financial_impact_estimate": risk.financial_impact_estimate,
            "escalation_required": risk.escalation_required,
            "reasoning": risk.reasoning,
        }

    resolution = None
    if res:
        resolution = {
            "recommended_action": res.recommended_action,
            "description": res.description,
            "estimated_resolution_days": res.estimated_resolution_days,
            "monetary_amount": res.monetary_amount,
            "confidence": res.confidence,
            "reasoning": res.reasoning,
        }

    root_cause_hypothesis = _safe_json_load(db_case.root_cause_hypothesis_json)
    if isinstance(root_cause_hypothesis, dict):
        raw_category = root_cause_hypothesis.get("root_cause_category")
        root_cause_hypothesis = {
            **root_cause_hypothesis,
            "root_cause_display_label": format_root_cause_category(raw_category),
        }
    cost_snapshot = _case_cost_snapshot(db_case)
    latest_run = None
    agent_costs: list[dict] = []
    run_snapshot = _run_cost_snapshot(None)
    if db is not None:
        latest_run = (
            db.query(WorkflowRun)
            .filter(WorkflowRun.case_id == db_case.id)
            .order_by(WorkflowRun.started_at.desc())
            .first()
        )
        run_snapshot = _run_cost_snapshot_with_ledger(latest_run, db)
        if latest_run is not None:
            agent_rows = (
                db.query(
                    LLMCallCost.agent_name,
                    func.count(LLMCallCost.id),
                    func.coalesce(func.sum(LLMCallCost.total_cost_usd), 0.0),
                    func.coalesce(func.sum(LLMCallCost.total_tokens), 0),
                )
                .filter(
                    LLMCallCost.run_id == latest_run.run_id,
                    LLMCallCost.agent_name.isnot(None),
                )
                .group_by(LLMCallCost.agent_name)
                .order_by(func.sum(LLMCallCost.total_cost_usd).desc())
                .all()
            )
            run_total_cost = float(run_snapshot["cost_estimate_total"] or 0.0)
            for agent_name, call_count, total_cost, total_tokens in agent_rows:
                share_pct = ((float(total_cost) / run_total_cost) * 100.0) if run_total_cost else 0.0
                agent_costs.append({
                    "agent_name": agent_name,
                    "call_count": int(call_count or 0),
                    "total_cost_usd": round(float(total_cost or 0.0), 4),
                    "total_tokens": int(total_tokens or 0),
                    "share_pct": round(share_pct, 1),
                })
    return {
        "id": db_case.id,
        "public_case_id": getattr(db_case, "public_case_id", None) or db_case.id[:12].upper(),
        "status": db_case.status or "unknown",
        "consumer_narrative": db_case.consumer_narrative,
        "product": db_case.product,
        "sub_product": db_case.sub_product,
        "company": db_case.company,
        "state": db_case.state,
        "zip_code": db_case.zip_code,
        "channel": db_case.channel,
        "submitted_at": db_case.submitted_at,
        "created_at": db_case.created_at,
        "classification": classification,
        "risk_assessment": risk_assessment,
        "resolution": resolution,
        "root_cause_hypothesis": root_cause_hypothesis,
        "compliance_flags": _safe_json_load(db_case.compliance_flags_json),
        "review_notes": db_case.review_notes,
        "routed_to": db_case.routed_to,
        "team_assignment": db_case.team_assignment,
        "severity_class": db_case.severity_class,
        "document_gate_result": _safe_json_load(getattr(db_case, "document_gate_result_json", None)),
        "document_consistency": _safe_json_load(getattr(db_case, "document_consistency_json", None)),
        "intake_session_transcript": _safe_json_load(
            getattr(db_case, "intake_session_transcript_json", None)
        ),
        "case_documents": _document_rows_from_case(db_case),
        "case_document_summary": build_case_document_summary(db_case.id).model_dump() if db_case.id else {},
        "token_total": cost_snapshot["token_total"],
        "cost_estimate_usd": cost_snapshot["cost_estimate_usd"],
        "latest_run": {
            "run_id": latest_run.run_id,
            "llm_call_count": run_snapshot["llm_call_count"],
            "llm_call_count_available": run_snapshot["llm_call_count_available"],
            "token_total": run_snapshot["token_total"],
            "cost_estimate_total": run_snapshot["cost_estimate_total"],
            "started_at": latest_run.started_at,
            "ended_at": latest_run.ended_at,
        } if latest_run is not None else None,
        "agent_costs": agent_costs,
    }


def build_operations_data(db: Session, range_key: str = "all") -> dict:
    """Build aggregate operations analytics for the admin dashboard."""
    selected_range_key = range_key if range_key in _REPORTING_RANGES else "all"
    selected_range_label, delta = _REPORTING_RANGES[selected_range_key]
    cutoff = datetime.utcnow() - delta if delta is not None else None

    complaint_query = db.query(ComplaintCase)
    run_query = db.query(WorkflowRun)
    llm_cost_query = db.query(LLMCallCost)

    if cutoff is not None:
        complaint_query = complaint_query.filter(ComplaintCase.created_at >= cutoff)
        run_query = run_query.filter(WorkflowRun.started_at >= cutoff)
        llm_cost_query = llm_cost_query.filter(LLMCallCost.started_at >= cutoff)

    total_complaints = complaint_query.count()

    # Complaints by product category
    category_rows = (
        db.query(ClassificationRecord.product_category)
        .join(ComplaintCase, ComplaintCase.id == ClassificationRecord.case_id)
        .filter(ComplaintCase.created_at >= cutoff) if cutoff is not None else
        db.query(ClassificationRecord.product_category)
        .join(ComplaintCase, ComplaintCase.id == ClassificationRecord.case_id)
    ).all()
    category_counts = Counter(r[0] for r in category_rows if r[0])

    # Risk distribution
    risk_rows = (
        db.query(RiskRecord.risk_level)
        .join(ComplaintCase, ComplaintCase.id == RiskRecord.case_id)
        .filter(ComplaintCase.created_at >= cutoff) if cutoff is not None else
        db.query(RiskRecord.risk_level)
        .join(ComplaintCase, ComplaintCase.id == RiskRecord.case_id)
    ).all()
    risk_counts = Counter(r[0] for r in risk_rows if r[0])

    # Team workload
    team_rows = (
        complaint_query
        .with_entities(ComplaintCase.routed_to)
        .filter(ComplaintCase.routed_to.isnot(None))
        .all()
    )
    team_counts = Counter(r[0] for r in team_rows)

    # Recent runs for resolution time
    runs = (
        run_query
        .filter(WorkflowRun.ended_at.isnot(None))
        .order_by(WorkflowRun.started_at.desc())
        .limit(100)
        .all()
    )
    resolution_times = []
    for run in runs:
        if run.started_at and run.ended_at:
            delta = (run.ended_at - run.started_at).total_seconds()
            resolution_times.append({
                "date": run.started_at.strftime("%Y-%m-%d"),
                "seconds": round(delta, 1),
            })

    avg_resolution = (
        sum(r["seconds"] for r in resolution_times) / len(resolution_times)
        if resolution_times else 0
    )

    llm_spend_total = (
        llm_cost_query.with_entities(func.coalesce(func.sum(LLMCallCost.total_cost_usd), 0.0)).scalar()
        or 0.0
    )
    atomic_llm_spend_total = (
        llm_cost_query.with_entities(_atomic_cost_sum_expr()).scalar()
        or 0.0
    )
    llm_call_count = (
        llm_cost_query.with_entities(_atomic_call_count_expr()).scalar()
        or 0
    )
    tracked_case_count = (
        run_query.with_entities(func.count(func.distinct(WorkflowRun.case_id)))
        .filter(WorkflowRun.case_id.isnot(None), WorkflowRun.cost_estimate_total.isnot(None))
        .scalar()
        or 0
    )
    completed_run_count = (
        run_query.filter(WorkflowRun.run_status == "completed").count()
    )
    avg_cost_per_complaint = (llm_spend_total / tracked_case_count) if tracked_case_count else 0.0
    avg_cost_per_call = (atomic_llm_spend_total / llm_call_count) if llm_call_count else 0.0
    tracking_coverage_pct = (
        round((tracked_case_count / total_complaints) * 100.0, 1)
        if total_complaints else 0.0
    )

    agent_rows = (
        llm_cost_query.with_entities(
            LLMCallCost.agent_name,
            func.count(LLMCallCost.id),
            func.coalesce(func.sum(LLMCallCost.total_cost_usd), 0.0),
            func.coalesce(func.sum(LLMCallCost.total_tokens), 0),
            func.coalesce(func.avg(LLMCallCost.total_cost_usd), 0.0),
        )
        .filter(LLMCallCost.agent_name.isnot(None))
        .group_by(LLMCallCost.agent_name)
        .order_by(func.sum(LLMCallCost.total_cost_usd).desc())
        .all()
    )
    agent_costs = []
    for agent_name, call_count, total_cost, total_tokens, avg_call_cost in agent_rows:
        share_pct = ((float(total_cost) / atomic_llm_spend_total) * 100.0) if atomic_llm_spend_total else 0.0
        avg_tokens_per_call = (int(total_tokens) / int(call_count)) if call_count else 0
        agent_costs.append({
            "agent_name": agent_name,
            "call_count": int(call_count or 0),
            "total_cost_usd": round(float(total_cost or 0.0), 4),
            "total_tokens": int(total_tokens or 0),
            "avg_cost_per_call_usd": round(float(avg_call_cost or 0.0), 4),
            "avg_tokens_per_call": round(avg_tokens_per_call),
            "share_pct": round(share_pct, 1),
        })

    daily_cost_rows = (
        llm_cost_query.with_entities(
            func.date(LLMCallCost.started_at),
            func.coalesce(func.sum(LLMCallCost.total_cost_usd), 0.0),
        )
        .group_by(func.date(LLMCallCost.started_at))
        .order_by(func.date(LLMCallCost.started_at))
        .all()
    )
    spend_trend = [
        {
            "date": str(day),
            "cost_usd": round(float(total_cost or 0.0), 4),
        }
        for day, total_cost in daily_cost_rows
    ]

    top_cost_case_rows = (
        complaint_query
        .filter(ComplaintCase.cost_estimate_usd.isnot(None))
        .order_by(ComplaintCase.cost_estimate_usd.desc(), ComplaintCase.updated_at.desc())
        .limit(10)
        .all()
    )
    top_cost_cases = [build_case_summary(row) for row in top_cost_case_rows]

    return {
        "total_complaints": total_complaints,
        "selected_range_key": selected_range_key,
        "selected_range_label": selected_range_label,
        "range_options": [
            {"key": key, "label": label}
            for key, (label, _delta) in _REPORTING_RANGES.items()
        ],
        "avg_resolution_seconds": round(avg_resolution, 1),
        "llm_spend_total_usd": round(float(llm_spend_total), 4),
        "atomic_llm_spend_total_usd": round(float(atomic_llm_spend_total), 4),
        "llm_call_count": int(llm_call_count),
        "tracked_case_count": int(tracked_case_count),
        "completed_run_count": int(completed_run_count),
        "avg_cost_per_complaint_usd": round(avg_cost_per_complaint, 4),
        "avg_cost_per_call_usd": round(avg_cost_per_call, 4),
        "tracking_coverage_pct": tracking_coverage_pct,
        "agent_costs": agent_costs[:8],
        "top_agent": agent_costs[0] if agent_costs else None,
        "agent_cost_chart_labels": [
            item["agent_name"].replace("_", " ").title() for item in agent_costs[:8]
        ],
        "agent_cost_chart_values": [
            item["total_cost_usd"] for item in agent_costs[:8]
        ],
        "spend_trend": spend_trend,
        "category_counts": dict(category_counts.most_common(10)),
        "risk_counts": dict(risk_counts),
        "team_counts": dict(team_counts.most_common(10)),
        "resolution_times": resolution_times[:20],
        "top_cost_cases": top_cost_cases,
        "model_costs": _build_model_cost_rows(llm_cost_query),
    }


def _build_model_cost_rows(llm_cost_query) -> list[dict]:
    rows = (
        llm_cost_query.with_entities(
            LLMCallCost.model_name,
            func.coalesce(func.sum(LLMCallCost.total_cost_usd), 0.0),
            _atomic_call_count_expr(),
            func.coalesce(func.sum(LLMCallCost.total_tokens), 0),
        )
        .filter(LLMCallCost.model_name.isnot(None), LLMCallCost.status != "backfilled_aggregate")
        .group_by(LLMCallCost.model_name)
        .order_by(func.sum(LLMCallCost.total_cost_usd).desc())
        .all()
    )
    result: list[dict] = []
    for model_name, total_cost, call_count, total_tokens in rows[:8]:
        result.append({
            "model_name": model_name,
            "total_cost_usd": round(float(total_cost or 0.0), 4),
            "call_count": int(call_count or 0),
            "total_tokens": int(total_tokens or 0),
        })
    return result


def build_analytics_data(db: Session, range_key: str = "all") -> dict:
    """Backward-compatible wrapper for the renamed operations dashboard."""
    return build_operations_data(db, range_key=range_key)


def build_settings_data() -> dict:
    """Load knowledge pack content for the settings view."""
    try:
        from app.knowledge.mock_company_pack import (
            COMPANY_PROFILE,
            PRODUCT_CATEGORIES,
            ISSUE_TYPES,
            PRODUCT_TO_SUB_PRODUCT_TAXONOMY,
            ISSUE_TO_SUB_ISSUE_TAXONOMY,
            SEVERITY_RUBRIC,
            POLICY_SNIPPETS,
            ROUTING_MATRIX,
            ROOT_CAUSE_CONTROLS,
            deployment_label,
        )

        return {
            "deployment": deployment_label(),
            "company_profile": COMPANY_PROFILE,
            "product_categories": PRODUCT_CATEGORIES,
            "issue_types": ISSUE_TYPES,
            "product_to_sub_product": PRODUCT_TO_SUB_PRODUCT_TAXONOMY,
            "issue_to_sub_issue": ISSUE_TO_SUB_ISSUE_TAXONOMY,
            "severity_rubric": SEVERITY_RUBRIC,
            "policy_snippets": POLICY_SNIPPETS,
            "routing_rules": ROUTING_MATRIX,
            "root_cause_controls": ROOT_CAUSE_CONTROLS,
        }
    except Exception as e:
        logger.warning("Could not load company knowledge: %s", e)
        return {
            "deployment": "unknown",
            "company_profile": {},
            "product_categories": [],
            "issue_types": [],
            "product_to_sub_product": {},
            "issue_to_sub_issue": {},
            "severity_rubric": [],
            "policy_snippets": [],
            "routing_rules": {},
            "root_cause_controls": [],
        }


def build_evaluation_data() -> dict:
    """Load benchmark datasets, recent eval runs, and disagreement queue."""
    try:
        return _build_evaluation_dashboard_data()
    except Exception as e:
        logger.warning("Could not load evaluation dashboard data: %s", e)
        return {
            "terms": [],
            "source_datasets": [],
            "datasets": [],
            "recent_runs": [],
            "open_disagreements": [],
            "coverage": {
                "products": {},
                "issues": {},
                "narrative_lengths": {},
            },
            "dimension_scores": {
                "system_vs_gold": [],
                "judge_vs_gold": [],
            },
            "disagreement_reason_counts": {},
            "summary": {
                "source_dataset_count": 0,
                "dataset_count": 0,
                "recent_run_count": 0,
                "open_disagreement_count": 0,
                "pass_count": 0,
                "needs_review_count": 0,
                "fail_count": 0,
            },
        }


def build_evaluation_case_data(eval_case_id: str) -> dict | None:
    """Load the full evaluation-case detail payload."""
    try:
        return _build_evaluation_case_detail(eval_case_id)
    except Exception as e:
        logger.warning("Could not load evaluation case detail: %s", e)
        return None


def build_production_evaluation_case_data(case_id: str) -> dict | None:
    """Load the full production complaint evaluation report."""
    try:
        return _build_production_evaluation_case_detail(case_id)
    except Exception as e:
        logger.warning("Could not load production evaluation case detail: %s", e)
        return None
