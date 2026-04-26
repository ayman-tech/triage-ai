"""Redacted workflow state summaries and diffs for audit rows."""

from __future__ import annotations

import json
from typing import Any

from app.observability.redaction import json_safe, redact_narrative
from app.orchestrator.state import WorkflowState


def _case_summary(case: Any) -> dict[str, Any]:
    if case is None:
        return {}
    data = case.model_dump() if hasattr(case, "model_dump") else dict(case)
    nar = data.get("consumer_narrative")
    return {
        "id": data.get("id"),
        "status": str(data.get("status", "")),
        "consumer_narrative": redact_narrative(nar),
        "product": data.get("product"),
        "sub_product": data.get("sub_product"),
        "company": data.get("company"),
        "state": data.get("state"),
        "routed_to": data.get("routed_to"),
        "severity_class": data.get("severity_class"),
    }


def _classification_summary(clf: Any) -> dict[str, Any] | None:
    if clf is None:
        return None
    d = clf.model_dump() if hasattr(clf, "model_dump") else dict(clf)
    return {
        "product_category": d.get("product_category"),
        "issue_type": d.get("issue_type"),
        "confidence": d.get("confidence"),
        "review_recommended": d.get("review_recommended"),
        "reason_codes": d.get("reason_codes"),
    }


def _risk_summary(risk: Any) -> dict[str, Any] | None:
    if risk is None:
        return None
    d = risk.model_dump() if hasattr(risk, "model_dump") else dict(risk)
    return {
        "risk_level": d.get("risk_level"),
        "risk_score": d.get("risk_score"),
        "escalation_required": d.get("escalation_required"),
        "regulatory_risk": d.get("regulatory_risk"),
    }


def _company_context_counts(ctx: dict | None) -> dict[str, Any] | None:
    if not ctx:
        return None

    def _len_key(k: str) -> int:
        v = ctx.get(k)
        if v is None:
            return 0
        if isinstance(v, list):
            return len(v)
        if isinstance(v, dict):
            return len(v)
        return 1

    return {
        "taxonomy_candidates": _len_key("taxonomy_candidates"),
        "severity_candidates": _len_key("severity_candidates"),
        "policy_candidates": _len_key("policy_candidates"),
        "routing_candidates": _len_key("routing_candidates"),
        "root_cause_controls": _len_key("root_cause_controls"),
    }


def summarize_workflow_state(state: WorkflowState) -> dict[str, Any]:
    """Compact, redacted view suitable for JSON snapshots."""
    summary: dict[str, Any] = {
        "retry_count": state.get("retry_count", 0),
        "case": _case_summary(state.get("case")),
        "classification": _classification_summary(state.get("classification")),
        "risk_assessment": _risk_summary(state.get("risk_assessment")),
        "company_context": _company_context_counts(state.get("company_context")),
        "routed_to": state.get("routed_to"),
        "review_decision": (state.get("review") or {}).get("decision"),
        "compliance_passed": (state.get("compliance") or {}).get("passed"),
        "compliance_flags_count": len((state.get("compliance") or {}).get("flags", []) or []),
    }
    rc = state.get("root_cause_hypothesis")
    if rc is not None:
        rd = rc.model_dump() if hasattr(rc, "model_dump") else dict(rc)
        summary["root_cause"] = {
            "root_cause_category": rd.get("root_cause_category"),
            "confidence": rd.get("confidence"),
        }
    res = state.get("resolution")
    if res is not None:
        resd = res.model_dump() if hasattr(res, "model_dump") else dict(res)
        summary["resolution"] = {
            "recommended_action": resd.get("recommended_action"),
            "confidence": resd.get("confidence"),
            "estimated_resolution_days": resd.get("estimated_resolution_days"),
        }
    return json_safe(summary)


def diff_summaries(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Shallow field-level diff for top-level keys."""
    changes: list[dict[str, Any]] = []
    keys = set(before) | set(after)
    for k in sorted(keys):
        b, a = before.get(k), after.get(k)
        if b != a:
            changes.append({"field": k, "old": b, "new": a})
    summary_bits = [f"{c['field']}: updated" for c in changes[:8]]
    if len(changes) > 8:
        summary_bits.append(f"... +{len(changes) - 8} more")
    return {
        "changes": changes,
        "human_summary": "; ".join(summary_bits) if summary_bits else "no changes",
    }


def dumps_compact(obj: Any) -> str:
    return json.dumps(obj, default=str, separators=(",", ":"))
