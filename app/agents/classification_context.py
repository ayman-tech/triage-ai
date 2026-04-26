"""Deterministic signals for structured-intake classification Assess phase."""

from __future__ import annotations

import re
from typing import Any

# Minimum characters for "rich" narrative (matches CaseCreate threshold).
RICH_NARRATIVE_MIN = 10


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def build_deterministic_signals(case_like: dict[str, Any]) -> dict[str, Any]:
    """Compute cheap signals from a case dict or CaseRead.model_dump()."""
    narrative = (case_like.get("consumer_narrative") or "").strip()
    n_len = len(narrative)

    cfpb_p = (case_like.get("cfpb_product") or "").strip()
    cfpb_sp = (case_like.get("cfpb_sub_product") or "").strip()
    cfpb_i = (case_like.get("cfpb_issue") or "").strip()
    cfpb_si = (case_like.get("cfpb_sub_issue") or "").strip()
    product = (case_like.get("product") or "").strip()
    sub_product = (case_like.get("sub_product") or "").strip()

    structured_parts = [cfpb_p, cfpb_sp, cfpb_i, cfpb_si, product, sub_product]
    structured_blob = " ".join(p for p in structured_parts if p).lower()

    ext_issue = ""
    ext = case_like.get("external_schema")
    if isinstance(ext, dict):
        ext_issue = (ext.get("external_issue_type") or "").strip().lower()

    narrative_rich = n_len >= RICH_NARRATIVE_MIN
    narrative_status = "absent"
    if n_len == 0:
        narrative_status = "absent"
    elif not narrative_rich:
        narrative_status = "short"
    else:
        narrative_status = "present"

    structured_field_count = sum(1 for p in [cfpb_p, cfpb_sp, cfpb_i, cfpb_si] if p)
    structured_complete_core = bool(cfpb_p and cfpb_i)

    st_tokens = _tokens(structured_blob + " " + ext_issue)
    na_tokens = _tokens(narrative)
    overlap = len(st_tokens & na_tokens) if st_tokens and na_tokens else 0
    union = len(st_tokens | na_tokens) if (st_tokens or na_tokens) else 1
    jaccard = overlap / union if union else 0.0

    # Heuristic tension: low overlap with rich narrative + non-empty structured → possible conflict
    tension = 0.0
    if narrative_rich and structured_blob:
        tension = max(0.0, 1.0 - jaccard * 2.0)  # scale up disagreement signal
        tension = min(1.0, tension)

    multi_issue_hint = False
    if narrative_rich:
        multi_issue_hint = bool(
            re.search(
                r"\b(and also|another issue|second problem|also they|in addition)\b",
                narrative.lower(),
            )
        )

    return {
        "narrative_length": n_len,
        "narrative_rich": narrative_rich,
        "narrative_status": narrative_status,
        "structured_field_count": structured_field_count,
        "structured_complete_core": structured_complete_core,
        "structured_blob_preview": structured_blob[:400],
        "token_overlap_jaccard": round(jaccard, 4),
        "structured_narrative_tension": round(tension, 4),
        "multi_issue_hint": multi_issue_hint,
    }


def should_skip_assess_llm(signals: dict[str, Any]) -> bool:
    """Trivial cases: use template SituationAssessment (no Assess LLM)."""
    if signals["narrative_status"] == "absent" and signals["structured_complete_core"]:
        return True
    if (
        signals["narrative_rich"]
        and signals["structured_complete_core"]
        and signals["structured_narrative_tension"] < 0.25
        and not signals["multi_issue_hint"]
    ):
        return True
    return False


def template_situation_assessment(signals: dict[str, Any]) -> dict[str, Any]:
    """Build SituationAssessment-compatible dict without LLM."""
    ns = signals["narrative_status"]
    if ns == "absent" and signals["structured_complete_core"]:
        return {
            "complexity": "trivial",
            "narrative_status": "absent",
            "structured_field_completeness": "core_product_issue",
            "consistency": "unknown",
            "conflict_score": 0.0,
            "recommended_weighting": "structured",
            "rationale": "No narrative; classify from structured intake fields and taxonomy mapping only.",
        }
    if (
        signals["narrative_rich"]
        and signals["structured_complete_core"]
        and signals["structured_narrative_tension"] < 0.25
    ):
        return {
            "complexity": "straightforward",
            "narrative_status": "present",
            "structured_field_completeness": "core_fields_plus_narrative",
            "consistency": "aligned",
            "conflict_score": round(signals["structured_narrative_tension"], 2),
            "recommended_weighting": "balanced",
            "rationale": "Narrative and structured intake fields appear aligned (high token overlap).",
        }
    # Fallback: should not be called if should_skip_assess_llm is correct; still safe default
    return {
        "complexity": "ambiguous",
        "narrative_status": ns if ns in ("absent", "short", "present") else "present",
        "structured_field_completeness": "partial",
        "consistency": "unknown",
        "conflict_score": round(signals.get("structured_narrative_tension", 0.5), 2),
        "recommended_weighting": "balanced",
        "rationale": "Heuristic template fallback; use full Assess in production if this appears.",
    }
