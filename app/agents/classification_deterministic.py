"""Deterministic classification when Assess + Plan indicate no retrieval is needed.

Skips the Execute LLM + tool loop for trivial/straightforward cases and maps labels
from ranked taxonomy candidates plus OPERATIONAL_TAXONOMY sub-product / sub-issue lists.
"""

from __future__ import annotations

import re
from typing import Any

from app.knowledge.mock_company_pack import OPERATIONAL_TAXONOMY
from app.schemas.case import CaseRead
from app.schemas.classification import ClassificationResult
from app.schemas.classification_pipeline import (
    ClassificationPlan,
    Complexity,
    Consistency,
    EvidenceWeighting,
    SituationAssessment,
)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def _score_query_against_cues(query: str, cues: list[str]) -> float:
    q_tokens = _tokens(query)
    if not q_tokens or not cues:
        return 0.0
    cue_tokens: set[str] = set()
    for c in cues:
        cue_tokens |= _tokens(c)
    if not cue_tokens:
        return 0.0
    overlap = q_tokens & cue_tokens
    return len(overlap) / max(1, len(cue_tokens))


def _case_query_text(case: CaseRead) -> str:
    parts = [
        case.consumer_narrative,
        case.cfpb_product,
        case.cfpb_sub_product,
        case.cfpb_issue,
        case.cfpb_sub_issue,
        case.product,
        case.sub_product,
    ]
    return " ".join(str(p).strip() for p in parts if p).strip()


def should_skip_execute_llm(
    signals: dict[str, Any],
    assessment: SituationAssessment,
    plan: ClassificationPlan,
) -> bool:
    """True when no tools/LLM execute pass is needed — map from taxonomy candidates only."""
    if plan.needs_retrieval or plan.tool_budget > 0:
        return False
    if signals.get("multi_issue_hint"):
        return False
    if assessment.complexity not in (Complexity.TRIVIAL, Complexity.STRAIGHTFORWARD):
        return False
    if assessment.conflict_score >= 0.35:
        return False
    if assessment.consistency in (Consistency.CONTRADICTION, Consistency.PARTIAL_CONFLICT):
        return False
    # Narrative-led weighting usually wants a model pass unless assess was template-aligned
    if assessment.recommended_weighting == EvidenceWeighting.NARRATIVE and signals.get(
        "narrative_rich"
    ):
        return False
    return True


def _best_sub_product(product_category: str, query: str) -> str | None:
    p2s = OPERATIONAL_TAXONOMY.get("product_to_sub_product_taxonomy") or {}
    rows = p2s.get(product_category) or []
    if not rows:
        return None
    best_key: str | None = None
    best_score = -1.0
    for row in rows:
        key = row.get("sub_product") or row.get("sup_product")
        if not key:
            continue
        cues = list(row.get("cues") or [])
        s = _score_query_against_cues(query, cues)
        if s > best_score:
            best_score = s
            best_key = str(key)
    if best_key and best_score > 0:
        return best_key
    # Fallback: first listed sub-product for stable mapping
    first = rows[0]
    return str(first.get("sub_product") or first.get("sup_product") or "") or None


def _best_sub_issue(
    issue_type: str,
    product_category: str,
    query: str,
) -> str | None:
    i2s = OPERATIONAL_TAXONOMY.get("issue_to_sub_issue_taxonomy") or {}
    rows = i2s.get(issue_type) or []
    if not rows:
        return None
    best_key: str | None = None
    best_score = -1.0
    for row in rows:
        key = row.get("sub_issue")
        if not key:
            continue
        products = list(row.get("applicable_products") or [])
        if products and product_category not in products:
            continue
        cues = list(row.get("cues") or [])
        s = _score_query_against_cues(query, cues)
        if s > best_score:
            best_score = s
            best_key = str(key)
    if best_key and best_score > 0:
        return best_key
    # Prefer first applicable row for this product, else first row
    for row in rows:
        products = list(row.get("applicable_products") or [])
        key = row.get("sub_issue")
        if not key:
            continue
        if not products or product_category in products:
            return str(key)
    return str(rows[0]["sub_issue"]) if rows and rows[0].get("sub_issue") else None


def build_template_classification_result(
    case: CaseRead,
    signals: dict[str, Any],
    taxonomy_candidates: dict[str, Any],
) -> ClassificationResult:
    """Build ClassificationResult without calling the Execute LLM."""
    products = taxonomy_candidates.get("product_categories") or []
    issues = taxonomy_candidates.get("issue_types") or []
    top_p = (products[0].get("product_category") if products else None) or "other"
    top_i = (issues[0].get("issue_type") if issues else None) or "other"

    query = _case_query_text(case)
    sub_p = _best_sub_product(top_p, query)
    sub_i = _best_sub_issue(top_i, top_p, query)

    conf = 0.78
    if signals.get("narrative_status") == "absent":
        conf = 0.72
    elif signals.get("narrative_status") == "short":
        conf = 0.68

    return ClassificationResult(
        product_category=top_p,
        issue_type=top_i,
        sub_product=sub_p,
        sub_issue=sub_i,
        confidence=conf,
        reasoning=(
            "Deterministic mapping from ranked operational taxonomy candidates and "
            "sub-product/sub-issue cue matching (execute LLM skipped for low-ambiguity case)."
        ),
        keywords=_keywords_from_case(case),
        review_recommended=False,
        reason_codes=["execute_llm_skipped", "deterministic_operational_map"],
        alternate_candidates=[],
    )


def enrich_operational_sub_labels(result: ClassificationResult, case: CaseRead) -> ClassificationResult:
    """Fill missing sub_product / sub_issue using operational taxonomy cue matching."""
    pc = result.product_category.value
    it = result.issue_type.value
    query = _case_query_text(case)
    sub_p = (result.sub_product or "").strip() or None
    sub_i = (result.sub_issue or "").strip() or None
    if not sub_p:
        sub_p = _best_sub_product(pc, query)
    if not sub_i:
        sub_i = _best_sub_issue(it, pc, query)
    if sub_p == result.sub_product and sub_i == result.sub_issue:
        return result
    return result.model_copy(update={"sub_product": sub_p, "sub_issue": sub_i})


def _keywords_from_case(case: CaseRead) -> list[str]:
    parts = [
        case.cfpb_product,
        case.cfpb_issue,
        case.product,
        case.sub_product,
    ]
    raw = " ".join(str(p).strip() for p in parts if p).strip()
    if raw:
        return [raw[:80]]
    nar = (case.consumer_narrative or "").strip()
    if len(nar) > 80:
        return [nar[:80]]
    return [nar] if nar else ["intake"]
