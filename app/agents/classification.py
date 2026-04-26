"""Classification agent – Assess → Plan → Execute → Verify."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.agents.adk_runner import run_adk_json_agent
from app.agents.classification_context import (
    build_deterministic_signals,
    should_skip_assess_llm,
    template_situation_assessment,
)
from app.agents.classification_deterministic import (
    build_template_classification_result,
    enrich_operational_sub_labels,
    should_skip_execute_llm,
)
from app.agents.classification_plan_rules import plan_from_assessment
from app.knowledge import CompanyKnowledgeService
from app.agents.tools import (
    get_case_document_facts,
    lookup_company_taxonomy,
    search_case_documents,
    search_similar_complaints,
)
from app.schemas.case import CaseRead
from app.schemas.classification import ClassificationResult
from app.schemas.classification_pipeline import (
    ClassificationAuditPackage,
    ClassificationPipelineOutput,
    ClassificationPlan,
    ClassificationStrategy,
    Complexity,
    Consistency,
    EvidenceWeighting,
    SituationAssessment,
)

logger = logging.getLogger(__name__)
_company_knowledge: CompanyKnowledgeService | None = None


def _company_knowledge_service() -> CompanyKnowledgeService:
    global _company_knowledge
    if _company_knowledge is None:
        _company_knowledge = CompanyKnowledgeService()
    return _company_knowledge


_CLASSIFICATION_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "classification.md"
_ASSESS_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "classification_assess.md"

_STRATEGY_INSTRUCTIONS = {
    ClassificationStrategy.MAPPING_ONLY: (
        "Strategy: **mapping_only** — Map structured intake fields to internal taxonomy first; "
        "treat the narrative as secondary or absent."
    ),
    ClassificationStrategy.MAPPING_PLUS_NARRATIVE_CONFIRM: (
        "Strategy: **mapping_plus_narrative_confirm** — Use structured fields as priors, then "
        "confirm or adjust using the narrative and tools."
    ),
    ClassificationStrategy.NARRATIVE_LED: (
        "Strategy: **narrative_led** — Prefer the consumer narrative when it conflicts with "
        "portal menu selections; use tools to resolve ambiguity."
    ),
    ClassificationStrategy.RETRIEVAL_DISAMBIGUATION: (
        "Strategy: **retrieval_disambiguation** — Heavy use of similar-complaint search and "
        "taxonomy tools to disambiguate conflicting or sparse evidence."
    ),
    ClassificationStrategy.LOW_CONFIDENCE_RETURN: (
        "Strategy: **low_confidence_return** — Prefer broader/other categories and "
        "low confidence rather than over-fitting sparse evidence."
    ),
}

_WEIGHTING_BLURB = {
    EvidenceWeighting.STRUCTURED: "Evidence weighting: **structured** — prioritize structured intake fields.",
    EvidenceWeighting.NARRATIVE: "Evidence weighting: **narrative** — prioritize free-text narrative.",
    EvidenceWeighting.BALANCED: "Evidence weighting: **balanced** — reconcile structured fields and narrative.",
}


def _load_classification_prompt() -> str:
    return _CLASSIFICATION_PROMPT_PATH.read_text(encoding="utf-8")


def _load_assess_prompt() -> str:
    return _ASSESS_PROMPT_PATH.read_text(encoding="utf-8")


def _classification_query_text(case: CaseRead) -> str:
    parts = [
        case.consumer_narrative,
        case.cfpb_product,
        case.cfpb_sub_product,
        case.cfpb_issue,
        case.cfpb_sub_issue,
        case.product,
        case.sub_product,
    ]
    return " ".join(str(part).strip() for part in parts if part).strip()


def _taxonomy_candidates_for_case(case: CaseRead) -> dict:
    narrative = _classification_query_text(case)
    ctx = _company_knowledge_service().build_company_context(narrative)
    return ctx.taxonomy_candidates


def _run_assess_llm(signals: dict, case: CaseRead) -> SituationAssessment:
    system = _load_assess_prompt()
    case_summary = {
        "consumer_narrative": case.consumer_narrative,
        "product": case.product,
        "sub_product": case.sub_product,
        "cfpb_product": case.cfpb_product,
        "cfpb_sub_product": case.cfpb_sub_product,
        "cfpb_issue": case.cfpb_issue,
        "cfpb_sub_issue": case.cfpb_sub_issue,
        "company": case.company,
        "state": case.state,
    }
    trimmed = {k: v for k, v in case_summary.items() if v}
    human = (
        "SIGNALS_JSON:\n"
        f"{json.dumps(signals, indent=2)}\n\n"
        "CASE_SUMMARY_JSON:\n"
        f"{json.dumps(trimmed, indent=2)}"
    )

    data = run_adk_json_agent(
        name="classification_assessment_agent",
        description="Assesses complaint evidence quality before classification planning.",
        instruction=system,
        user_message=human,
        tools=[],
    )
    return SituationAssessment.model_validate(data)


def _build_execute_user_message(
    case: CaseRead,
    assessment: SituationAssessment,
    plan: ClassificationPlan,
    instructions: str,
    taxonomy_candidates: dict,
    tools_available: list[str],
) -> str:
    strategy_line = _STRATEGY_INSTRUCTIONS.get(
        plan.strategy,
        _STRATEGY_INSTRUCTIONS[ClassificationStrategy.MAPPING_PLUS_NARRATIVE_CONFIRM],
    )
    weight_line = _WEIGHTING_BLURB.get(
        assessment.recommended_weighting,
        _WEIGHTING_BLURB[EvidenceWeighting.BALANCED],
    )
    lines = [
        "## Case evidence",
        "",
        "**Structured intake fields** (may be incomplete or wrong):",
        f"- cfpb_product: {case.cfpb_product or 'N/A'}",
        f"- cfpb_sub_product: {case.cfpb_sub_product or 'N/A'}",
        f"- cfpb_issue: {case.cfpb_issue or 'N/A'}",
        f"- cfpb_sub_issue: {case.cfpb_sub_issue or 'N/A'}",
        "",
        "**Legacy / free-form hints:**",
        f"- product: {case.product or 'N/A'}",
        f"- sub_product: {case.sub_product or 'N/A'}",
        f"- company: {case.company or 'N/A'}",
        f"- state: {case.state or 'N/A'}",
        "",
        "**Consumer complaint narrative** (may be empty or short — do not invent facts):",
        case.consumer_narrative or "[empty]",
        "",
        "## Document evidence",
        json.dumps(case.case_document_summary or {}, indent=2),
        "",
        "## Deterministic document consistency check",
        json.dumps(case.document_consistency or {}, indent=2),
        "",
        "## Plan (follow strictly)",
        strategy_line,
        weight_line,
        f"- Situation complexity (assessment): {assessment.complexity.value}",
        f"- Consistency (structured vs narrative): {assessment.consistency.value}",
        f"- Conflict score (0–1): {assessment.conflict_score:.2f}",
        f"- Tool budget (max rounds): {plan.tool_budget}",
        f"- Retrieval emphasis: {plan.needs_retrieval}",
        "",
        "## Company taxonomy candidates",
        json.dumps(taxonomy_candidates, indent=2),
        "",
        f"Available tools: {', '.join(tools_available) if tools_available else 'none'}",
    ]
    if instructions:
        lines.extend(["", f"Supervisor instructions: {instructions}"])
    lines.extend(
        [
            "",
            "If tools are available, use them only when the plan requires extra evidence. "
            "When finished, respond with **only** the classification JSON per the system schema.",
        ]
    )
    return "\n".join(lines)


def _select_execution_tools(plan: ClassificationPlan) -> list:
    tools = []
    if plan.needs_retrieval and plan.tool_budget > 0:
        tools.extend([search_similar_complaints, lookup_company_taxonomy])
    return tools


def _run_execute_no_tools(system_prompt: str, user_message: str) -> dict:
    return run_adk_json_agent(
        name="classification_agent",
        description="Classifies financial complaint product and issue taxonomy from structured and narrative evidence.",
        instruction=system_prompt,
        user_message=user_message,
        tools=[],
    )


def _v2_dual_hypothesis_eligible(assessment: SituationAssessment) -> bool:
    return (
        assessment.consistency == Consistency.CONTRADICTION
        or assessment.complexity == Complexity.MULTI_ISSUE
        or assessment.conflict_score >= 0.65
    )


def _verify_classification(
    result: ClassificationResult,
    assessment: SituationAssessment,
    plan: ClassificationPlan,
    signals: dict,
) -> ClassificationResult:
    codes = list(result.reason_codes)
    review = result.review_recommended

    if not signals.get("narrative_rich"):
        codes.append("narrative_absent_or_short")

    if plan.needs_human_review_hint:
        if "plan_needs_human_review_hint" not in codes:
            codes.append("plan_needs_human_review_hint")
        review = True

    if result.confidence < 0.55:
        if "below_confidence_floor_0_55" not in codes:
            codes.append("below_confidence_floor_0_55")
        review = True

    if assessment.consistency == Consistency.CONTRADICTION:
        if "assessment_contradiction" not in codes:
            codes.append("assessment_contradiction")
        review = True
    elif assessment.conflict_score >= 0.55 and result.confidence >= 0.85:
        if "high_conflict_high_model_confidence" not in codes:
            codes.append("high_conflict_high_model_confidence")
        review = True

    if assessment.complexity in (Complexity.AMBIGUOUS, Complexity.UNDER_SPECIFIED) and result.confidence >= 0.9:
        if "high_confidence_despite_under_specified" not in codes:
            codes.append("high_confidence_despite_under_specified")
        review = True

    return result.model_copy(update={"reason_codes": codes, "review_recommended": review})


def run_classification(
    *,
    case: CaseRead | None = None,
    narrative: str | None = None,
    product: str | None = None,
    sub_product: str | None = None,
    company: str | None = None,
    state: str | None = None,
    instructions: str = "",
    model_name: str | None = None,
    temperature: float = 0.0,
) -> ClassificationPipelineOutput:
    """Classify using Assess → Plan (rules) → Execute → Verify (code)."""
    if case is None:
        if narrative is None:
            raise ValueError("Provide ``case`` or ``narrative``.")
        case = CaseRead(
            consumer_narrative=(narrative or "").strip(),
            product=product,
            sub_product=sub_product,
            company=company,
            state=state,
        )

    logger.info("Classification pipeline starting")

    case_dict = case.model_dump()
    signals = build_deterministic_signals(case_dict)
    assess_skipped = should_skip_assess_llm(signals)

    if assess_skipped:
        assessment = SituationAssessment.model_validate(
            template_situation_assessment(signals)
        )
    else:
        assessment = _run_assess_llm(signals, case)

    plan = plan_from_assessment(assessment)
    taxonomy_candidates = _taxonomy_candidates_for_case(case)
    has_documents = bool((case.case_document_summary or {}).get("total_documents"))

    execute_skipped = False
    if should_skip_execute_llm(signals, assessment, plan) and (
        taxonomy_candidates.get("product_categories")
        and taxonomy_candidates.get("issue_types")
    ) and not has_documents:
        result = build_template_classification_result(case, signals, taxonomy_candidates)
        evidence_used: dict[str, bool] = {}
        execute_skipped = True
    else:
        system_prompt = _load_classification_prompt()
        tools = _select_execution_tools(plan)
        if has_documents and getattr(case, "id", None):
            tools = [*tools, get_case_document_facts, search_case_documents]
        user_message = _build_execute_user_message(
            case,
            assessment,
            plan,
            instructions,
            taxonomy_candidates,
            [fn.__name__ for fn in tools],
        )
        if tools:
            exec_out = run_adk_json_agent(
                name="classification_agent",
                description=(
                    "Classifies financial complaint product and issue taxonomy "
                    "using retrieval, taxonomy, and document tools when needed."
                ),
                instruction=system_prompt,
                user_message=user_message,
                tools=tools,
                return_evidence=True,
            )
            result_dict, evidence_used = exec_out  # type: ignore[misc]
        else:
            result_dict = _run_execute_no_tools(system_prompt, user_message)
            evidence_used = {}
        result = ClassificationResult(**result_dict)

    result = enrich_operational_sub_labels(result, case)
    result = _verify_classification(result, assessment, plan, signals)
    consistency = case.document_consistency or {}
    if consistency.get("status") == "contradiction":
        codes = list(result.reason_codes)
        if "document_contradiction_detected" not in codes:
            codes.append("document_contradiction_detected")
        result = result.model_copy(
            update={
                "reason_codes": codes,
                "review_recommended": True,
                "confidence": min(float(result.confidence), 0.65),
            }
        )

    v2_flag = _v2_dual_hypothesis_eligible(assessment)

    audit = ClassificationAuditPackage(
        situation_assessment=assessment.model_dump(mode="json"),
        plan=plan.model_dump(mode="json"),
        deterministic_signals=signals,
        evidence_used=evidence_used,
        consistency_assessment=assessment.consistency.value,
        alternate_candidates=list(result.alternate_candidates),
        review_recommended=result.review_recommended,
        reason_codes=list(result.reason_codes),
        assess_skipped_llm=assess_skipped,
        plan_skipped_llm=False,
        execute_skipped_llm=execute_skipped,
        v2_dual_hypothesis_eligible=v2_flag,
    )

    logger.info(
        "Classification complete – category=%s, confidence=%.2f, review=%s",
        result.product_category,
        result.confidence,
        result.review_recommended,
    )
    return ClassificationPipelineOutput(result=result, audit=audit)
