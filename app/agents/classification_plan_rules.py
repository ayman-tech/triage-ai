"""Rule-based ClassificationPlan from SituationAssessment (no LLM).

v2 dual-hypothesis (see product plan): when assessment indicates contradiction,
multi_issue, or high conflict_score, the audit package sets
``v2_dual_hypothesis_eligible``; a future pass may run structured-led vs
narrative-led micro-classify and reconcile. v1 uses a single Execute pass only.
"""

from __future__ import annotations

from app.schemas.classification_pipeline import (
    ClassificationPlan,
    ClassificationStrategy,
    Complexity,
    Consistency,
    EvidenceWeighting,
    SituationAssessment,
)


def plan_from_assessment(assessment: SituationAssessment) -> ClassificationPlan:
    """Map assessment → deterministic action policy."""

    cx = assessment.complexity
    cons = assessment.consistency
    cs = assessment.conflict_score
    weight = assessment.recommended_weighting

    # High conflict / contradiction → heavy retrieval
    if cons == Consistency.CONTRADICTION or cs >= 0.65:
        return ClassificationPlan(
            strategy=ClassificationStrategy.RETRIEVAL_DISAMBIGUATION,
            tool_budget=6,
            needs_retrieval=True,
            needs_human_review_hint=True,
        )

    # Straightforward / trivial cases MUST be checked before the generic
    # "narrative weighting ⇒ retrieval" rule, otherwise Assess often labels
    # weighting=narrative for every narrative and we never reach no-tool plans.
    if cx == Complexity.STRAIGHTFORWARD:
        return ClassificationPlan(
            strategy=ClassificationStrategy.MAPPING_PLUS_NARRATIVE_CONFIRM,
            tool_budget=0,
            needs_retrieval=False,
        )

    if cx == Complexity.TRIVIAL and cons in (Consistency.UNKNOWN, Consistency.ALIGNED):
        if weight == EvidenceWeighting.STRUCTURED:
            return ClassificationPlan(
                strategy=ClassificationStrategy.MAPPING_ONLY,
                tool_budget=0,
                needs_retrieval=False,
            )
        return ClassificationPlan(
            strategy=ClassificationStrategy.MAPPING_PLUS_NARRATIVE_CONFIRM,
            tool_budget=0,
            needs_retrieval=False,
        )

    if cx == Complexity.MULTI_ISSUE:
        return ClassificationPlan(
            strategy=ClassificationStrategy.RETRIEVAL_DISAMBIGUATION,
            tool_budget=6,
            needs_retrieval=True,
            needs_human_review_hint=cs >= 0.45,
        )

    if cx in (Complexity.AMBIGUOUS, Complexity.UNDER_SPECIFIED):
        return ClassificationPlan(
            strategy=ClassificationStrategy.RETRIEVAL_DISAMBIGUATION,
            tool_budget=5,
            needs_retrieval=True,
            needs_human_review_hint=cs >= 0.5,
        )

    if assessment.recommended_weighting == EvidenceWeighting.NARRATIVE:
        return ClassificationPlan(
            strategy=ClassificationStrategy.RETRIEVAL_DISAMBIGUATION,
            tool_budget=6,
            needs_retrieval=True,
            needs_human_review_hint=cs >= 0.45,
        )

    if cons == Consistency.PARTIAL_CONFLICT:
        return ClassificationPlan(
            strategy=ClassificationStrategy.NARRATIVE_LED,
            tool_budget=5,
            needs_retrieval=True,
            needs_human_review_hint=cs >= 0.4,
        )

    return ClassificationPlan(
        strategy=ClassificationStrategy.MAPPING_PLUS_NARRATIVE_CONFIRM,
        tool_budget=4,
        needs_retrieval=True,
    )
