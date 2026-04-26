"""Structured artifacts for complaint classification (Assess → Plan → Verify)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.schemas.classification import ClassificationResult


class NarrativeStatus(str, Enum):
    ABSENT = "absent"
    SHORT = "short"  # present but below minimum rich-text threshold
    PRESENT = "present"


class Complexity(str, Enum):
    TRIVIAL = "trivial"
    STRAIGHTFORWARD = "straightforward"
    AMBIGUOUS = "ambiguous"
    CONTRADICTORY = "contradictory"
    MULTI_ISSUE = "multi_issue"
    UNDER_SPECIFIED = "under_specified"


class Consistency(str, Enum):
    ALIGNED = "aligned"
    PARTIAL_CONFLICT = "partial_conflict"
    CONTRADICTION = "contradiction"
    UNKNOWN = "unknown"  # e.g. no narrative to compare


class EvidenceWeighting(str, Enum):
    STRUCTURED = "structured"
    NARRATIVE = "narrative"
    BALANCED = "balanced"


class ClassificationStrategy(str, Enum):
    MAPPING_ONLY = "mapping_only"
    MAPPING_PLUS_NARRATIVE_CONFIRM = "mapping_plus_narrative_confirm"
    NARRATIVE_LED = "narrative_led"
    RETRIEVAL_DISAMBIGUATION = "retrieval_disambiguation"
    LOW_CONFIDENCE_RETURN = "low_confidence_return"


class SituationAssessment(BaseModel):
    """Output of Assess phase (LLM or deterministic template)."""

    complexity: Complexity
    narrative_status: NarrativeStatus
    structured_field_completeness: str = Field(
        ...,
        description="e.g. all_four | product_issue_only | sparse",
    )
    consistency: Consistency
    conflict_score: float = Field(..., ge=0.0, le=1.0)
    recommended_weighting: EvidenceWeighting
    rationale: str = Field(..., max_length=800)


class ClassificationPlan(BaseModel):
    """Action policy for Execute phase."""

    strategy: ClassificationStrategy
    tool_budget: int = Field(..., ge=0, le=10)
    needs_retrieval: bool = False
    needs_human_review_hint: bool = False


class ClassificationAuditPackage(BaseModel):
    """Persisted audit trail for a classification run (Verify + context)."""

    situation_assessment: dict[str, Any]
    plan: dict[str, Any]
    deterministic_signals: dict[str, Any] = Field(default_factory=dict)
    evidence_used: dict[str, bool] = Field(
        default_factory=dict,
        description="Flags such as taxonomy_tool, similar_complaints_tool",
    )
    consistency_assessment: str = ""
    alternate_candidates: list[dict[str, Any]] = Field(default_factory=list)
    review_recommended: bool = False
    reason_codes: list[str] = Field(default_factory=list)
    assess_skipped_llm: bool = False
    plan_skipped_llm: bool = False
    execute_skipped_llm: bool = Field(
        default=False,
        description="True when classification was mapped without the Execute LLM/tools.",
    )

    # v2: dual-hypothesis reconciliation (documented; not run in v1)
    v2_dual_hypothesis_eligible: bool = Field(
        default=False,
        description="If true, a future pipeline may run structured-led vs narrative-led micro-classify.",
    )


class ClassificationPipelineOutput(BaseModel):
    """Full classification run: operational labels + audit sidecar."""

    result: ClassificationResult
    audit: ClassificationAuditPackage
