"""DB-backed benchmark datasets, runners, and review helpers."""

from __future__ import annotations

import json
import logging
import uuid
import random
from collections import defaultdict
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from app.db.models import (
    CaseDocument,
    ComplaintCase,
    ComplaintEvaluationReport,
    EvaluationCase,
    EvaluationDataset,
    EvaluationDisagreement,
    EvaluationGoldLabel,
    EvaluationJudgeRun,
    EvaluationReviewRecord,
    EvaluationRun,
    EvaluationSystemPrediction,
    DocumentArtifact,
    ClassificationRecord,
    RiskRecord,
    ResolutionRecord,
    SourceDataset,
    SourceDatasetItem,
)
from app.db.session import SessionLocal
from app.evals.judge import run_rubric_judge
from app.knowledge.company_knowledge import CompanyKnowledgeService
from app.observability.versions import (
    default_chat_model,
    knowledge_pack_version,
    llm_provider,
    prompt_bundle_version,
    workflow_version,
)
from app.orchestrator.workflow import process_complaint
from app.retrieval.ingest import (
    COL_COMPLAINT_ID,
    COL_COMPANY,
    COL_COMPANY_PUBLIC_RESPONSE,
    COL_DATE_RECEIVED,
    COL_ISSUE,
    COL_NARRATIVE,
    COL_PRODUCT,
    COL_RESPONSE,
    COL_STATE,
    COL_SUB_ISSUE,
    COL_SUB_PRODUCT,
    COL_SUBMITTED_VIA,
    DEFAULT_CSV,
    MIN_NARRATIVE_LENGTH,
)
from app.utils.pii import redact_pii

logger = logging.getLogger(__name__)

EVALUATION_TERM_DESCRIPTIONS = [
    {
        "term": "Source Dataset",
        "description": "The sampled external corpus used to build benchmark cases. Here it is a stratified CFPB sample acting as mock-bank complaint data.",
    },
    {
        "term": "Evaluation Dataset",
        "description": "A test set materialized from the source dataset. Each row becomes a benchmark case the workflow can be run against.",
    },
    {
        "term": "Weak Gold",
        "description": "Expected labels derived heuristically from source data and company taxonomy, useful for regression tracking but not yet human-adjudicated truth.",
    },
    {
        "term": "Judge",
        "description": "A rubric-based evaluator that inspects the system output and scores grounding, completeness, contradiction handling, and calibration.",
    },
    {
        "term": "Disagreement Queue",
        "description": "Cases where system output, weak gold, and judge output do not align, and which should be reviewed by a human.",
    },
]


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _json_loads(raw: str | None) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _normalize(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _normalize(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, tuple):
        return [_normalize(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _clean_cfpb_narrative(row: dict[str, Any]) -> str:
    narrative = (row.get(COL_NARRATIVE) or "").strip()
    if len(narrative) < MIN_NARRATIVE_LENGTH:
        return ""
    return redact_pii(narrative)


def _collect_stratum_counts(csv_path: Path) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    with open(csv_path, newline="", encoding="utf-8") as f:
        import csv

        reader = csv.DictReader(f)
        for row in reader:
            narrative = _clean_cfpb_narrative(row)
            if not narrative:
                continue
            key = (str(row.get(COL_PRODUCT) or "unknown"), str(row.get(COL_ISSUE) or "unknown"))
            counts[key] += 1
    return counts


def _allocate_stratified_targets(counts: dict[tuple[str, str], int], sample_size: int) -> dict[tuple[str, str], int]:
    if not counts:
        return {}
    total = sum(counts.values())
    allocations: dict[tuple[str, str], int] = {}
    guaranteed = min(len(counts), sample_size)
    remaining = max(0, sample_size - guaranteed)
    for key, count in counts.items():
        allocations[key] = min(1, count)
        if remaining <= 0:
            continue
        allocations[key] += min(count - allocations[key], int(remaining * (count / total)))
    allocated = sum(allocations.values())
    if allocated < sample_size:
        leftovers = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        idx = 0
        while allocated < sample_size and leftovers:
            key, count = leftovers[idx % len(leftovers)]
            if allocations[key] < count:
                allocations[key] += 1
                allocated += 1
            idx += 1
    return allocations


def _reservoir_sample_cfpb_rows(
    csv_path: Path,
    targets: dict[tuple[str, str], int],
    seed: int,
) -> list[dict[str, Any]]:
    reservoirs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: dict[tuple[str, str], int] = defaultdict(int)
    rng = random.Random(seed)

    with open(csv_path, newline="", encoding="utf-8") as f:
        import csv

        reader = csv.DictReader(f)
        for row in reader:
            narrative = _clean_cfpb_narrative(row)
            if not narrative:
                continue
            key = (str(row.get(COL_PRODUCT) or "unknown"), str(row.get(COL_ISSUE) or "unknown"))
            target = targets.get(key, 0)
            if target <= 0:
                continue

            normalized = dict(row)
            normalized[COL_NARRATIVE] = narrative
            seen[key] += 1
            bucket = reservoirs[key]
            if len(bucket) < target:
                bucket.append(normalized)
                continue

            index = rng.randint(0, seen[key] - 1)
            if index < target:
                bucket[index] = normalized

    rows: list[dict[str, Any]] = []
    for bucket in reservoirs.values():
        rows.extend(bucket)
    rng.shuffle(rows)
    return rows


def _weak_gold_from_source_row(row: dict[str, Any]) -> dict[str, Any]:
    svc = CompanyKnowledgeService()
    query = " ".join(
        str(part).strip()
        for part in (
            row.get(COL_PRODUCT),
            row.get(COL_SUB_PRODUCT),
            row.get(COL_ISSUE),
            row.get(COL_SUB_ISSUE),
            row.get(COL_NARRATIVE),
        )
        if part
    )
    ctx = svc.build_company_context(query)
    top_products = ctx.taxonomy_candidates.get("product_categories") or []
    top_issues = ctx.taxonomy_candidates.get("issue_types") or []
    return {
        "classification": {
            "product_category": (top_products[0].get("product_category") if top_products else "other"),
            "issue_type": (top_issues[0].get("issue_type") if top_issues else "other"),
        },
        "document": {"status": "not_applicable"},
        "rubric": {
            "classification_present": True,
            "risk_present": True,
            "root_cause_present": True,
            "resolution_present": True,
            "document_grounded": True,
            "contradiction_handled": True,
            "monetary_amount_grounded": True,
            "confidence_calibrated": True,
        },
        "notes": "Weak gold derived from CFPB row and company taxonomy ranking; use for baseline benchmarking, not final adjudication.",
        "adjudication_confidence": 0.55,
    }


def populate_cfpb_source_dataset(
    *,
    csv_path: Path = DEFAULT_CSV,
    sample_size: int = 500,
    seed: int = 42,
    dataset_name: str = "mock_bank_cfpb_source_sample",
    company_id: str = "mock_bank",
    replace: bool = False,
) -> dict[str, Any]:
    """Create or refresh a stratified CFPB-backed source dataset in Postgres."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CFPB CSV not found: {csv_path}")

    counts = _collect_stratum_counts(csv_path)
    targets = _allocate_stratified_targets(counts, sample_size)
    rows = _reservoir_sample_cfpb_rows(csv_path, targets, seed=seed)

    session = SessionLocal()
    try:
        dataset = session.query(SourceDataset).filter(SourceDataset.name == dataset_name).first()
        if dataset is not None and not replace and session.query(SourceDatasetItem).filter(SourceDatasetItem.dataset_id == dataset.id).count() > 0:
            return {
                "dataset_id": dataset.id,
                "dataset_name": dataset.name,
                "rows_sampled": session.query(SourceDatasetItem).filter(SourceDatasetItem.dataset_id == dataset.id).count(),
                "status": "existing",
            }

        if dataset is None:
            dataset = SourceDataset(
                id=uuid.uuid4().hex,
                name=dataset_name,
                company_id=company_id,
                source_type="cfpb_stratified_sample",
                description="Stratified sample of CFPB complaints with cleaned non-null narratives, used as mock-bank source corpus.",
                version="v1",
                status="active",
            )
            session.add(dataset)
            session.flush()
        elif replace:
            session.query(SourceDatasetItem).filter(SourceDatasetItem.dataset_id == dataset.id).delete()

        for row in rows:
            session.add(SourceDatasetItem(
                id=uuid.uuid4().hex,
                dataset_id=dataset.id,
                external_id=str(row.get(COL_COMPLAINT_ID) or ""),
                split="evaluation",
                consumer_narrative=str(row.get(COL_NARRATIVE) or ""),
                product=row.get(COL_PRODUCT),
                sub_product=row.get(COL_SUB_PRODUCT),
                issue=row.get(COL_ISSUE),
                sub_issue=row.get(COL_SUB_ISSUE),
                company=row.get(COL_COMPANY),
                state=row.get(COL_STATE),
                submitted_via=row.get(COL_SUBMITTED_VIA),
                date_received=row.get(COL_DATE_RECEIVED),
                company_response=row.get(COL_RESPONSE),
                company_public_response=row.get(COL_COMPANY_PUBLIC_RESPONSE),
                metadata_json=_json_dumps({
                    "source": "cfpb",
                    "complaint_id": row.get(COL_COMPLAINT_ID),
                    "sampling_stratum": [row.get(COL_PRODUCT), row.get(COL_ISSUE)],
                }),
            ))

        dataset.sampling_strategy_json = _json_dumps({
            "type": "stratified_reservoir",
            "strata": ["product", "issue"],
            "sample_size": sample_size,
            "seed": seed,
            "min_narrative_length": MIN_NARRATIVE_LENGTH,
            "csv_path": str(csv_path),
        })
        dataset.stats_json = _json_dumps({
            "rows_sampled": len(rows),
            "strata_count": len(counts),
            "non_null_narrative_rows": sum(counts.values()),
        })
        session.commit()
        return {
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
            "rows_sampled": len(rows),
            "status": "created",
        }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_eval_dataset_from_source_dataset(
    source_dataset_id: str,
    *,
    dataset_name: str | None = None,
    replace: bool = False,
) -> dict[str, Any]:
    """Materialize an evaluation dataset from a source dataset using weak gold labels."""
    session = SessionLocal()
    try:
        source_dataset = session.get(SourceDataset, source_dataset_id)
        if source_dataset is None:
            raise ValueError(f"Unknown source dataset: {source_dataset_id}")

        eval_name = dataset_name or f"{source_dataset.name}_eval"
        dataset = session.query(EvaluationDataset).filter(EvaluationDataset.name == eval_name).first()
        if dataset is not None and not replace and session.query(EvaluationCase).filter(EvaluationCase.dataset_id == dataset.id).count() > 0:
            return {
                "dataset_id": dataset.id,
                "dataset_name": dataset.name,
                "case_count": session.query(EvaluationCase).filter(EvaluationCase.dataset_id == dataset.id).count(),
                "status": "existing",
            }

        if dataset is None:
            dataset = EvaluationDataset(
                id=uuid.uuid4().hex,
                source_dataset_id=source_dataset.id,
                name=eval_name,
                description=f"Weak-gold evaluation dataset generated from source dataset {source_dataset.name}.",
                source="cfpb_sample",
                version="v1",
                is_gold=False,
                status="active",
            )
            session.add(dataset)
            session.flush()
        elif replace:
            case_ids = [row[0] for row in session.query(EvaluationCase.id).filter(EvaluationCase.dataset_id == dataset.id).all()]
            if case_ids:
                session.query(EvaluationGoldLabel).filter(EvaluationGoldLabel.eval_case_id.in_(case_ids)).delete(synchronize_session=False)
            session.query(EvaluationCase).filter(EvaluationCase.dataset_id == dataset.id).delete()

        source_rows = (
            session.query(SourceDatasetItem)
            .filter(SourceDatasetItem.dataset_id == source_dataset.id)
            .order_by(SourceDatasetItem.created_at.asc())
            .all()
        )
        for item in source_rows:
            metadata = _json_loads(item.metadata_json) or {}
            input_payload = {
                "consumer_narrative": item.consumer_narrative,
                "product": item.product,
                "sub_product": item.sub_product,
                "company": item.company,
                "state": item.state,
                "cfpb_product": item.product,
                "cfpb_sub_product": item.sub_product,
                "cfpb_issue": item.issue,
                "cfpb_sub_issue": item.sub_issue,
            }
            eval_case = EvaluationCase(
                id=uuid.uuid4().hex,
                dataset_id=dataset.id,
                external_case_id=item.external_id,
                title=f"{item.product or 'unknown'} · {item.issue or 'unknown'} · #{item.external_id or item.id[:8]}",
                source="cfpb_sample",
                narrative=item.consumer_narrative,
                input_payload_json=_json_dumps(input_payload) or "{}",
                documents_json=_json_dumps([]),
                tags_json=_json_dumps([
                    "cfpb_sample",
                    item.product or "unknown_product",
                    item.issue or "unknown_issue",
                    item.state or "unknown_state",
                ]),
            )
            session.add(eval_case)
            session.flush()

            gold = _weak_gold_from_source_row({
                COL_PRODUCT: item.product,
                COL_SUB_PRODUCT: item.sub_product,
                COL_ISSUE: item.issue,
                COL_SUB_ISSUE: item.sub_issue,
                COL_NARRATIVE: item.consumer_narrative,
            })
            session.add(EvaluationGoldLabel(
                id=uuid.uuid4().hex,
                eval_case_id=eval_case.id,
                expected_classification_json=_json_dumps(gold.get("classification")),
                expected_risk_json=_json_dumps(gold.get("risk")),
                expected_root_cause_json=_json_dumps(gold.get("root_cause")),
                expected_resolution_json=_json_dumps(gold.get("resolution")),
                expected_document_json=_json_dumps(gold.get("document")),
                rubric_json=_json_dumps(gold.get("rubric")),
                adjudication_notes=gold.get("notes"),
                adjudication_confidence=gold.get("adjudication_confidence"),
            ))

        session.commit()
        return {
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
            "case_count": len(source_rows),
            "status": "created",
        }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def seed_default_eval_dataset(
    *,
    csv_path: Path = DEFAULT_CSV,
    sample_size: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Seed evaluation from a stratified CFPB sample instead of synthetic cases."""
    source = populate_cfpb_source_dataset(
        csv_path=csv_path,
        sample_size=sample_size,
        seed=seed,
        dataset_name="mock_bank_cfpb_source_sample",
        replace=False,
    )
    evaluation = create_eval_dataset_from_source_dataset(
        source["dataset_id"],
        dataset_name="mock_bank_cfpb_eval_sample",
        replace=False,
    )
    return {"source_dataset": source, "evaluation_dataset": evaluation}


def _expected_match(actual: dict[str, Any] | None, expected: dict[str, Any] | None) -> tuple[bool, list[str]]:
    if not expected:
        return True, []
    actual = actual or {}
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if key == "monetary_amount" and expected_value is not None:
            if actual_value is None or abs(float(actual_value) - float(expected_value)) > 1.0:
                mismatches.append(key)
            continue
        if actual_value != expected_value:
            mismatches.append(key)
    return not mismatches, mismatches


def _build_system_output(final_state: dict[str, Any]) -> dict[str, Any]:
    case = final_state.get("case")
    case_dump = _normalize(case.model_dump()) if hasattr(case, "model_dump") else _normalize(case or {})
    classification = final_state.get("classification")
    risk = final_state.get("risk_assessment")
    resolution = final_state.get("resolution")
    root_cause = final_state.get("root_cause_hypothesis")

    return {
        "case": case_dump,
        "classification": _normalize(classification),
        "risk_assessment": _normalize(risk),
        "resolution": _normalize(resolution),
        "root_cause_hypothesis": _normalize(root_cause),
        "review": _normalize(final_state.get("review")),
        "routed_to": final_state.get("routed_to"),
        "document_gate_result": case_dump.get("document_gate_result") if isinstance(case_dump, dict) else None,
        "document_consistency": case_dump.get("document_consistency") if isinstance(case_dump, dict) else None,
    }


def _pick_fields(payload: dict[str, Any] | None, keys: list[str]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    cleaned = {key: payload.get(key) for key in keys if payload.get(key) is not None}
    return cleaned or None


def _build_normalized_prediction(system_output: dict[str, Any]) -> dict[str, Any]:
    """Return the cleaned judge-facing prediction record for one eval run."""
    classification = _pick_fields(
        system_output.get("classification"),
        [
            "product_category",
            "issue_type",
            "sub_product",
            "sub_issue",
            "confidence",
            "reasoning",
            "keywords",
            "review_recommended",
            "reason_codes",
            "alternate_candidates",
        ],
    )
    risk = _pick_fields(
        system_output.get("risk_assessment"),
        [
            "risk_level",
            "risk_score",
            "factors",
            "regulatory_risk",
            "financial_impact_estimate",
            "escalation_required",
            "reasoning",
        ],
    )
    root_cause = _pick_fields(
        system_output.get("root_cause_hypothesis"),
        [
            "root_cause_category",
            "confidence",
            "reasoning",
            "controls_to_check",
            "notes",
        ],
    )
    resolution = _pick_fields(
        system_output.get("resolution"),
        [
            "recommended_action",
            "description",
            "estimated_resolution_days",
            "monetary_amount",
            "confidence",
            "reasoning",
            "similar_case_ids",
        ],
    )
    document = _pick_fields(
        system_output.get("document_consistency"),
        [
            "status",
            "summary",
            "conflicts",
            "verified_facts",
            "grounding_summary",
        ],
    )
    review = _pick_fields(system_output.get("review"), ["decision", "notes"])
    document_gate = _pick_fields(system_output.get("document_gate_result"), ["status", "timed_out", "documents_present"])

    notes_parts: list[str] = [
        "Normalized judge-facing prediction derived from the workflow output.",
    ]
    if system_output.get("routed_to"):
        notes_parts.append(f"Supervisor route: {system_output['routed_to']}.")
    if review and review.get("decision"):
        notes_parts.append(f"Review decision: {review['decision']}.")

    metadata: dict[str, Any] = {}
    if review:
        metadata["review"] = review
    if document_gate:
        metadata["document_gate"] = document_gate
    if system_output.get("routed_to"):
        metadata["routed_to"] = system_output.get("routed_to")

    confidence = None
    if classification and classification.get("confidence") is not None:
        confidence = classification.get("confidence")
    elif resolution and resolution.get("confidence") is not None:
        confidence = resolution.get("confidence")
    elif root_cause and root_cause.get("confidence") is not None:
        confidence = root_cause.get("confidence")

    return {
        "classification": classification,
        "risk": risk,
        "root_cause": root_cause,
        "resolution": resolution,
        "document": document,
        "confidence": confidence,
        "notes": " ".join(notes_parts),
        "metadata": metadata or None,
    }


def _collect_case_documents_for_evaluation(case_id: str) -> list[dict[str, Any]]:
    session = SessionLocal()
    try:
        rows = (
            session.query(CaseDocument, DocumentArtifact)
            .outerjoin(DocumentArtifact, DocumentArtifact.document_id == CaseDocument.id)
            .filter(CaseDocument.case_id == case_id)
            .order_by(CaseDocument.created_at.asc())
            .all()
        )
        documents: list[dict[str, Any]] = []
        for document, artifact in rows:
            extracted = _json_loads(artifact.extracted_json) if artifact else None
            documents.append({
                "id": document.id,
                "name": document.original_filename,
                "mime_type": document.mime_type,
                "document_type": document.document_type,
                "upload_status": document.upload_status,
                "parser_status": document.parser_status,
                "extraction_status": document.extraction_status,
                "raw_text": getattr(artifact, "raw_text", None) if artifact else None,
                "normalized_text": getattr(artifact, "normalized_text", None) if artifact else None,
                "content": getattr(artifact, "normalized_text", None) if artifact else None,
                "extracted": extracted if isinstance(extracted, dict) else None,
            })
        return documents
    finally:
        session.close()


def _load_case_for_evaluation(case_id: str) -> dict[str, Any] | None:
    session = SessionLocal()
    try:
        case = session.get(ComplaintCase, case_id)
        if case is None:
            return None
        classification = session.query(ClassificationRecord).filter(ClassificationRecord.case_id == case_id).first()
        risk = session.query(RiskRecord).filter(RiskRecord.case_id == case_id).first()
        resolution = session.query(ResolutionRecord).filter(ResolutionRecord.case_id == case_id).first()
        return {
            "id": case.id,
            "public_case_id": case.public_case_id,
            "consumer_narrative": case.consumer_narrative or "",
            "review_notes": case.review_notes,
            "routed_to": case.routed_to,
            "document_gate_result": _json_loads(case.document_gate_result_json) or {},
            "document_consistency": _json_loads(case.document_consistency_json) or {},
            "root_cause_hypothesis": _json_loads(case.root_cause_hypothesis_json) or {},
            "classification": {
                "product_category": classification.product_category,
                "issue_type": classification.issue_type,
                "sub_issue": classification.sub_issue,
                "confidence": classification.confidence,
                "reasoning": classification.reasoning,
                "review_recommended": bool(classification.review_recommended),
                "reason_codes": _json_loads(classification.reason_codes_json) or [],
                "keywords": _json_loads(classification.keywords_json) or [],
            } if classification else {},
            "risk_assessment": {
                "risk_level": risk.risk_level,
                "risk_score": risk.risk_score,
                "regulatory_risk": risk.regulatory_risk,
                "financial_impact_estimate": risk.financial_impact_estimate,
                "escalation_required": risk.escalation_required,
                "reasoning": risk.reasoning,
            } if risk else {},
            "resolution": {
                "recommended_action": resolution.recommended_action,
                "description": resolution.description,
                "estimated_resolution_days": resolution.estimated_resolution_days,
                "monetary_amount": resolution.monetary_amount,
                "confidence": resolution.confidence,
                "reasoning": resolution.reasoning,
            } if resolution else {},
            "documents": _collect_case_documents_for_evaluation(case_id),
        }
    finally:
        session.close()


def _build_production_system_assessment(
    system_output: dict[str, Any],
    *,
    documents_present: bool,
) -> dict[str, Any]:
    classification = system_output.get("classification") or {}
    risk = system_output.get("risk") or {}
    root_cause = system_output.get("root_cause") or {}
    resolution = system_output.get("resolution") or {}
    document = system_output.get("document") or {}
    metadata = system_output.get("metadata") or {}
    review = metadata.get("review") or {}

    checks = {
        "classification_complete": bool(classification.get("product_category") and classification.get("issue_type")),
        "risk_complete": bool(risk.get("risk_level") is not None and risk.get("risk_score") is not None),
        "root_cause_complete": bool(root_cause.get("root_cause_category") and root_cause.get("reasoning")),
        "resolution_complete": bool(resolution.get("recommended_action") and resolution.get("description")),
        "document_considered": (not documents_present) or bool(document),
        "review_signal_present": bool(classification.get("review_recommended")) or bool(risk.get("escalation_required")) or bool(review.get("decision")),
    }
    failed_dimensions = [key for key, passed in checks.items() if not passed]
    overall = "pass" if not failed_dimensions else "needs_review"
    if len(failed_dimensions) >= 3:
        overall = "fail"

    reasoning_parts: list[str] = []
    if checks["classification_complete"]:
        reasoning_parts.append("Classification output is present with product and issue labels.")
    else:
        reasoning_parts.append("Classification output is incomplete.")
    if checks["risk_complete"]:
        reasoning_parts.append("Risk assessment includes a level and score.")
    else:
        reasoning_parts.append("Risk assessment is missing key fields.")
    if checks["root_cause_complete"]:
        reasoning_parts.append("Root cause analysis includes a category and supporting reasoning.")
    else:
        reasoning_parts.append("Root cause analysis is incomplete or missing reasoning.")
    if checks["resolution_complete"]:
        reasoning_parts.append("Resolution output includes an action and business description.")
    else:
        reasoning_parts.append("Resolution output is incomplete.")
    if documents_present:
        if checks["document_considered"]:
            reasoning_parts.append("Attached documents were incorporated into the evaluation context.")
        else:
            reasoning_parts.append("Documents were attached but the analysis does not show clear document grounding.")

    return {
        "overall_verdict": overall,
        "checks": checks,
        "failed_dimensions": failed_dimensions,
        "reasoning": " ".join(reasoning_parts),
    }


def evaluate_production_case(case_id: str) -> dict[str, Any] | None:
    """Create or refresh the production evaluation report for a real complaint case."""
    payload = _load_case_for_evaluation(case_id)
    if payload is None:
        return None

    raw_system_output = {
        "classification": payload.get("classification") or {},
        "risk_assessment": payload.get("risk_assessment") or {},
        "resolution": payload.get("resolution") or {},
        "root_cause_hypothesis": payload.get("root_cause_hypothesis") or {},
        "review": {"decision": "review" if payload.get("classification", {}).get("review_recommended") else "approve"},
        "routed_to": payload.get("routed_to"),
        "document_gate_result": payload.get("document_gate_result") or {},
        "document_consistency": payload.get("document_consistency") or {},
    }
    normalized_prediction = _build_normalized_prediction(raw_system_output)
    judge_output = run_rubric_judge(
        case_input={
            "narrative": payload.get("consumer_narrative") or "",
            "documents": payload.get("documents") or [],
        },
        system_output=normalized_prediction,
    )
    system_assessment = _build_production_system_assessment(
        normalized_prediction,
        documents_present=bool(payload.get("documents")),
    )

    judge_verdict = judge_output.get("overall_verdict") or "needs_review"
    system_verdict = system_assessment.get("overall_verdict") or "needs_review"
    overall_status = "pass" if judge_verdict == system_verdict == "pass" else "needs_review"
    if judge_verdict == "fail" or system_verdict == "fail":
        overall_status = "fail"

    disagreement_types: list[str] = []
    if judge_verdict != system_verdict:
        disagreement_types.append("system_and_judge_verdict_mismatch")
    if (judge_output.get("summary") or {}).get("failed_dimensions"):
        disagreement_types.extend(f"judge_{item}" for item in judge_output["summary"]["failed_dimensions"])
    if system_assessment.get("failed_dimensions"):
        disagreement_types.extend(f"system_{item}" for item in system_assessment["failed_dimensions"])

    session = SessionLocal()
    try:
        report = (
            session.query(ComplaintEvaluationReport)
            .filter(ComplaintEvaluationReport.case_id == case_id)
            .first()
        )
        if report is None:
            report = ComplaintEvaluationReport(id=uuid.uuid4().hex, case_id=case_id)
            session.add(report)
        report.run_status = "completed"
        report.system_prediction_json = _json_dumps(normalized_prediction)
        report.system_assessment_json = _json_dumps(system_assessment)
        report.judge_output_json = _json_dumps(judge_output)
        report.overall_status = overall_status
        report.needs_human_review = overall_status != "pass"
        report.disagreement_types_json = _json_dumps(sorted(set(disagreement_types)))
        report.metrics_json = _json_dumps({
            "judge_verdict": judge_verdict,
            "system_verdict": system_verdict,
            "documents_present": bool(payload.get("documents")),
        })
        report.judge_reasoning = judge_output.get("reasoning")
        report.system_reasoning = system_assessment.get("reasoning")
        report.evaluated_at = datetime.utcnow()
        session.commit()
        return {
            "case_id": case_id,
            "overall_status": overall_status,
            "judge_verdict": judge_verdict,
            "system_verdict": system_verdict,
        }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _system_vs_gold(system_output: dict[str, Any], gold: EvaluationGoldLabel | None) -> tuple[dict[str, Any], list[str]]:
    if gold is None:
        return {"status": "unlabeled"}, []

    expected_classification = _json_loads(gold.expected_classification_json) or {}
    expected_risk = _json_loads(gold.expected_risk_json) or {}
    expected_root_cause = _json_loads(gold.expected_root_cause_json) or {}
    expected_resolution = _json_loads(gold.expected_resolution_json) or {}
    expected_document = _json_loads(gold.expected_document_json) or {}

    class_ok, class_mismatches = _expected_match(system_output.get("classification"), expected_classification)
    risk_ok, risk_mismatches = _expected_match(system_output.get("risk"), expected_risk)
    root_ok, root_mismatches = _expected_match(system_output.get("root_cause"), expected_root_cause)
    resolution_ok, resolution_mismatches = _expected_match(system_output.get("resolution"), expected_resolution)
    document_ok, document_mismatches = _expected_match(system_output.get("document"), expected_document)

    disagreements: list[str] = []
    if not class_ok:
        disagreements.append("classification_mismatch")
    if not risk_ok:
        disagreements.append("risk_mismatch")
    if not root_ok:
        disagreements.append("root_cause_mismatch")
    if not resolution_ok:
        disagreements.append("resolution_mismatch")
    if not document_ok:
        disagreements.append("document_consistency_mismatch")

    checks = {
        "classification": {"pass": class_ok, "mismatches": class_mismatches},
        "risk": {"pass": risk_ok, "mismatches": risk_mismatches},
        "root_cause": {"pass": root_ok, "mismatches": root_mismatches},
        "resolution": {"pass": resolution_ok, "mismatches": resolution_mismatches},
        "document_consistency": {"pass": document_ok, "mismatches": document_mismatches},
    }
    pass_count = sum(1 for payload in checks.values() if payload["pass"])
    status = "pass" if pass_count == len(checks) else ("needs_review" if pass_count >= 3 else "fail")
    return {
        "status": status,
        "pass_count": pass_count,
        "total_checks": len(checks),
        "checks": checks,
    }, disagreements


def _judge_vs_gold(judge_output: dict[str, Any], gold: EvaluationGoldLabel | None) -> tuple[dict[str, Any], list[str]]:
    expected_rubric = (_json_loads(gold.rubric_json) if gold is not None else None) or {}
    actual_rubric = judge_output.get("rubric") or {}
    disagreements: list[str] = []
    checks: dict[str, Any] = {}
    for key, expected in expected_rubric.items():
        actual = actual_rubric.get(key)
        passed = actual == expected
        checks[key] = {"pass": passed, "expected": expected, "actual": actual}
        if not passed:
            disagreements.append(f"judge_{key}")
    pass_count = sum(1 for payload in checks.values() if payload["pass"])
    total_checks = len(checks)
    status = "pass" if pass_count == total_checks else ("needs_review" if pass_count >= max(1, total_checks - 2) else "fail")
    return {
        "status": status,
        "pass_count": pass_count,
        "total_checks": total_checks,
        "checks": checks,
    }, disagreements


def _system_vs_judge(system_output: dict[str, Any], judge_output: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    classification = system_output.get("classification") or {}
    metadata = system_output.get("metadata") or {}
    review = metadata.get("review") or {}
    document_consistency = system_output.get("document") or {}
    failed_dimensions = ((judge_output.get("summary") or {}).get("failed_dimensions")) or []

    system_requested_review = bool(classification.get("review_recommended")) or str(review.get("decision") or "") != "approve"
    judge_requests_review = judge_output.get("overall_verdict") != "pass"
    contradiction_alignment = (
        ("contradiction_handled" not in failed_dimensions)
        == (str(document_consistency.get("status") or "") == "contradiction" or bool(classification.get("review_recommended")))
    )

    checks = {
        "review_signal_alignment": {
            "pass": system_requested_review == judge_requests_review,
            "system_requested_review": system_requested_review,
            "judge_requests_review": judge_requests_review,
        },
        "contradiction_alignment": {
            "pass": contradiction_alignment,
            "system_document_status": document_consistency.get("status"),
            "judge_failed_dimensions": failed_dimensions,
        },
    }
    disagreements = [key for key, payload in checks.items() if not payload["pass"]]
    status = "pass" if not disagreements else "needs_review"
    return {"status": status, "checks": checks}, disagreements


def _system_versions() -> dict[str, str]:
    return {
        "workflow_version": workflow_version(),
        "prompt_bundle_version": prompt_bundle_version(),
        "knowledge_pack_version": knowledge_pack_version(),
        "model_version": default_chat_model(),
        "llm_provider": llm_provider(),
    }


def run_dataset_benchmark(dataset_id: str, *, limit: int | None = None) -> dict[str, Any]:
    """Run the production workflow against a DB-backed benchmark dataset."""
    session = SessionLocal()
    totals = {"runs": 0, "passed": 0, "needs_review": 0, "failed": 0, "human_review": 0}
    try:
        dataset = session.get(EvaluationDataset, dataset_id)
        if dataset is None:
            raise ValueError(f"Unknown evaluation dataset: {dataset_id}")

        query = (
            session.query(EvaluationCase)
            .filter(EvaluationCase.dataset_id == dataset_id)
            .order_by(EvaluationCase.created_at.asc())
        )
        if limit is not None:
            query = query.limit(limit)
        cases = query.all()

        for eval_case in cases:
            run_row = EvaluationRun(
                id=uuid.uuid4().hex,
                dataset_id=dataset_id,
                eval_case_id=eval_case.id,
                execution_mode="workflow",
                run_status="running",
                system_version_json=_json_dumps(_system_versions()),
                input_snapshot_json=_json_dumps({
                    "narrative": eval_case.narrative,
                    "payload": _json_loads(eval_case.input_payload_json) or {},
                    "documents": _json_loads(eval_case.documents_json) or [],
                    "tags": _json_loads(eval_case.tags_json) or [],
                }),
                started_at=datetime.utcnow(),
            )
            session.add(run_row)
            session.commit()

            try:
                payload = (_json_loads(eval_case.input_payload_json) or {}).copy()
                final_state = process_complaint(payload)
                raw_system_output = _build_system_output(final_state)
                normalized_prediction = _build_normalized_prediction(raw_system_output)

                session.add(EvaluationSystemPrediction(
                    id=uuid.uuid4().hex,
                    eval_run_id=run_row.id,
                    classification_json=_json_dumps(normalized_prediction.get("classification")),
                    predicted_risk_json=_json_dumps(normalized_prediction.get("risk")),
                    predicted_root_cause_json=_json_dumps(normalized_prediction.get("root_cause")),
                    predicted_resolution_json=_json_dumps(normalized_prediction.get("resolution")),
                    predicted_document_json=_json_dumps(normalized_prediction.get("document")),
                    confidence=normalized_prediction.get("confidence"),
                    notes=normalized_prediction.get("notes"),
                    metadata_json=_json_dumps(normalized_prediction.get("metadata")),
                ))
                session.flush()

                judge_output = run_rubric_judge(
                    case_input={
                        "narrative": eval_case.narrative,
                        "documents": _json_loads(eval_case.documents_json) or [],
                    },
                    system_output=normalized_prediction,
                )

                gold = eval_case.gold_label
                system_vs_gold, system_disagreements = _system_vs_gold(normalized_prediction, gold)
                judge_vs_gold, judge_disagreements = _judge_vs_gold(judge_output, gold)
                system_vs_judge, system_judge_disagreements = _system_vs_judge(normalized_prediction, judge_output)

                all_disagreements = system_disagreements + judge_disagreements + system_judge_disagreements
                overall_status = "pass"
                if system_vs_gold.get("status") == "fail" or judge_vs_gold.get("status") == "fail":
                    overall_status = "fail"
                elif all_disagreements:
                    overall_status = "needs_review"

                needs_human_review = bool(all_disagreements)

                judge_row = EvaluationJudgeRun(
                    id=uuid.uuid4().hex,
                    eval_run_id=run_row.id,
                    judge_name=str(judge_output.get("judge_name") or "rubric_judge"),
                    judge_version=str(judge_output.get("judge_version") or "v1"),
                    run_status="completed",
                    rubric_json=_json_dumps(judge_output.get("rubric") or {}),
                    summary_json=_json_dumps(judge_output.get("summary") or {}),
                    started_at=run_row.started_at,
                    ended_at=datetime.utcnow(),
                )
                session.add(judge_row)
                session.flush()

                review_row = EvaluationReviewRecord(
                    id=uuid.uuid4().hex,
                    eval_run_id=run_row.id,
                    judge_run_id=judge_row.id,
                    overall_status=overall_status,
                    system_vs_gold_json=_json_dumps(system_vs_gold),
                    judge_vs_gold_json=_json_dumps(judge_vs_gold),
                    system_vs_judge_json=_json_dumps(system_vs_judge),
                    disagreement_types_json=_json_dumps(all_disagreements),
                    needs_human_review=needs_human_review,
                )
                session.add(review_row)
                session.flush()

                if needs_human_review:
                    session.add(EvaluationDisagreement(
                        id=uuid.uuid4().hex,
                        review_record_id=review_row.id,
                        status="open",
                        severity="high" if overall_status == "fail" else "medium",
                        reason_codes_json=_json_dumps(all_disagreements),
                        notes=(gold.adjudication_notes if gold is not None else None),
                    ))

                run_row.run_status = "completed"
                run_row.output_snapshot_json = _json_dumps(raw_system_output)
                run_row.metrics_json = _json_dumps({
                    "system_vs_gold_status": system_vs_gold.get("status"),
                    "judge_vs_gold_status": judge_vs_gold.get("status"),
                    "system_vs_judge_status": system_vs_judge.get("status"),
                })
                run_row.ended_at = datetime.utcnow()
                session.commit()

                totals["runs"] += 1
                if overall_status == "pass":
                    totals["passed"] += 1
                elif overall_status == "fail":
                    totals["failed"] += 1
                else:
                    totals["needs_review"] += 1
                if needs_human_review:
                    totals["human_review"] += 1
            except Exception as exc:
                session.rollback()
                run_row = session.get(EvaluationRun, run_row.id)
                if run_row is not None:
                    run_row.run_status = "failed"
                    run_row.error_message = str(exc)
                    run_row.ended_at = datetime.utcnow()
                    session.commit()
                totals["runs"] += 1
                totals["failed"] += 1
                logger.exception("Evaluation run failed for case %s", eval_case.id)

        return totals
    finally:
        session.close()


def build_evaluation_dashboard_data() -> dict[str, Any]:
    """Return admin-facing evaluation summary for the benchmark dashboard."""
    session = SessionLocal()
    try:
        datasets = session.query(EvaluationDataset).order_by(EvaluationDataset.created_at.desc()).all()
        source_datasets = session.query(SourceDataset).order_by(SourceDataset.created_at.desc()).all()
        recent_runs = (
            session.query(EvaluationRun)
            .order_by(EvaluationRun.started_at.desc())
            .limit(12)
            .all()
        )
        open_disagreements = (
            session.query(EvaluationDisagreement)
            .filter(EvaluationDisagreement.status == "open")
            .order_by(EvaluationDisagreement.created_at.desc())
            .limit(12)
            .all()
        )
        all_review_records = session.query(EvaluationReviewRecord).all()
        all_eval_cases = session.query(EvaluationCase).all()

        dataset_cards = []
        for dataset in datasets:
            total_cases = session.query(EvaluationCase).filter(EvaluationCase.dataset_id == dataset.id).count()
            completed_runs = (
                session.query(EvaluationRun)
                .filter(EvaluationRun.dataset_id == dataset.id, EvaluationRun.run_status == "completed")
                .count()
            )
            latest_run = (
                session.query(EvaluationRun)
                .filter(EvaluationRun.dataset_id == dataset.id)
                .order_by(EvaluationRun.started_at.desc())
                .first()
            )
            dataset_cards.append({
                "id": dataset.id,
                "name": dataset.name,
                "description": dataset.description,
                "version": dataset.version,
                "source": dataset.source,
                "is_gold": dataset.is_gold,
                "status": dataset.status,
                "total_cases": total_cases,
                "completed_runs": completed_runs,
                "latest_run_at": latest_run.started_at if latest_run is not None else None,
                "latest_run_status": latest_run.run_status if latest_run is not None else None,
            })

        source_cards = []
        for dataset in source_datasets:
            stats = _json_loads(dataset.stats_json) or {}
            source_cards.append({
                "id": dataset.id,
                "name": dataset.name,
                "source_type": dataset.source_type,
                "company_id": dataset.company_id,
                "rows_sampled": int(stats.get("rows_sampled") or 0),
                "strata_count": int(stats.get("strata_count") or 0),
                "non_null_narrative_rows": int(stats.get("non_null_narrative_rows") or 0),
                "status": dataset.status,
            })

        run_rows = []
        for run in recent_runs:
            review = run.review_record
            judge = run.judge_runs[0] if run.judge_runs else None
            eval_case = run.eval_case
            run_rows.append({
                "id": run.id,
                "eval_case_id": eval_case.id if eval_case else run.eval_case_id,
                "dataset_name": run.dataset.name if run.dataset else "Unknown",
                "case_title": eval_case.title if eval_case else run.eval_case_id,
                "run_status": run.run_status,
                "review_status": review.overall_status if review else "pending",
                "needs_human_review": bool(review.needs_human_review) if review else False,
                "judge_name": judge.judge_name if judge else None,
                "started_at": run.started_at,
            })

        disagreement_rows = []
        for row in open_disagreements:
            review = row.review_record
            eval_run = review.eval_run if review else None
            eval_case = eval_run.eval_case if eval_run else None
            disagreement_rows.append({
                "id": row.id,
                "eval_case_id": eval_case.id if eval_case else (eval_run.eval_case_id if eval_run else None),
                "case_title": eval_case.title if eval_case else (eval_run.eval_case_id if eval_run else "Unknown"),
                "dataset_name": eval_run.dataset.name if eval_run and eval_run.dataset else "Unknown",
                "severity": row.severity,
                "status": row.status,
                "reason_codes": _json_loads(row.reason_codes_json) or [],
                "created_at": row.created_at,
            })

        pass_count = sum(1 for row in run_rows if row["review_status"] == "pass")
        needs_review_count = sum(1 for row in run_rows if row["review_status"] == "needs_review")
        fail_count = sum(1 for row in run_rows if row["review_status"] == "fail")

        disagreement_reason_counts: Counter[str] = Counter()
        for row in open_disagreements:
            disagreement_reason_counts.update(_json_loads(row.reason_codes_json) or [])

        system_dimension_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "total": 0})
        judge_dimension_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "total": 0})
        for record in all_review_records:
            system_vs_gold = _json_loads(record.system_vs_gold_json) or {}
            for dimension, payload in (system_vs_gold.get("checks") or {}).items():
                system_dimension_counts[dimension]["total"] += 1
                if payload.get("pass"):
                    system_dimension_counts[dimension]["pass"] += 1

            judge_vs_gold = _json_loads(record.judge_vs_gold_json) or {}
            for dimension, payload in (judge_vs_gold.get("checks") or {}).items():
                judge_dimension_counts[dimension]["total"] += 1
                if payload.get("pass"):
                    judge_dimension_counts[dimension]["pass"] += 1

        def _dimension_rows(counts: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for dimension, payload in sorted(counts.items()):
                total = payload["total"]
                passed = payload["pass"]
                rows.append({
                    "dimension": dimension,
                    "pass_count": passed,
                    "total": total,
                    "accuracy": round((passed / total) * 100, 1) if total else 0.0,
                })
            return rows

        product_counts: Counter[str] = Counter()
        issue_counts: Counter[str] = Counter()
        narrative_length_bands: Counter[str] = Counter()
        for eval_case in all_eval_cases:
            payload = _json_loads(eval_case.input_payload_json) or {}
            product_counts.update([str(payload.get("cfpb_product") or payload.get("product") or "unknown")])
            issue_counts.update([str(payload.get("cfpb_issue") or "unknown")])
            n_len = len(eval_case.narrative or "")
            if n_len < 80:
                narrative_length_bands.update(["short"])
            elif n_len < 200:
                narrative_length_bands.update(["medium"])
            else:
                narrative_length_bands.update(["long"])

        return {
            "terms": EVALUATION_TERM_DESCRIPTIONS,
            "source_datasets": source_cards,
            "datasets": dataset_cards,
            "recent_runs": run_rows,
            "open_disagreements": disagreement_rows,
            "coverage": {
                "products": dict(product_counts.most_common(8)),
                "issues": dict(issue_counts.most_common(8)),
                "narrative_lengths": dict(narrative_length_bands),
            },
            "dimension_scores": {
                "system_vs_gold": _dimension_rows(system_dimension_counts),
                "judge_vs_gold": _dimension_rows(judge_dimension_counts),
            },
            "disagreement_reason_counts": dict(disagreement_reason_counts.most_common(8)),
            "summary": {
                "source_dataset_count": len(source_cards),
                "dataset_count": len(dataset_cards),
                "recent_run_count": len(run_rows),
                "open_disagreement_count": len(disagreement_rows),
                "pass_count": pass_count,
                "needs_review_count": needs_review_count,
                "fail_count": fail_count,
            },
        }
    finally:
        session.close()


def build_evaluation_case_detail(eval_case_id: str) -> dict[str, Any] | None:
    """Return a case-centric evaluation detail payload for the UI."""
    session = SessionLocal()
    try:
        eval_case = session.get(EvaluationCase, eval_case_id)
        if eval_case is None:
            return None

        dataset = eval_case.dataset
        source_dataset = dataset.source_dataset if dataset is not None else None
        gold = eval_case.gold_label
        runs = (
            session.query(EvaluationRun)
            .filter(EvaluationRun.eval_case_id == eval_case_id)
            .order_by(EvaluationRun.started_at.desc())
            .all()
        )
        latest_run = runs[0] if runs else None
        latest_prediction = latest_run.system_prediction if latest_run else None
        latest_judge = latest_run.judge_runs[0] if latest_run and latest_run.judge_runs else None
        latest_review = latest_run.review_record if latest_run else None
        disagreement = latest_review.disagreement if latest_review else None
        raw_snapshot = _json_loads(latest_run.output_snapshot_json) if latest_run else {}
        normalized_fallback = _build_normalized_prediction(raw_snapshot or {}) if latest_run and latest_prediction is None else None

        return {
            "id": eval_case.id,
            "title": eval_case.title,
            "narrative": eval_case.narrative,
            "source": eval_case.source,
            "external_case_id": eval_case.external_case_id,
            "dataset": {
                "id": dataset.id if dataset else None,
                "name": dataset.name if dataset else "Unknown",
                "version": dataset.version if dataset else None,
                "source": dataset.source if dataset else None,
            },
            "source_dataset": {
                "id": source_dataset.id if source_dataset else None,
                "name": source_dataset.name if source_dataset else None,
                "source_type": source_dataset.source_type if source_dataset else None,
            } if source_dataset else None,
            "input_payload": _json_loads(eval_case.input_payload_json) or {},
            "documents": _json_loads(eval_case.documents_json) or [],
            "tags": _json_loads(eval_case.tags_json) or [],
            "weak_gold": {
                "classification": _json_loads(gold.expected_classification_json) if gold else None,
                "risk": _json_loads(gold.expected_risk_json) if gold else None,
                "root_cause": _json_loads(gold.expected_root_cause_json) if gold else None,
                "resolution": _json_loads(gold.expected_resolution_json) if gold else None,
                "document": _json_loads(gold.expected_document_json) if gold else None,
                "rubric": _json_loads(gold.rubric_json) if gold else None,
                "notes": gold.adjudication_notes if gold else None,
                "confidence": gold.adjudication_confidence if gold else None,
            },
            "latest_run": {
                "id": latest_run.id,
                "status": latest_run.run_status,
                "started_at": latest_run.started_at,
                "ended_at": latest_run.ended_at,
                "system_versions": _json_loads(latest_run.system_version_json) or {},
                "system_output": {
                    "classification": _json_loads(latest_prediction.classification_json) if latest_prediction else normalized_fallback.get("classification"),
                    "risk": _json_loads(latest_prediction.predicted_risk_json) if latest_prediction else normalized_fallback.get("risk"),
                    "root_cause": _json_loads(latest_prediction.predicted_root_cause_json) if latest_prediction else normalized_fallback.get("root_cause"),
                    "resolution": _json_loads(latest_prediction.predicted_resolution_json) if latest_prediction else normalized_fallback.get("resolution"),
                    "document": _json_loads(latest_prediction.predicted_document_json) if latest_prediction else normalized_fallback.get("document"),
                    "confidence": latest_prediction.confidence if latest_prediction else normalized_fallback.get("confidence"),
                    "notes": latest_prediction.notes if latest_prediction else normalized_fallback.get("notes"),
                    "metadata": _json_loads(latest_prediction.metadata_json) if latest_prediction else normalized_fallback.get("metadata"),
                } if latest_run else {},
                "raw_snapshot": raw_snapshot or {},
                "metrics": _json_loads(latest_run.metrics_json) or {},
                "error_message": latest_run.error_message,
            } if latest_run else None,
            "latest_judge": {
                "id": latest_judge.id,
                "judge_name": latest_judge.judge_name,
                "judge_version": latest_judge.judge_version,
                "rubric": _json_loads(latest_judge.rubric_json) or {},
                "summary": _json_loads(latest_judge.summary_json) or {},
            } if latest_judge else None,
            "latest_review": {
                "id": latest_review.id,
                "overall_status": latest_review.overall_status,
                "system_vs_gold": _json_loads(latest_review.system_vs_gold_json) or {},
                "judge_vs_gold": _json_loads(latest_review.judge_vs_gold_json) or {},
                "system_vs_judge": _json_loads(latest_review.system_vs_judge_json) or {},
                "disagreement_types": _json_loads(latest_review.disagreement_types_json) or [],
                "needs_human_review": latest_review.needs_human_review,
            } if latest_review else None,
            "disagreement": {
                "id": disagreement.id,
                "status": disagreement.status,
                "severity": disagreement.severity,
                "reason_codes": _json_loads(disagreement.reason_codes_json) or [],
                "notes": disagreement.notes,
            } if disagreement else None,
            "run_history": [
                {
                    "id": row.id,
                    "status": row.run_status,
                    "started_at": row.started_at,
                    "review_status": row.review_record.overall_status if row.review_record else "pending",
                }
                for row in runs[:10]
            ],
            "terms": EVALUATION_TERM_DESCRIPTIONS,
        }
    finally:
        session.close()


def build_production_evaluation_dashboard_data() -> dict[str, Any]:
    """Return admin-facing production evaluation summary for real complaint cases."""
    session = SessionLocal()
    try:
        reports = (
            session.query(ComplaintEvaluationReport)
            .order_by(ComplaintEvaluationReport.evaluated_at.desc().nullslast(), ComplaintEvaluationReport.updated_at.desc())
            .all()
        )
        recent_cases = (
            session.query(ComplaintCase)
            .order_by(ComplaintCase.created_at.desc())
            .limit(20)
            .all()
        )

        overall_counts = Counter(report.overall_status or "pending" for report in reports)
        judge_counts = Counter(
            ((_json_loads(report.metrics_json) or {}).get("judge_verdict") or "pending")
            for report in reports
        )
        system_counts = Counter(
            ((_json_loads(report.metrics_json) or {}).get("system_verdict") or "pending")
            for report in reports
        )

        case_rows = []
        for case in recent_cases:
            report = case.evaluation_report
            metrics = _json_loads(report.metrics_json) if report else {}
            case_rows.append({
                "id": case.id,
                "public_case_id": case.public_case_id or case.id[:12].upper(),
                "subject": (case.consumer_narrative or "")[:120],
                "status": case.status,
                "created_at": case.created_at,
                "overall_status": report.overall_status if report else "pending",
                "judge_verdict": (metrics or {}).get("judge_verdict", "pending") if report else "pending",
                "system_verdict": (metrics or {}).get("system_verdict", "pending") if report else "pending",
                "evaluated_at": report.evaluated_at if report else None,
                "needs_human_review": bool(report.needs_human_review) if report else False,
            })

        return {
            "summary": {
                "evaluated_count": len(reports),
                "pass_count": overall_counts.get("pass", 0),
                "needs_review_count": overall_counts.get("needs_review", 0),
                "fail_count": overall_counts.get("fail", 0),
                "pending_count": max(0, session.query(ComplaintCase).count() - len(reports)),
            },
            "verdict_breakdown": {
                "overall": dict(overall_counts),
                "judge": dict(judge_counts),
                "system": dict(system_counts),
            },
            "recent_case_reports": case_rows,
        }
    finally:
        session.close()


def build_production_evaluation_case_detail(case_id: str) -> dict[str, Any] | None:
    """Return the stored production evaluation report for one complaint case."""
    session = SessionLocal()
    try:
        case = session.get(ComplaintCase, case_id)
        if case is None:
            return None
        report = case.evaluation_report
        return {
            "case": {
                "id": case.id,
                "public_case_id": case.public_case_id or case.id[:12].upper(),
                "status": case.status,
                "consumer_narrative": case.consumer_narrative or "",
                "created_at": case.created_at,
                "routed_to": case.routed_to,
                "review_notes": case.review_notes,
            },
            "report": {
                "overall_status": report.overall_status if report else "pending",
                "system_prediction": _json_loads(report.system_prediction_json) if report else None,
                "system_assessment": _json_loads(report.system_assessment_json) if report else None,
                "judge_output": _json_loads(report.judge_output_json) if report else None,
                "judge_reasoning": report.judge_reasoning if report else None,
                "system_reasoning": report.system_reasoning if report else None,
                "disagreement_types": _json_loads(report.disagreement_types_json) if report else [],
                "metrics": _json_loads(report.metrics_json) if report else {},
                "evaluated_at": report.evaluated_at if report else None,
                "needs_human_review": bool(report.needs_human_review) if report else False,
            },
            "documents": _collect_case_documents_for_evaluation(case.id),
        }
    finally:
        session.close()
