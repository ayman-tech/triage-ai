"""Evaluation harness for the complaint-classification pipeline.

Loads labelled datasets from ``datasets/`` and measures accuracy,
precision and recall across the key pipeline outputs.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from app.agents.classification import run_classification
from app.db.models import EvaluationDataset
from app.db.session import SessionLocal
from app.evals.service import run_dataset_benchmark, seed_default_eval_dataset
from app.schemas.case import CaseRead
from app.schemas.classification import ClassificationResult

logger = logging.getLogger(__name__)

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_dataset(filename: str) -> list[dict[str, Any]]:
    """Load a CSV or JSON-lines evaluation dataset.

    Expected columns / keys:
        narrative, expected_product_category, expected_issue_type
    """
    path = DATASETS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    rows: list[dict[str, Any]] = []

    if path.suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
    elif path.suffix in (".jsonl", ".json"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    logger.info("Loaded %d rows from %s", len(rows), filename)
    return rows


# ── Evaluation runner ────────────────────────────────────────────────────────

def _row_slice_tags(row: dict[str, Any]) -> list[str]:
    """Heuristic eval slices (see docs/CLASSIFICATION_EVAL_SLICES.md)."""
    tags: list[str] = []
    nar = (row.get("narrative") or "").strip()
    if len(nar) < 10:
        tags.append("narrative_absent")
    else:
        tags.append("long_narrative")
    low = nar.lower()
    if any(
        x in low
        for x in (" and also ", "another issue", "second problem", "in addition")
    ):
        tags.append("multi_issue_heuristic")
    if row.get("cfpb_product") and row.get("cfpb_issue") and len(nar) >= 10:
        tags.append("structured_plus_narrative")
    return tags


def evaluate_classification(
    dataset_file: str = "classification_eval.csv",
    model_name: str | None = None,
) -> dict[str, float]:
    """Run classification evaluation and return metric summary.

    Returns
    -------
    dict with keys: total, correct_product, correct_issue,
                    product_accuracy, issue_accuracy, avg_confidence,
                    slice_counts (JSON-serializable)
    """
    rows = load_dataset(dataset_file)

    total = len(rows)
    correct_product = 0
    correct_issue = 0
    confidence_sum = 0.0
    slice_counts: dict[str, dict[str, int]] = {}

    for row in rows:
        narrative = (row.get("narrative") or "").strip()
        case = CaseRead(
            consumer_narrative=narrative,
            product=row.get("product") or None,
            sub_product=row.get("sub_product") or None,
            cfpb_product=row.get("cfpb_product") or None,
            cfpb_sub_product=row.get("cfpb_sub_product") or None,
            cfpb_issue=row.get("cfpb_issue") or None,
            cfpb_sub_issue=row.get("cfpb_sub_issue") or None,
        )
        pipeline_out = run_classification(case=case, model_name=model_name)
        result: ClassificationResult = pipeline_out.result

        for tag in _row_slice_tags(row):
            bucket = slice_counts.setdefault(tag, {"n": 0, "correct_product": 0, "correct_issue": 0})
            bucket["n"] += 1
            if result.product_category.value == row["expected_product_category"]:
                bucket["correct_product"] += 1
            if result.issue_type.value == row["expected_issue_type"]:
                bucket["correct_issue"] += 1

        if result.product_category.value == row["expected_product_category"]:
            correct_product += 1
        if result.issue_type.value == row["expected_issue_type"]:
            correct_issue += 1
        confidence_sum += result.confidence

    metrics = {
        "total": total,
        "correct_product": correct_product,
        "correct_issue": correct_issue,
        "product_accuracy": correct_product / total if total else 0.0,
        "issue_accuracy": correct_issue / total if total else 0.0,
        "avg_confidence": confidence_sum / total if total else 0.0,
        "slice_counts": slice_counts,
    }

    logger.info("Evaluation results: %s", metrics)
    return metrics


def _latest_benchmark_dataset_id() -> str | None:
    """Return the most recent DB-backed evaluation dataset id, if any."""
    session = SessionLocal()
    try:
        row = (
            session.query(EvaluationDataset)
            .order_by(EvaluationDataset.created_at.desc())
            .first()
        )
        return row.id if row is not None else None
    except Exception:
        return None
    finally:
        session.close()


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run complaint pipeline evals")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Legacy classification dataset filename inside datasets/ (optional)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (defaults to provider's default)",
    )
    parser.add_argument(
        "--benchmark-dataset-id",
        default=None,
        help="DB-backed benchmark dataset id to execute through the full workflow",
    )
    parser.add_argument(
        "--seed-cfpb-benchmark",
        action="store_true",
        help="Seed a stratified CFPB-backed benchmark dataset before running",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Sample size when seeding the CFPB-backed benchmark dataset",
    )
    args = parser.parse_args()

    from app.observability.logging import setup_logging

    setup_logging("INFO")

    if args.seed_cfpb_benchmark:
        seeded = seed_default_eval_dataset(sample_size=args.sample_size)
        print(json.dumps(seeded, indent=2))
        if not args.benchmark_dataset_id:
            benchmark_dataset_id = (
                (seeded.get("evaluation_dataset") or {}).get("dataset_id")
                if isinstance(seeded, dict)
                else None
            )
            args.benchmark_dataset_id = benchmark_dataset_id

    if args.benchmark_dataset_id:
        results = run_dataset_benchmark(args.benchmark_dataset_id)
    elif args.dataset:
        results = evaluate_classification(
            dataset_file=args.dataset, model_name=args.model
        )
    else:
        latest_dataset_id = _latest_benchmark_dataset_id()
        if latest_dataset_id:
            results = run_dataset_benchmark(latest_dataset_id)
        else:
            raise SystemExit(
                "No DB-backed evaluation dataset found. "
                "Run with --seed-cfpb-benchmark to create one, "
                "or pass --dataset <file> for the legacy CSV classification eval."
            )
    print(json.dumps(results, indent=2))
