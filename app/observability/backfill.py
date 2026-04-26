"""Backfill historical cost ledger entries from legacy workflow aggregates."""

from __future__ import annotations

import json
import logging
from datetime import datetime

from sqlalchemy import or_
from sqlalchemy.exc import SQLAlchemyError

from app.db.models import ComplaintCase, LLMCallCost, WorkflowRun
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)


def backfill_cost_ledger_from_workflow_runs(limit: int | None = None) -> int:
    """Create aggregate ledger rows for historical runs that predate call-level tracking.

    For legacy workflow runs we only know complaint/run totals, not per-agent or per-call
    splits. This function inserts exactly one synthetic ``llm_call_costs`` row per missing
    run so aggregate analytics can use historical spend while agent analytics remain clean.
    Synthetic rows keep ``agent_name`` null and set ``metadata_json.backfilled_aggregate_only``.
    """
    inserted = 0
    skipped = 0
    try:
        session = SessionLocal()
        try:
            query = (
                session.query(WorkflowRun, ComplaintCase)
                .outerjoin(LLMCallCost, LLMCallCost.run_id == WorkflowRun.run_id)
                .outerjoin(ComplaintCase, ComplaintCase.id == WorkflowRun.case_id)
                .filter(LLMCallCost.id.is_(None))
                .filter(
                    or_(
                        WorkflowRun.cost_estimate_total.isnot(None),
                        WorkflowRun.token_total.isnot(None),
                        ComplaintCase.cost_estimate_usd.isnot(None),
                        ComplaintCase.token_total.isnot(None),
                    )
                )
                .order_by(WorkflowRun.started_at.asc())
            )
            if limit is not None:
                query = query.limit(limit)

            rows = query.all()
            for run, case in rows:
                token_total = run.token_total
                if token_total is None and case is not None:
                    token_total = case.token_total
                token_total = int(token_total or 0)

                total_cost = run.cost_estimate_total
                if total_cost is None and case is not None:
                    total_cost = case.cost_estimate_usd
                if total_cost is None:
                    total_cost = 0.0

                started_at = run.started_at or datetime.utcnow()
                ended_at = run.ended_at or started_at
                resolved_case_id = case.id if case is not None else None

                try:
                    session.add(
                        LLMCallCost(
                            run_id=run.run_id,
                            case_id=resolved_case_id,
                            sequence_number=None,
                            agent_name=None,
                            langsmith_run_id=None,
                            provider=None,
                            model_name=run.model_version,
                            prompt_tokens=token_total,
                            completion_tokens=0,
                            total_tokens=token_total,
                            input_cost_usd=float(total_cost),
                            output_cost_usd=0.0,
                            total_cost_usd=float(total_cost),
                            latency_ms=max((ended_at - started_at).total_seconds() * 1000.0, 0.0),
                            status="backfilled_aggregate",
                            retry_number=run.retry_count_total or 0,
                            started_at=started_at,
                            ended_at=ended_at,
                            metadata_json=json.dumps(
                                {
                                    "backfilled_aggregate_only": True,
                                    "source": "workflow_runs",
                                    "legacy_run_cost_estimate_total": run.cost_estimate_total,
                                    "legacy_case_cost_estimate_usd": getattr(case, "cost_estimate_usd", None),
                                    "orphaned_case_reference": case is None and run.case_id is not None,
                                    "original_case_id": run.case_id,
                                },
                                separators=(",", ":"),
                            ),
                        )
                    )
                    session.commit()
                    inserted += 1
                except SQLAlchemyError as exc:
                    session.rollback()
                    skipped += 1
                    logger.warning(
                        "Skipping historical cost backfill row for run %s: %s",
                        run.run_id,
                        exc,
                    )

            if inserted:
                logger.info(
                    "Backfilled %d historical cost ledger rows (%d skipped)",
                    inserted,
                    skipped,
                )
            else:
                session.rollback()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    except SQLAlchemyError as exc:
        logger.warning("Historical cost ledger backfill skipped: %s", exc)
        return 0

    return inserted
