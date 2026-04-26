"""Structured workflow events (merge into JSON logs via logging ``extra``)."""

from __future__ import annotations

import logging
import time
from typing import Any

from app.observability.context import get_active_run

logger = logging.getLogger("workflow.trace")


def _base_extra() -> dict[str, Any]:
    extra: dict[str, Any] = {"event_source": "workflow"}
    ar = get_active_run()
    if ar:
        extra["run_id"] = ar.run_id
        extra["company_id"] = ar.company_id
        if ar.case_id:
            extra["case_id"] = ar.case_id
        if ar.trace_id:
            extra["trace_id"] = ar.trace_id
    return extra


def log_workflow_event(event_type: str, **fields: Any) -> None:
    payload = {**_base_extra(), "event_type": event_type, **fields}
    logger.info(event_type, extra=payload)


def monotonic_ms() -> float:
    return time.perf_counter() * 1000.0
