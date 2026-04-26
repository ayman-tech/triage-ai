"""OpenTelemetry tracer setup (trace_id / span hierarchy for the workflow)."""

from __future__ import annotations

import logging
import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = logging.getLogger(__name__)

_TRACER_NAME = "complaint-workflow"
_initialized = False


def setup_tracing() -> None:
    """Install a SDK TracerProvider once. Optional console export for debugging."""
    global _initialized
    if _initialized:
        return

    resource = Resource.create(
        {
            "service.name": os.getenv("OTEL_SERVICE_NAME", "complaint-classification-agent"),
            "service.version": os.getenv("WORKFLOW_VERSION", "1.0.0"),
        }
    )
    provider = TracerProvider(resource=resource)

    if os.getenv("OTEL_TRACES_CONSOLE", "").lower() in ("1", "true", "yes"):
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.info("OpenTelemetry console span export enabled")

    trace.set_tracer_provider(provider)
    _initialized = True


def get_workflow_tracer():
    return trace.get_tracer(_TRACER_NAME)


def trace_id_hex_from_span(span) -> str:
    ctx = span.get_span_context()
    if not ctx or not ctx.trace_id:
        return ""
    return format(ctx.trace_id, "032x")
