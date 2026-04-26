"""LLM token and cost tracking for Gemini API calls."""

from __future__ import annotations

import logging
from datetime import datetime

from app.observability.context import get_active_run, get_active_step
from app.observability.persistence import insert_llm_call_cost

logger = logging.getLogger(__name__)

# USD per 1k input/output tokens
_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.0003, 0.0025),
    "gemini-2.5-flash-lite": (0.0001, 0.0004),
    "gemini-2.0-flash": (0.000075, 0.0003),
    "gemini-2.0-flash-lite": (0.000075, 0.0003),
    "gemini-1.5-flash": (0.000075, 0.0003),
    "gemini-1.5-pro": (0.00125, 0.005),
    "gemini-1.0-pro": (0.0005, 0.0015),
}

_DEFAULT_PRICING = (0.001, 0.001)


def _pricing_for(model_name: str | None) -> tuple[float, float]:
    if not model_name:
        return _DEFAULT_PRICING
    key = model_name.lower()
    for name, rates in _PRICING.items():
        if key.startswith(name):
            return rates
    return _DEFAULT_PRICING


def estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    model_name: str | None,
) -> float:
    input_rate, output_rate = _pricing_for(model_name)
    return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1000.0


def estimate_cost_breakdown_usd(
    prompt_tokens: int,
    completion_tokens: int,
    model_name: str | None,
) -> tuple[float, float, float]:
    input_rate, output_rate = _pricing_for(model_name)
    input_cost = (prompt_tokens * input_rate) / 1000.0
    output_cost = (completion_tokens * output_rate) / 1000.0
    return input_cost, output_cost, input_cost + output_cost


def record_gemini_call(
    response: object,
    *,
    model_name: str,
    started_at: datetime,
    ended_at: datetime,
) -> None:
    """Persist token counts and cost from a Gemini GenerateContentResponse.

    Called after each generate_content() call in tool_loop.py.
    No-ops gracefully when there is no active workflow run.
    """
    active_run = get_active_run()
    if active_run is None:
        return

    try:
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    except Exception:
        prompt_tokens, completion_tokens = 0, 0

    input_cost, output_cost, total_cost = estimate_cost_breakdown_usd(
        prompt_tokens, completion_tokens, model_name
    )

    active_step = get_active_step()
    latency_ms = max((ended_at - started_at).total_seconds() * 1000.0, 0.0)

    try:
        insert_llm_call_cost(
            run_id=active_run.run_id,
            case_id=active_run.case_id,
            sequence_number=active_step.sequence_number if active_step else None,
            agent_name=active_step.node_name if active_step else None,
            langsmith_run_id=None,
            provider="google",
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            total_cost_usd=total_cost,
            latency_ms=latency_ms,
            status="success",
            retry_number=active_step.retry_number if active_step else 0,
            started_at=started_at,
            ended_at=ended_at,
            metadata={
                "trace_id": active_run.trace_id,
                "provider": "google",
                "model_name": model_name,
            },
        )
    except Exception:
        logger.exception("Failed to persist llm_call_costs entry")


class TokenCostAccumulator:
    """Simple accumulator for tracking total tokens across a workflow run.

    Replaces the LangChain BaseCallbackHandler pattern. Callers manually
    call add_usage() after each Gemini API call.
    """

    def __init__(self) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.llm_call_count: int = 0

    def add_usage(self, response: object) -> None:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return
        self.prompt_tokens += int(getattr(usage, "prompt_token_count", 0) or 0)
        self.completion_tokens += int(getattr(usage, "candidates_token_count", 0) or 0)
        self.llm_call_count += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
