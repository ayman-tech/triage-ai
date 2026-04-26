"""Synchronous bridge for running Google ADK LlmAgents.

ADK agents are async (``run_async`` returns an async generator). This module
wraps them so the existing synchronous complaint workflow can migrate to ADK
one specialist at a time.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from collections.abc import Callable
from contextvars import ContextVar, Token
from datetime import datetime
from typing import Any

from google.genai import types as genai_types

from app.agents.llm_factory import candidate_model_names, default_model_name
from app.agents.llm_json import parse_llm_json
from app.observability.cost import record_gemini_call
from app.observability.context import get_active_run

logger = logging.getLogger(__name__)

_session_service: Any | None = None
_active_adk_session: ContextVar[tuple[str, str] | None] = ContextVar(
    "active_adk_session",
    default=None,
)

_APP_NAME = "triage-ai"
_RETRYABLE_MODEL_ERROR_MARKERS = (
    "429",
    "RESOURCE_EXHAUSTED",
    "quota",
    "rate",
    "retry",
    "404",
    "NOT_FOUND",
    "is not found",
    "not supported for generateContent",
)


def _adk_imports() -> tuple[Any, Any, Any]:
    try:
        from google.adk.agents import LlmAgent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "google-adk is not installed in this Python environment. "
            "Install project dependencies before running ADK-backed agents."
        ) from exc
    return LlmAgent, Runner, InMemorySessionService


def _get_session_service() -> Any:
    global _session_service
    if _session_service is None:
        _, _, InMemorySessionService = _adk_imports()
        _session_service = InMemorySessionService()
    return _session_service


def _run_coro_sync(coro: Any) -> Any:
    """Run a coroutine from sync code, including inside an active event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}

    def _target() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # propagate back to caller thread
            result["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join()

    if "error" in result:
        raise result["error"]
    return result.get("value")


def set_adk_session(user_id: str, session_id: str) -> Token:
    """Bind subsequent ADK agent calls in this context to one session."""
    return _active_adk_session.set((user_id, session_id))


def reset_adk_session(token: Token) -> None:
    _active_adk_session.reset(token)


def _default_session_ids() -> tuple[str, str]:
    active_run = get_active_run()
    if active_run is not None:
        return "workflow", f"workflow-{active_run.run_id}"
    return "system", f"ephemeral-{uuid.uuid4().hex}"


async def _ensure_session(user_id: str, session_id: str) -> Any:
    session_service = _get_session_service()
    try:
        existing = await session_service.get_session(
            app_name=_APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        if existing is not None:
            return existing
    except Exception:
        logger.debug("ADK session lookup failed; creating a new session", exc_info=True)
    return await session_service.create_session(
        app_name=_APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )


def make_llm_agent(
    *,
    name: str,
    instruction: str,
    description: str = "",
    tools: list[Callable[..., Any]] | None = None,
    sub_agents: list[Any] | None = None,
    model_name: str | None = None,
    temperature: float = 0.0,
    output_schema: type[Any] | None = None,
    output_key: str | None = None,
) -> Any:
    """Create an ADK LlmAgent using the repo's model defaults."""
    LlmAgent, _, _ = _adk_imports()
    config = genai_types.GenerateContentConfig(temperature=temperature)
    kwargs: dict[str, Any] = {
        "name": name,
        "model": model_name or default_model_name(),
        "description": description,
        "instruction": instruction,
        "tools": tools or [],
        "sub_agents": sub_agents or [],
        "generate_content_config": config,
    }
    if output_schema is not None:
        kwargs["output_schema"] = output_schema
    if output_key is not None:
        kwargs["output_key"] = output_key
    return LlmAgent(**kwargs)


def run_adk_agent(agent: Any, user_message: str) -> str:
    """Run an ADK LlmAgent synchronously and return the final text response.

    Creates a fresh session per call so each specialist invocation is
    completely stateless.  The caller is responsible for parsing the
    returned string (typically JSON).
    """
    final_text, _ = _run_coro_sync(_run_async(agent, user_message))
    return final_text


def run_adk_json_agent(
    *,
    name: str,
    instruction: str,
    user_message: str,
    description: str = "",
    tools: list[Callable[..., Any]] | None = None,
    model_name: str | None = None,
    temperature: float = 0.0,
    output_schema: type[Any] | None = None,
    output_key: str | None = None,
    return_evidence: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, bool]]:
    """Run an ADK LlmAgent and parse the final response as a JSON object."""
    last_error: BaseException | None = None
    for candidate_model in candidate_model_names(model_name):
        agent = make_llm_agent(
            name=name,
            instruction=instruction,
            description=description,
            tools=tools,
            model_name=candidate_model,
            temperature=temperature,
            output_schema=output_schema,
            output_key=output_key,
        )
        try:
            text, tool_calls = _run_coro_sync(_run_async(agent, user_message))
            parsed = parse_llm_json(text)
            if return_evidence:
                return parsed, {name: True for name in sorted(tool_calls)}
            return parsed
        except Exception as exc:
            last_error = exc
            if not _is_retryable_model_error(exc):
                raise
            logger.warning(
                "ADK agent %s failed on model %s with retryable model error: %s",
                name,
                candidate_model,
                str(exc)[:300],
            )
    if last_error is not None:
        raise last_error
    raise RuntimeError("No Gemini model candidates configured.")


def _is_retryable_model_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(marker.lower() in text.lower() for marker in _RETRYABLE_MODEL_ERROR_MARKERS)


async def _run_async(agent: Any, user_message: str) -> tuple[str, set[str]]:
    _, Runner, _ = _adk_imports()
    session_service = _get_session_service()
    runner = Runner(
        agent=agent,
        app_name=_APP_NAME,
        session_service=session_service,
    )

    session_binding = _active_adk_session.get()
    user_id, session_id = session_binding or _default_session_ids()
    session_id = f"{session_id}:{agent.name}"
    session = await _ensure_session(user_id, session_id)

    new_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=user_message)],
    )

    final_text = ""
    tool_calls: set[str] = set()
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=new_message,
    ):
        _record_adk_event_usage(event, agent)
        content = getattr(event, "content", None)
        if content:
            for part in getattr(content, "parts", []) or []:
                fn_call = getattr(part, "function_call", None)
                name = getattr(fn_call, "name", None) if fn_call else None
                if name:
                    tool_calls.add(str(name))
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if part.text:
                    final_text += part.text

    if not final_text:
        logger.warning("ADK agent '%s' returned empty final response", agent.name)

    return final_text, tool_calls


def _record_adk_event_usage(event: Any, agent: Any) -> None:
    """Persist token usage from ADK events when an active run is available."""
    usage = getattr(event, "usage_metadata", None)
    if usage is None:
        return

    prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
    if prompt_tokens + completion_tokens + total_tokens <= 0:
        return

    now = datetime.utcnow()
    model_name = getattr(event, "model_version", None) or getattr(agent, "model", None)
    try:
        record_gemini_call(
            response=event,
            model_name=str(model_name or default_model_name()),
            started_at=now,
            ended_at=now,
        )
    except Exception:
        logger.debug("ADK cost recording skipped", exc_info=True)
