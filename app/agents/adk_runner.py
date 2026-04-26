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
from typing import Any

from google.genai import types as genai_types

from app.agents.llm_factory import default_model_name
from app.agents.llm_json import parse_llm_json

logger = logging.getLogger(__name__)

_session_service: Any | None = None

_APP_NAME = "triage-ai"


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


def make_llm_agent(
    *,
    name: str,
    instruction: str,
    description: str = "",
    tools: list[Callable[..., Any]] | None = None,
    model_name: str | None = None,
    temperature: float = 0.0,
    output_schema: type[Any] | None = None,
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
        "generate_content_config": config,
    }
    if output_schema is not None:
        kwargs["output_schema"] = output_schema
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
    return_evidence: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, bool]]:
    """Run an ADK LlmAgent and parse the final response as a JSON object."""
    agent = make_llm_agent(
        name=name,
        instruction=instruction,
        description=description,
        tools=tools,
        model_name=model_name,
        temperature=temperature,
        output_schema=output_schema,
    )
    text, tool_calls = _run_coro_sync(_run_async(agent, user_message))
    parsed = parse_llm_json(text)
    if return_evidence:
        return parsed, {name: True for name in sorted(tool_calls)}
    return parsed


async def _run_async(agent: Any, user_message: str) -> tuple[str, set[str]]:
    _, Runner, _ = _adk_imports()
    session_service = _get_session_service()
    runner = Runner(
        agent=agent,
        app_name=_APP_NAME,
        session_service=session_service,
    )

    user_id = f"system-{uuid.uuid4().hex[:8]}"
    session = await session_service.create_session(
        app_name=_APP_NAME,
        user_id=user_id,
    )

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
