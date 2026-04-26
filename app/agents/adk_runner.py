"""Synchronous bridge for running Google ADK LlmAgents.

ADK agents are async (run_async returns an async generator).  This module
wraps them so they can be called from synchronous agent functions without
changing the rest of the codebase to async.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

# One session service shared across all specialist agent runs
_session_service = InMemorySessionService()

_APP_NAME = "triage-ai"


def run_adk_agent(agent: LlmAgent, user_message: str) -> str:
    """Run an ADK LlmAgent synchronously and return the final text response.

    Creates a fresh session per call so each specialist invocation is
    completely stateless.  The caller is responsible for parsing the
    returned string (typically JSON).
    """
    return asyncio.run(_run_async(agent, user_message))


async def _run_async(agent: LlmAgent, user_message: str) -> str:
    runner = Runner(
        agent=agent,
        app_name=_APP_NAME,
        session_service=_session_service,
    )

    user_id = f"system-{uuid.uuid4().hex[:8]}"
    session = await _session_service.create_session(
        app_name=_APP_NAME,
        user_id=user_id,
    )

    new_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=user_message)],
    )

    final_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=new_message,
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if part.text:
                    final_text += part.text

    if not final_text:
        logger.warning("ADK agent '%s' returned empty final response", agent.name)

    return final_text
