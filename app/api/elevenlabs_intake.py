"""ElevenLabs Conversational AI integration: OpenAI-compatible Custom LLM + optional TTS.

Configure an ElevenLabs agent with Custom LLM and set the server URL to::

    {PUBLIC_BASE_URL}/api/v1/integrations/elevenlabs

so requests hit ``POST .../v1/chat/completions`` (SSE).

Multi-turn intake requires a stable per-conversation key. Pass ElevenLabs' ``user_id``
field through to OpenAI ``user`` (ElevenLabs maps ``user_id`` → ``user``). This
implementation uses that value to map to an ``IntakeSessionState`` (creating a
``voice`` session on first use). Alternatively set the ``user`` / ``user_id``
value to an existing intake ``session_id`` from ``POST /api/v1/intake/session``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from app.agents.intake_engine import (
    get_intake_session,
    process_intake_message,
    start_intake_session,
)
from app.env_elevenlabs import elevenlabs_api_key, elevenlabs_voice_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/integrations/elevenlabs", tags=["elevenlabs-intake"])

# Maps ElevenLabs/OpenAI ``user`` key → intake session_id (voice channel).
_USER_KEY_TO_SESSION: dict[str, str] = {}

_HEX_SESSION = re.compile(r"^[0-9a-f]{32}$", re.IGNORECASE)


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").lower() in ("1", "true", "yes", "on")


def _custom_llm_secret() -> str | None:
    return os.getenv("ELEVENLABS_CUSTOM_LLM_SECRET") or os.getenv("CUSTOM_LLM_BEARER_TOKEN")


def _require_user_key() -> bool:
    return _truthy_env("ELEVENLABS_INTAKE_REQUIRE_USER")


async def verify_custom_llm_auth(request: Request) -> None:
    secret = _custom_llm_secret()
    if not secret:
        return
    auth = request.headers.get("authorization") or ""
    if not auth.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
        )
    token = auth[7:].strip()
    if token != secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
        )


def _extract_user_key(body: dict[str, Any]) -> str | None:
    u = body.get("user")
    if u is not None and str(u).strip():
        return str(u).strip()
    uid = body.get("user_id")
    if uid is not None and str(uid).strip():
        return str(uid).strip()
    return None


def _extract_last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                elif isinstance(part, str):
                    parts.append(part)
            return " ".join(parts).strip()
    return ""


def _resolve_session_id(user_key: str | None) -> tuple[str, bool]:
    """Return (session_id, created_new_mapping).

    If ``user_key`` is a known intake session id, use it. Otherwise map
    ``user_key`` to a session (creating one for a new key).
    """
    if user_key and _HEX_SESSION.match(user_key):
        if get_intake_session(user_key) is not None:
            return user_key, False

    if user_key:
        mapped = _USER_KEY_TO_SESSION.get(user_key)
        if mapped and get_intake_session(mapped) is not None:
            return mapped, False

    sid, _ = start_intake_session(channel="voice")
    if user_key:
        _USER_KEY_TO_SESSION[user_key] = sid
    return sid, True


def _run_intake_turn(session_id: str, user_text: str) -> Any:
    return process_intake_message(session_id=session_id, user_message=user_text)


def _sse_chat_chunk(
    *,
    chunk_id: str,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None,
) -> str:
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_intake_reply(
    *,
    model: str,
    session_id: str,
    user_text: str,
) -> AsyncIterator[str]:
    chunk_id = f"chatcmpl-{session_id[:12]}"
    try:
        state = await asyncio.to_thread(_run_intake_turn, session_id, user_text)
    except KeyError:
        yield f"data: {json.dumps({'error': {'message': f'Unknown session {session_id}'}})}\n\n"
        return
    except Exception as exc:
        logger.exception("ElevenLabs custom LLM intake turn failed")
        yield f"data: {json.dumps({'error': {'message': str(exc)}})}\n\n"
        return

    text = (state.last_agent_message or "").strip()
    if not text:
        text = "I'm here to help. Please describe what happened and which product or service it relates to."

    # Role + content in one chunk is valid; split deltas if you need finer-grained TTS pacing.
    yield _sse_chat_chunk(
        chunk_id=chunk_id,
        model=model,
        delta={"role": "assistant", "content": text},
        finish_reason=None,
    )
    yield _sse_chat_chunk(
        chunk_id=chunk_id,
        model=model,
        delta={},
        finish_reason="stop",
    )
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
async def elevenlabs_custom_llm(
    request: Request,
    _auth: None = Depends(verify_custom_llm_auth),
) -> StreamingResponse:
    """OpenAI-compatible streaming chat completions for ElevenLabs Custom LLM."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Expected JSON body") from None

    model = str(body.get("model") or "intake-voice")
    stream = body.get("stream", True)
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be a list")

    user_key = _extract_user_key(body)
    if user_key is None and _require_user_key():
        raise HTTPException(
            status_code=400,
            detail="Set user or user_id (per-conversation key) for multi-turn intake; "
            "or disable ELEVENLABS_INTAKE_REQUIRE_USER.",
        )
    if user_key is None:
        logger.warning(
            "ElevenLabs custom LLM: missing user/user_id; a new voice intake session is started "
            "every turn. Configure a per-conversation user_id in ElevenLabs or set "
            "ELEVENLABS_INTAKE_REQUIRE_USER=1 to fail closed."
        )

    user_text = _extract_last_user_text(messages)
    if not user_text:
        raise HTTPException(
            status_code=400,
            detail="No user message found in messages (last user role content empty).",
        )

    session_id, _created = _resolve_session_id(user_key)

    if not stream:
        # Non-streaming response (rare for EL); still return JSON shape OpenAI uses.
        try:
            state = await asyncio.to_thread(_run_intake_turn, session_id, user_text)
        except KeyError:
            raise HTTPException(status_code=404, detail="Intake session not found") from None
        text = state.last_agent_message or ""
        return Response(
            content=json.dumps(
                {
                    "id": f"chatcmpl-{session_id[:12]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            media_type="application/json",
        )

    return StreamingResponse(
        _stream_intake_reply(model=model, session_id=session_id, user_text=user_text),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice_id: str | None = Field(
        None,
        description="ElevenLabs voice id; defaults to ELEVENLABS_VOICE_ID env.",
    )


def synthesize_speech_bytes(text: str, voice_id: str | None = None) -> tuple[bytes, str]:
    """Call ElevenLabs TTS; returns (audio_bytes, content_type)."""
    api_key = elevenlabs_api_key()
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ELEVENLABS_API_KEY is not configured",
        )
    voice = voice_id or elevenlabs_voice_id()
    if not voice:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="voice_id is required or set ELEVENLABS_VOICE_ID",
        )
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
    payload = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            audio = resp.read()
            content_type = resp.headers.get("Content-Type", "audio/mpeg")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        logger.warning("ElevenLabs TTS HTTP %s: %s", e.code, err_body[:500])
        detail = f"ElevenLabs TTS upstream error {e.code}"
        if e.code == 401:
            detail = "ElevenLabs TTS rejected the API key (401). Check ELEVENLABS_API_KEY."
        elif e.code == 402:
            detail = (
                "ElevenLabs TTS returned 402 Payment Required. "
                "Your ElevenLabs account likely has no credits or this feature is unavailable on the current plan."
            )
        elif e.code == 403:
            detail = "ElevenLabs TTS returned 403 Forbidden. Check account permissions, voice access, or project restrictions."
        elif e.code == 404:
            detail = "ElevenLabs TTS could not find the configured voice. Check ELEVENLABS_VOICE_ID."
        elif e.code == 422:
            detail = f"ElevenLabs TTS rejected the request payload (422): {err_body[:200]}"
        raise HTTPException(status_code=502, detail=detail) from None
    except Exception as exc:
        logger.exception("ElevenLabs TTS request failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return audio, content_type


@router.post(
    "/tts",
    summary="Text-to-speech (optional helper for non-Agent clients)",
    response_class=Response,
)
async def elevenlabs_tts(
    body: TtsRequest,
    _auth: None = Depends(verify_custom_llm_auth),
) -> Response:
    """Synthesize speech via ElevenLabs HTTP API (for testing or custom clients)."""
    audio, content_type = synthesize_speech_bytes(body.text, body.voice_id)
    return Response(content=audio, media_type=content_type)


@router.get(
    "/health",
    summary="ElevenLabs integration health",
)
async def elevenlabs_integration_health() -> dict[str, Any]:
    return {
        "custom_llm_auth_configured": bool(_custom_llm_secret()),
        "tts_api_key_configured": bool(elevenlabs_api_key()),
        "tts_voice_id_configured": bool(elevenlabs_voice_id()),
        "tts_enabled": bool(elevenlabs_api_key() and elevenlabs_voice_id()),
        "require_user_key": _require_user_key(),
    }
