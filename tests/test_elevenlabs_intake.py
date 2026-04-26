"""Tests for ElevenLabs Custom LLM bridge."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.elevenlabs_intake import router as elevenlabs_router


def _test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(elevenlabs_router, prefix="/api/v1")
    return app


class ElevenLabsIntakeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(_test_app())
        os.environ.pop("ELEVENLABS_CUSTOM_LLM_SECRET", None)
        os.environ.pop("ELEVENLABS_INTAKE_REQUIRE_USER", None)

    def test_chat_completions_stream_returns_sse_and_intake_reply(self) -> None:
        from app.schemas.intake import IntakePacket, IntakeSessionState

        fake_state = IntakeSessionState(
            session_id="a" * 32,
            channel="voice",
            turn_index=1,
            packet=IntakePacket(channel="voice"),
            last_agent_message="Thanks — what product was involved?",
            last_user_message="",
            conversation_history=[],
        )

        with patch(
            "app.api.elevenlabs_intake._resolve_session_id",
            return_value=("a" * 32, False),
        ), patch(
            "app.api.elevenlabs_intake._run_intake_turn",
            return_value=fake_state,
        ):
            response = self.client.post(
                "/api/v1/integrations/elevenlabs/v1/chat/completions",
                json={
                    "model": "intake-voice",
                    "stream": True,
                    "user": "conv-123",
                    "messages": [
                        {"role": "system", "content": "You are an agent."},
                        {"role": "user", "content": "I was double billed."},
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers.get("content-type", ""))
        body = response.text
        self.assertIn("data: ", body)
        self.assertIn("[DONE]", body)
        self.assertIn("Thanks — what product was involved?", body)

    def test_chat_completions_requires_user_when_env_set(self) -> None:
        os.environ["ELEVENLABS_INTAKE_REQUIRE_USER"] = "1"
        try:
            response = self.client.post(
                "/api/v1/integrations/elevenlabs/v1/chat/completions",
                json={
                    "model": "intake-voice",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
        finally:
            os.environ.pop("ELEVENLABS_INTAKE_REQUIRE_USER", None)

        self.assertEqual(response.status_code, 400)

    def test_bearer_auth_rejects_wrong_token(self) -> None:
        os.environ["ELEVENLABS_CUSTOM_LLM_SECRET"] = "secret-test"
        try:
            response = self.client.post(
                "/api/v1/integrations/elevenlabs/v1/chat/completions",
                headers={"Authorization": "Bearer wrong"},
                json={
                    "model": "m",
                    "stream": False,
                    "user": "u1",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
        finally:
            os.environ.pop("ELEVENLABS_CUSTOM_LLM_SECRET", None)

        self.assertEqual(response.status_code, 401)

    def test_integration_health(self) -> None:
        r = self.client.get("/api/v1/integrations/elevenlabs/health")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("custom_llm_auth_configured", data)
        self.assertIn("tts_api_key_configured", data)


if __name__ == "__main__":
    unittest.main()
