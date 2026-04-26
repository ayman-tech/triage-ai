import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import app.agents.intake_engine as intake_engine
from app.agents.intake_engine import (
    _SESSIONS,
    _build_case_payload,
    _build_intake_system_prompt,
    _compute_sufficiency,
    process_intake_message,
    start_intake_session,
)
from app.schemas.intake import (
    InformationSufficiency,
    IntakeIntent,
    IntakePacket,
    RecommendedHandoff,
)


def _make_fake_client(response_text: str, capture_list: list | None = None):
    """Build a fake google.genai.Client that returns a fixed text response."""
    class FakeUsage:
        prompt_token_count = 0
        candidates_token_count = 0

    class FakeResponse:
        text = response_text
        candidates = []
        usage_metadata = FakeUsage()

    class FakeModels:
        def generate_content(self, model, contents, config=None):
            if capture_list is not None:
                capture_list.append({"contents": contents, "config": config})
            return FakeResponse()

    class FakeClient:
        models = FakeModels()

    return FakeClient()


class IntakeEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        _SESSIONS.clear()
        intake_engine._DB_SESSION_STORE_AVAILABLE = False

    def test_compute_sufficiency_requires_description_and_product_or_issue(self) -> None:
        packet = IntakePacket(
            product_hint="Credit card",
            issue_hint="Billing dispute",
            narrative_for_case="I was charged twice for one purchase.",
        )

        updated = _compute_sufficiency(packet)

        self.assertEqual(updated.information_sufficiency, InformationSufficiency.SUFFICIENT)
        self.assertEqual(updated.recommended_handoff, RecommendedHandoff.SUPERVISOR)
        self.assertEqual(updated.missing_fields, [])

    def test_compute_sufficiency_marks_urgent_fraud_for_human_escalation(self) -> None:
        packet = IntakePacket(
            intent=IntakeIntent.FRAUD_REPORT,
            product_hint="Checking account",
            issue_hint="Unauthorized transfer",
            narrative_for_case="There are transfers I did not authorize from my account.",
            urgency="high",
            escalation_reasons=["fraud_suspected"],
        )

        updated = _compute_sufficiency(packet)

        self.assertEqual(updated.information_sufficiency, InformationSufficiency.SUFFICIENT)
        self.assertEqual(updated.recommended_handoff, RecommendedHandoff.HUMAN_ESCALATION)

    def test_build_case_payload_keeps_issue_out_of_sub_product(self) -> None:
        packet = IntakePacket(
            channel="voice",
            product_hint="Credit card",
            issue_hint="Billing dispute",
            sub_issue_hint="Duplicate charge",
            customer_summary="Customer reports a duplicate credit card charge that was not reversed.",
        )

        payload = _build_case_payload(packet, "company-123")

        self.assertEqual(payload["company_id"], "company-123")
        self.assertEqual(payload["channel"], "phone")
        self.assertIsNone(payload["sub_product"])
        self.assertEqual(payload["external_issue_type"], "Billing dispute / Duplicate charge")
        self.assertEqual(payload["intake_intent"], "complaint")
        self.assertEqual(payload["intake_urgency"], "medium")
        self.assertEqual(payload["intake_recommended_handoff"], "supervisor")
        self.assertEqual(
            payload["consumer_narrative"],
            "Customer reports a duplicate credit card charge that was not reversed.",
        )

    def test_process_message_sends_redacted_transcript_window(self) -> None:
        captured_calls = []

        response_payload = json.dumps(
            {
                "assistant_message": "When did this happen?",
                "intake_packet": {
                    "product_hint": "Credit card",
                    "issue_hint": "Billing dispute",
                    "narrative_for_case": "I was charged twice for the same purchase.",
                    "customer_summary": "Customer says they were charged twice on a credit card purchase.",
                },
            }
        )

        session_id, _state = start_intake_session()

        with patch(
            "app.agents.intake_engine.get_gemini_client",
            return_value=_make_fake_client(response_payload, capture_list=captured_calls),
        ):
            state = process_intake_message(
                session_id,
                "My card is 4111 1111 1111 1111 and I was charged twice.",
            )

        self.assertEqual(state.packet.information_sufficiency, InformationSufficiency.SUFFICIENT)
        # Extract the user content sent to the model
        contents = captured_calls[0]["contents"]
        user_text = contents[0].parts[0].text
        payload = json.loads(user_text)
        self.assertEqual(payload["last_user_message"], "My card is [CARD_REDACTED] and I was charged twice.")
        self.assertEqual(payload["conversation_history"][-1]["message"], payload["last_user_message"])
        self.assertIn("minimum information needed to file your complaint", state.last_agent_message)

    def test_process_message_falls_back_when_llm_returns_invalid_packet(self) -> None:
        response_payload = json.dumps(
            {
                "assistant_message": "Thanks.",
                "intake_packet": {
                    "intent": "not-a-real-intent",
                    "sentiment": "broken",
                },
            }
        )

        session_id, _state = start_intake_session()

        with patch(
            "app.agents.intake_engine.get_gemini_client",
            return_value=_make_fake_client(response_payload),
        ):
            state = process_intake_message(session_id, "I need help with a duplicate charge.")

        self.assertEqual(state.packet.intent, IntakeIntent.COMPLAINT)
        self.assertEqual(state.packet.information_sufficiency, InformationSufficiency.INSUFFICIENT)
        self.assertIn("couldn't process", state.last_agent_message)

    def test_company_intake_prompt_positions_agent_as_the_bank(self) -> None:
        prompt = _build_intake_system_prompt("mock_bank")

        self.assertIn("Mock Bank", prompt)


if __name__ == "__main__":
    unittest.main()
