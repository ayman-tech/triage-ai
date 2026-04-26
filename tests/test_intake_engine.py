import unittest
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


class IntakeEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        _SESSIONS.clear()
        intake_engine._DB_SESSION_STORE_AVAILABLE = False

    def test_compute_sufficiency_requires_description_and_product_or_issue(self) -> None:
        packet = IntakePacket(
            product_hint="Credit card",
            issue_hint="Billing dispute",
            narrative_for_case="I was charged twice for one purchase.",
            prior_contact_attempted=False,
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
            prior_contact_attempted=True,
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

        payload = _build_case_payload(packet)

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

        session_id, _state = start_intake_session()

        def fake_adk_agent(**kwargs):
            captured_calls.append(kwargs)
            return {
                "assistant_message": "When did this happen?",
                "intake_packet": {
                    "product_hint": "Credit card",
                    "issue_hint": "Billing dispute",
                    "narrative_for_case": "I was charged twice for the same purchase.",
                    "customer_summary": "Customer says they were charged twice on a credit card purchase.",
                    "prior_contact_attempted": True,
                },
            }

        with patch("app.agents.intake_engine.run_adk_json_agent", side_effect=fake_adk_agent):
            state = process_intake_message(
                session_id,
                "My card is 4111 1111 1111 1111 and I was charged twice.",
            )

        self.assertEqual(state.packet.information_sufficiency, InformationSufficiency.SUFFICIENT)
        # Extract the user content sent to the model
        payload = __import__("json").loads(captured_calls[0]["user_message"])
        self.assertEqual(payload["last_user_message"], "My card is [CARD_REDACTED] and I was charged twice.")
        self.assertEqual(payload["conversation_history"][-1]["message"], payload["last_user_message"])
        self.assertIn("minimum information needed to document your complaint", state.last_agent_message)

    def test_process_message_falls_back_when_llm_returns_invalid_packet(self) -> None:
        session_id, _state = start_intake_session()

        with patch(
            "app.agents.intake_engine.run_adk_json_agent",
            return_value={
                "assistant_message": "Thanks.",
                "intake_packet": {
                    "intent": "not-a-real-intent",
                    "sentiment": "broken",
                },
            },
        ):
            state = process_intake_message(session_id, "I need help with a duplicate charge.")

        self.assertEqual(state.packet.intent, IntakeIntent.COMPLAINT)
        self.assertEqual(state.packet.information_sufficiency, InformationSufficiency.INSUFFICIENT)
        self.assertIn("couldn't process", state.last_agent_message)

    def test_company_intake_prompt_positions_agent_as_the_bank(self) -> None:
        prompt = _build_intake_system_prompt()

        self.assertIn("Mock Bank", prompt)


if __name__ == "__main__":
    unittest.main()
