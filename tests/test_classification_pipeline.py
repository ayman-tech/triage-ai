import unittest
from unittest.mock import patch

from app.agents.classification import run_classification
from app.schemas.case import CaseRead

class ClassificationPipelineTests(unittest.TestCase):
    def test_trivial_case_skips_llm_execution(self) -> None:
        case = CaseRead(
            consumer_narrative="",
            cfpb_product="Credit card",
            cfpb_issue="Billing dispute",
        )

        with patch(
            "app.agents.classification.run_adk_json_agent",
            side_effect=AssertionError("ADK LLM execution should not run for trivial cases"),
        ):
            out = run_classification(case=case)

        self.assertEqual(out.audit.evidence_used, {})
        self.assertTrue(out.audit.assess_skipped_llm)
        self.assertFalse(out.audit.plan["needs_retrieval"])

    def test_ambiguous_case_uses_adk_agent_with_tools(self) -> None:
        case = CaseRead(
            consumer_narrative="My account was frozen and then I also saw an unauthorized transfer.",
            product="Checking account",
            cfpb_product="Checking or savings account",
            cfpb_issue="Managing an account",
        )

        assess = {
            "complexity": "ambiguous",
            "narrative_status": "present",
            "structured_field_completeness": "partial",
            "consistency": "partial_conflict",
            "conflict_score": 0.6,
            "recommended_weighting": "narrative",
            "rationale": "Narrative suggests overlapping access and fraud problems.",
        }

        result = {
            "product_category": "checking_savings",
            "issue_type": "fraud_or_scam",
            "sub_issue": "unauthorized_transaction",
            "confidence": 0.61,
            "reasoning": "Narrative and retrieval indicate the primary issue is an unauthorized transfer.",
            "keywords": ["account frozen", "unauthorized transfer"],
            "review_recommended": True,
            "reason_codes": ["retrieval_used"],
            "alternate_candidates": [],
        }

        with patch(
            "app.agents.classification.run_adk_json_agent",
            side_effect=[
                assess,
                (result, {"search_similar_complaints": True, "lookup_company_taxonomy": True}),
            ],
        ) as adk_agent:
            out = run_classification(case=case)

        self.assertEqual(adk_agent.call_count, 2)
        self.assertTrue(out.audit.plan["needs_retrieval"])
        self.assertIn("search_similar_complaints", out.audit.evidence_used)


if __name__ == "__main__":
    unittest.main()
