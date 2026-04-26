import unittest
from unittest.mock import patch

from app.integrations.jira_client import create_complaint_ticket


class _FakeResponse:
    def __init__(self, status_code: int, text: str, json_data: dict | None = None) -> None:
        self.status_code = status_code
        self.text = text
        self._json_data = json_data or {}

    def json(self) -> dict:
        return self._json_data


class _FakeClient:
    def __init__(
        self,
        responses: list[_FakeResponse],
        recorder: list[tuple[str, object]],
        fields_payload: list[dict] | None = None,
    ) -> None:
        self._responses = responses
        self._recorder = recorder
        self._fields_payload = fields_payload or []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, auth=None, headers=None, json=None):
        self._recorder.append(("post", json))
        return self._responses.pop(0)

    def get(self, url, auth=None, headers=None):
        self._recorder.append(("get", url))
        return _FakeResponse(200, "ok", self._fields_payload)

    def put(self, url, auth=None, headers=None, json=None):
        self._recorder.append(("put", json))
        return _FakeResponse(204, "")


class JiraClientTests(unittest.TestCase):
    def test_retries_without_optional_fields_after_400(self) -> None:
        recorded_calls: list[tuple[str, object]] = []
        responses = [
            _FakeResponse(400, 'Field "customfield_10001" cannot be set.'),
            _FakeResponse(201, "created", {"key": "KAN-42"}),
        ]
        fields_payload = [
            {
                "id": "customfield_10001",
                "key": "customfield_10001",
                "name": "Team",
                "schema": {"custom": "com.atlassian.teams:rm-teams-custom-field-team"},
            }
        ]

        with patch.dict(
            "os.environ",
            {
                "JIRA_BASE_URL": "https://triageai.atlassian.net",
                "JIRA_USER_EMAIL": "tester@example.com",
                "JIRA_API_TOKEN": "secret",
                "JIRA_PROJECT_KEY": "KAN",
                "JIRA_ASSIGNEE_ID": "557058:test-user",
            },
            clear=False,
        ), patch(
            "app.integrations.jira_client.httpx.Client",
            side_effect=[
                _FakeClient(responses, recorded_calls, fields_payload),
                _FakeClient([], recorded_calls, fields_payload),
            ],
        ):
            ticket = create_complaint_ticket(
                case_id="abc12345deadbeef",
                team="credit_card_team",
                product_category="credit_card",
                issue_type="billing_dispute",
                risk_level="high",
                channel="web",
                consumer_narrative="Customer says they were charged twice.",
            )

        self.assertEqual(ticket["key"], "KAN-42")
        post_payloads = [payload for method, payload in recorded_calls if method == "post"]
        put_payloads = [payload for method, payload in recorded_calls if method == "put"]
        self.assertEqual(len(post_payloads), 2)
        self.assertEqual(len(put_payloads), 1)
        first_fields = post_payloads[0]["fields"]
        second_fields = post_payloads[1]["fields"]
        update_fields = put_payloads[0]["fields"]
        self.assertIn("assignee", first_fields)
        self.assertIn("customfield_10001", first_fields)
        self.assertNotIn("assignee", second_fields)
        self.assertNotIn("customfield_10001", second_fields)
        self.assertIn("assignee", update_fields)
        self.assertIn("customfield_10001", update_fields)


if __name__ == "__main__":
    unittest.main()
