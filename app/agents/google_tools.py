"""Optional Google Cloud-backed tools for ADK agents."""

from __future__ import annotations

import json

from app.integrations.vertex_ai_search import search_vertex_ai_datastore


def search_google_policy_knowledge(query: str, page_size: int = 5) -> str:
    """Search a configured Vertex AI Search datastore for policy/complaint knowledge.

    Args:
        query: Search query for policy, procedure, or complaint precedent.
        page_size: Number of results to return.
    """
    return json.dumps(
        search_vertex_ai_datastore(query=query, page_size=page_size),
        indent=2,
        default=str,
    )
