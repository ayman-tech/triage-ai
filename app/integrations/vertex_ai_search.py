"""Optional Vertex AI Search adapter for managed complaint/policy retrieval."""

from __future__ import annotations

import os


def vertex_ai_search_configured() -> bool:
    return bool(
        os.getenv("GOOGLE_CLOUD_PROJECT")
        and os.getenv("VERTEX_AI_SEARCH_LOCATION")
        and os.getenv("VERTEX_AI_SEARCH_DATA_STORE_ID")
    )


def search_vertex_ai_datastore(query: str, page_size: int = 5) -> list[dict]:
    """Search a configured Vertex AI Search datastore.

    The existing pgvector retrieval remains the default. This function gives
    the agent framework a Google-managed retrieval path when env vars and
    dependencies are available.
    """
    if not vertex_ai_search_configured():
        raise RuntimeError(
            "Vertex AI Search is not configured. Set GOOGLE_CLOUD_PROJECT, "
            "VERTEX_AI_SEARCH_LOCATION, and VERTEX_AI_SEARCH_DATA_STORE_ID."
        )

    try:
        from google.cloud import discoveryengine_v1 as discoveryengine
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Install google-cloud-discoveryengine to use Vertex AI Search."
        ) from exc

    project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ["VERTEX_AI_SEARCH_LOCATION"]
    data_store_id = os.environ["VERTEX_AI_SEARCH_DATA_STORE_ID"]
    serving_config = (
        f"projects/{project_id}/locations/{location}/collections/default_collection/"
        f"dataStores/{data_store_id}/servingConfigs/default_config"
    )

    client = discoveryengine.SearchServiceClient()
    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=query,
        page_size=page_size,
    )
    response = client.search(request)
    results: list[dict] = []
    for item in response.results:
        document = item.document
        results.append(
            {
                "id": document.id,
                "name": document.name,
                "derived_struct_data": dict(document.derived_struct_data),
            }
        )
    return results
