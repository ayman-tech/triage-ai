"""Plain Python tools for specialist agents.

These functions are passed directly to Google ADK LlmAgent instances (which
introspect their signatures and docstrings to build FunctionDeclarations) and
are also called manually in the Gemini function-calling loop in tool_loop.py.

All business logic is unchanged from the original implementation.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from app.documents.service import build_case_document_summary, search_case_documents as _search_case_documents

logger = logging.getLogger(__name__)

# ── Lazy singletons ──────────────────────────────────────────────────────────

_complaint_index = None
_resolution_index = None
_company_knowledge = None


def _vector_db_available() -> bool:
    from app.orchestrator.retrieval_gate import vector_db_available
    return vector_db_available()


def _complaint_index_singleton():
    global _complaint_index
    if not _vector_db_available():
        return None
    if _complaint_index is None:
        from app.retrieval.complaint_index import ComplaintIndex
        _complaint_index = ComplaintIndex()
    return _complaint_index


def _resolution_index_singleton():
    global _resolution_index
    if not _vector_db_available():
        return None
    if _resolution_index is None:
        from app.retrieval.resolution_index import ResolutionIndex
        _resolution_index = ResolutionIndex()
    return _resolution_index


def _company_knowledge_service():
    global _company_knowledge
    if _company_knowledge is None:
        from app.knowledge import CompanyKnowledgeService
        _company_knowledge = CompanyKnowledgeService()
    return _company_knowledge


# ── Retrieval tools ──────────────────────────────────────────────────────────

def search_similar_complaints(
    query: str,
    k: int = 3,
    product_filter: Optional[str] = None,
) -> str:
    """Search for historically similar consumer complaints using vector similarity.

    Use this tool to find past complaints that resemble the current one.
    Returns complaint narratives with metadata (product, issue, company).

    Args:
        query: The complaint narrative or keywords to search for.
        k: Number of similar complaints to return (default 3).
        product_filter: Optional product category to filter by.
    """
    index = _complaint_index_singleton()
    if index is None:
        return "Vector database is not available. No similar complaints found."

    docs = index.search(query, k=k, product_filter=product_filter)
    if not docs:
        return "No similar complaints found."

    results = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        results.append(
            f"--- Similar complaint {i} ---\n"
            f"Product: {m.get('product', 'N/A')}\n"
            f"Issue: {m.get('issue', 'N/A')}\n"
            f"Company: {m.get('company', 'N/A')}\n"
            f"Similarity: {1 - m.get('distance', 0):.2f}\n"
            f"Narrative: {doc.page_content}\n"
        )
    return "\n".join(results)


def search_similar_resolutions(
    query: str,
    k: int = 3,
) -> str:
    """Search for historically similar resolution outcomes using vector similarity.

    Use this tool to find how past complaints similar to this one were resolved.
    Returns resolution descriptions with outcome metadata.

    Args:
        query: The complaint narrative or keywords to search for.
        k: Number of similar resolutions to return (default 3).
    """
    index = _resolution_index_singleton()
    if index is None:
        return "Vector database is not available. No similar resolutions found."

    docs = index.search(query, k=k)
    if not docs:
        return "No similar resolutions found."

    results = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        results.append(
            f"--- Similar resolution {i} ---\n"
            f"Product: {m.get('product', 'N/A')}\n"
            f"Issue: {m.get('issue', 'N/A')}\n"
            f"Resolution outcome: {m.get('resolution_outcome', 'N/A')}\n"
            f"Similarity: {1 - m.get('distance', 0):.2f}\n"
            f"Details: {doc.page_content}\n"
        )
    return "\n".join(results)


# ── Knowledge tools ──────────────────────────────────────────────────────────

def lookup_company_taxonomy(narrative: str) -> str:
    """Retrieve operational taxonomy candidates relevant to this complaint.

    Returns product category and issue type candidates ranked by relevance
    to the complaint narrative. Use this to ground classification decisions.

    Args:
        narrative: The complaint narrative text.
    """
    svc = _company_knowledge_service()
    ctx = svc.build_company_context(narrative)
    return json.dumps(ctx.taxonomy_candidates, indent=2, default=str)


def lookup_severity_rubric(narrative: str) -> str:
    """Retrieve severity rubric and policy snippets relevant to this complaint.

    Returns severity level definitions and policy candidates ranked by relevance.
    Use this to ground risk assessment, compliance checks, and resolution planning.

    Args:
        narrative: The complaint narrative text.
    """
    svc = _company_knowledge_service()
    ctx = svc.build_company_context(narrative)
    return json.dumps(
        {
            "severity_candidates": ctx.severity_candidates,
            "policy_candidates": ctx.policy_candidates,
        },
        indent=2,
        default=str,
    )


def lookup_routing_rules() -> str:
    """Retrieve the routing matrix — team ownership by product category.

    Returns the mapping of product categories to internal teams, plus
    escalation team names. Use this to understand routing options.
    """
    svc = _company_knowledge_service()
    ctx = svc.build_company_context("")
    return json.dumps(ctx.routing_candidates, indent=2, default=str)


def lookup_root_cause_controls(narrative: str) -> str:
    """Retrieve root cause control knowledge relevant to this complaint.

    Returns control categories and checkpoints that help identify how
    internal failures typically arise. Use this to ground root cause analysis.

    Args:
        narrative: The complaint narrative text.
    """
    svc = _company_knowledge_service()
    ctx = svc.build_company_context(narrative)
    return json.dumps(ctx.root_cause_controls, indent=2, default=str)


def search_case_documents(
    case_id: str,
    query: str,
    k: int = 3,
) -> str:
    """Search processed uploaded documents attached to a specific case.

    Use this when the complaint has supporting documents and you need
    document-grounded evidence rather than narrative-only reasoning.

    Args:
        case_id: The unique case identifier.
        query: Search query text.
        k: Number of document chunks to return (default 3).
    """
    docs = _search_case_documents(case_id=case_id, query=query, k=k)
    if not docs:
        return "No processed case documents matched the query."
    results = []
    for i, doc in enumerate(docs, 1):
        results.append(
            f"--- Case document chunk {i} ---\n"
            f"Document ID: {doc.metadata.get('document_id')}\n"
            f"Chunk: {doc.metadata.get('chunk_index')}\n"
            f"Similarity: {1 - doc.metadata.get('distance', 0):.2f}\n"
            f"Content: {doc.page_content}\n"
        )
    return "\n".join(results)


def get_case_document_facts(case_id: str) -> str:
    """Return extracted document facts for a complaint case.

    Args:
        case_id: The unique case identifier.
    """
    summary = build_case_document_summary(case_id)
    if summary.total_documents == 0:
        return "No uploaded case documents are attached to this complaint."
    return json.dumps(summary.model_dump(), indent=2, default=str)
