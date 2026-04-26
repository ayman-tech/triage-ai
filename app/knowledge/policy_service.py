"""Service for managing company policy documents in the vector DB.

Policies are organized by product (e.g. checking_savings) and topic
(e.g. billing_disputes). Each uploaded document is chunked, embedded,
and stored in PolicyEmbedding for semantic retrieval.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from app.db.models import KnowledgeCollection, KnowledgeEntry, PolicyEmbedding
from app.db.session import SessionLocal
from app.retrieval.document import Document
from app.retrieval.embeddings import get_embeddings

logger = logging.getLogger(__name__)

# ── Display name mappings ──────────────────────────────────────────────────────

PRODUCT_DISPLAY_NAMES: dict[str, str] = {
    "checking_savings": "Checking & Savings",
    "credit_card": "Credit Card",
    "mortgage": "Mortgage",
    "student_loan": "Student Loan",
    "vehicle_loan": "Vehicle Loan",
    "debt_collection": "Debt Collection",
    "credit_reporting": "Credit Reporting",
    "payday_loan": "Personal / Payday Loan",
    "money_transfer": "Money Transfer",
    "prepaid_card": "Prepaid Card",
    "other": "Other",
}

TOPIC_DISPLAY_NAMES: dict[str, str] = {
    "billing_disputes": "Fees & Billing Disputes",
    "payment_processing": "Payment Processing",
    "account_management": "Account Management",
    "fraud_or_scam": "Fraud & Scams",
    "communication_tactics": "Communication",
    "incorrect_information": "Incorrect Information",
    "loan_modification": "Loan Modification",
    "disclosure_transparency": "Disclosures & Transparency",
    "closing_or_cancelling": "Closing & Cancellation",
    "other": "General Policy",
}

# Topics that belong to the Risk section
RISK_TOPICS = {"fraud_or_scam", "account_management", "loan_modification", "closing_or_cancelling"}
# Topics that belong to the Compliance section
COMPLIANCE_TOPICS = {"communication_tactics", "disclosure_transparency", "incorrect_information", "billing_disputes"}

_CHUNK_SIZE = 700
_CHUNK_OVERLAP = 120


def _chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        chunk_words = words[start: start + _CHUNK_SIZE]
        chunk = " ".join(chunk_words)
        if chunk.strip():
            chunks.append(chunk.strip())
        start += _CHUNK_SIZE - _CHUNK_OVERLAP
    return chunks


def _ensure_product_collection(session: Any, company_id: str, product: str) -> KnowledgeCollection:
    """Get or create a KnowledgeCollection for a company+product pair."""
    name = f"{company_id}::{product}"
    row = session.query(KnowledgeCollection).filter(KnowledgeCollection.name == name).first()
    if row is None:
        row = KnowledgeCollection(
            id=uuid.uuid4().hex,
            company_id=company_id,
            name=name,
            knowledge_type="policy",
            description=PRODUCT_DISPLAY_NAMES.get(product, product),
            status="active",
        )
        session.add(row)
        session.flush()
    return row


def ingest_policy(
    *,
    company_id: str = "mock_bank",
    product: str,
    topic: str,
    title: str,
    content_text: str,
    original_filename: str,
    storage_uri: str | None = None,
) -> dict:
    """Create a KnowledgeEntry, chunk+embed its text, store PolicyEmbedding rows."""
    session = SessionLocal()
    try:
        collection = _ensure_product_collection(session, company_id, product)

        entry = KnowledgeEntry(
            id=uuid.uuid4().hex,
            collection_id=collection.id,
            title=title,
            content=content_text,
            product=product,
            topic=topic,
            original_filename=original_filename,
            storage_uri=storage_uri,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        session.add(entry)
        session.flush()

        chunks = _chunk_text(content_text)
        if chunks:
            vectors = get_embeddings().embed_documents(chunks)
            rows = [
                PolicyEmbedding(
                    company_id=company_id,
                    entry_id=entry.id,
                    product=product,
                    topic=topic,
                    chunk_index=i,
                    content=chunk,
                    embedding=vec,
                )
                for i, (chunk, vec) in enumerate(zip(chunks, vectors))
            ]
            session.bulk_save_objects(rows)

        session.commit()
        logger.info(
            "Ingested policy: company=%s product=%s topic=%s chunks=%d",
            company_id, product, topic, len(chunks),
        )
        return {
            "entry_id": entry.id,
            "product": product,
            "product_display": PRODUCT_DISPLAY_NAMES.get(product, product),
            "topic": topic,
            "topic_display": TOPIC_DISPLAY_NAMES.get(topic, topic),
            "title": title,
            "original_filename": original_filename,
            "chunk_count": len(chunks),
            "created_at": entry.created_at.isoformat(),
        }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def list_policies(company_id: str = "mock_bank") -> dict[str, list[dict]]:
    """Return policies grouped by product key."""
    session = SessionLocal()
    try:
        entries = (
            session.query(KnowledgeEntry)
            .join(KnowledgeCollection)
            .filter(
                KnowledgeCollection.company_id == company_id,
                KnowledgeEntry.product.isnot(None),
            )
            .order_by(KnowledgeEntry.product, KnowledgeEntry.topic, KnowledgeEntry.created_at)
            .all()
        )

        result: dict[str, list[dict]] = {}
        for entry in entries:
            chunk_count = (
                session.query(PolicyEmbedding)
                .filter(PolicyEmbedding.entry_id == entry.id)
                .count()
            )
            product = entry.product or "other"
            result.setdefault(product, []).append({
                "entry_id": entry.id,
                "product": product,
                "product_display": PRODUCT_DISPLAY_NAMES.get(product, product),
                "topic": entry.topic or "other",
                "topic_display": TOPIC_DISPLAY_NAMES.get(entry.topic or "other", entry.topic or "other"),
                "title": entry.title,
                "original_filename": entry.original_filename or "",
                "chunk_count": chunk_count,
                "created_at": entry.created_at.isoformat() if entry.created_at else "",
            })
        return result
    finally:
        session.close()


def delete_policy(entry_id: str) -> bool:
    """Delete a KnowledgeEntry and its PolicyEmbedding rows. Returns True if found."""
    session = SessionLocal()
    try:
        entry = session.query(KnowledgeEntry).filter(KnowledgeEntry.id == entry_id).first()
        if entry is None:
            return False
        session.query(PolicyEmbedding).filter(PolicyEmbedding.entry_id == entry_id).delete()
        session.delete(entry)
        session.commit()
        return True
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def search_policies(
    query: str,
    *,
    product: str | None = None,
    topic: str | None = None,
    company_id: str = "mock_bank",
    k: int = 5,
) -> list[Document]:
    """Semantic search across PolicyEmbedding with optional product/topic filters."""
    from sqlalchemy import text as sql_text

    query_vec = get_embeddings().embed_query(query)
    session = SessionLocal()
    try:
        sql = """
            SELECT pe.content, pe.product, pe.topic, pe.chunk_index,
                   1 - (pe.embedding <=> CAST(:qvec AS vector)) AS score
            FROM policy_embeddings pe
            WHERE pe.company_id = :company_id
        """
        params: dict = {"qvec": str(query_vec), "company_id": company_id}
        if product:
            sql += " AND pe.product = :product"
            params["product"] = product
        if topic:
            sql += " AND pe.topic = :topic"
            params["topic"] = topic
        sql += " ORDER BY pe.embedding <=> CAST(:qvec AS vector) LIMIT :k"
        params["k"] = k

        rows = session.execute(sql_text(sql), params).fetchall()
        return [
            Document(
                page_content=row.content,
                metadata={
                    "product": row.product,
                    "topic": row.topic,
                    "chunk_index": row.chunk_index,
                    "score": float(row.score),
                },
            )
            for row in rows
        ]
    finally:
        session.close()
