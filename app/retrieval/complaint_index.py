"""Vector index for historical complaint narratives.

Backed by PostgreSQL + pgvector.  Uses the configured embedding model
(free local HuggingFace by default, or OpenAI if EMBEDDING_PROVIDER=openai).
"""

from __future__ import annotations

import logging
from typing import Optional

from app.retrieval.document import Document
from sqlalchemy import select, text

from app.db.models import ComplaintEmbedding
from app.db.session import SessionLocal
from app.retrieval.embeddings import get_embeddings

logger = logging.getLogger(__name__)


class ComplaintIndex:
    """Thin wrapper around pgvector for complaint similarity search.

    Stores and retrieves complaint narratives using HNSW‑indexed cosine
    similarity inside PostgreSQL.
    """

    def __init__(self) -> None:
        self._embeddings = get_embeddings()

    # ── Indexing ─────────────────────────────────────────────────────────

    def add_complaints(self, docs: list[Document]) -> None:
        """Embed and INSERT complaint documents into Postgres.

        Each ``Document`` must carry metadata keys that map to the
        ``ComplaintEmbedding`` columns (complaint_id, product, issue, …).
        """
        if not docs:
            return

        # Batch‑embed all page_content strings
        texts = [doc.page_content for doc in docs]
        vectors = self._embeddings.embed_documents(texts)

        session = SessionLocal()
        try:
            rows = []
            for doc, vec in zip(docs, vectors):
                m = doc.metadata
                rows.append(
                    ComplaintEmbedding(
                        complaint_id=m.get("complaint_id", ""),
                        content=doc.page_content,
                        embedding=vec,
                        product=m.get("product"),
                        sub_product=m.get("sub_product"),
                        issue=m.get("issue"),
                        sub_issue=m.get("sub_issue"),
                        company=m.get("company"),
                        state=m.get("state"),
                        zip_code=m.get("zip_code"),
                        date_received=m.get("date_received"),
                        submitted_via=m.get("submitted_via"),
                    )
                )
            session.bulk_save_objects(rows)
            session.commit()
            logger.info("Inserted %d complaint embeddings into Postgres", len(rows))
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ── Retrieval ────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 5,
        product_filter: Optional[str] = None,
        company_filter: Optional[str] = None,
    ) -> list[Document]:
        """Return the *k* most similar complaint narratives.

        Uses pgvector's cosine distance operator ``<=>`` with optional
        metadata pre‑filtering (product, company) pushed down to Postgres
        so the index handles it efficiently.
        """
        query_vec = self._embeddings.embed_query(query)

        session = SessionLocal()
        try:
            stmt = (
                select(
                    ComplaintEmbedding.content,
                    ComplaintEmbedding.complaint_id,
                    ComplaintEmbedding.product,
                    ComplaintEmbedding.issue,
                    ComplaintEmbedding.company,
                    ComplaintEmbedding.state,
                    ComplaintEmbedding.embedding.cosine_distance(query_vec).label("distance"),
                )
                .order_by("distance")
                .limit(k)
            )

            if product_filter:
                stmt = stmt.where(ComplaintEmbedding.product == product_filter)
            if company_filter:
                stmt = stmt.where(ComplaintEmbedding.company == company_filter)

            results = session.execute(stmt).all()

            docs = []
            for row in results:
                docs.append(
                    Document(
                        page_content=row.content,
                        metadata={
                            "complaint_id": row.complaint_id,
                            "product": row.product,
                            "issue": row.issue,
                            "company": row.company,
                            "state": row.state,
                            "distance": float(row.distance),
                        },
                    )
                )

            logger.debug("Complaint search returned %d results (k=%d)", len(docs), k)
            return docs

        finally:
            session.close()

    def search_with_scores(
        self,
        query: str,
        k: int = 5,
        product_filter: Optional[str] = None,
    ) -> list[tuple[Document, float]]:
        """Return the *k* most similar complaints with cosine distance scores."""
        docs = self.search(query, k=k, product_filter=product_filter)
        return [(doc, doc.metadata.get("distance", 0.0)) for doc in docs]

    # ── Stats ────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Return the total number of complaint embeddings stored."""
        session = SessionLocal()
        try:
            result = session.execute(
                text("SELECT COUNT(*) FROM complaint_embeddings")
            )
            return result.scalar() or 0
        finally:
            session.close()
