"""Vector index for historical resolution outcomes.

Backed by PostgreSQL + pgvector.  Uses the configured embedding model
(free local HuggingFace by default, or OpenAI if EMBEDDING_PROVIDER=openai).
"""

from __future__ import annotations

import logging
from typing import Optional

from app.retrieval.document import Document
from sqlalchemy import select, text

from app.db.models import ResolutionEmbedding
from app.db.session import SessionLocal
from app.retrieval.embeddings import get_embeddings

logger = logging.getLogger(__name__)


class ResolutionIndex:
    """Thin wrapper around pgvector for resolution similarity search."""

    def __init__(self) -> None:
        self._embeddings = get_embeddings()

    # ── Indexing ─────────────────────────────────────────────────────────

    def add_resolutions(self, docs: list[Document]) -> None:
        """Embed and INSERT resolution documents into Postgres."""
        if not docs:
            return

        texts = [doc.page_content for doc in docs]
        vectors = self._embeddings.embed_documents(texts)

        session = SessionLocal()
        try:
            rows = []
            for doc, vec in zip(docs, vectors):
                m = doc.metadata
                rows.append(
                    ResolutionEmbedding(
                        complaint_id=m.get("complaint_id", ""),
                        content=doc.page_content,
                        embedding=vec,
                        product=m.get("product"),
                        issue=m.get("issue"),
                        company=m.get("company"),
                        resolution_outcome=m.get("resolution_outcome"),
                        date_received=m.get("date_received"),
                    )
                )
            session.bulk_save_objects(rows)
            session.commit()
            logger.info("Inserted %d resolution embeddings into Postgres", len(rows))
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
        resolution_filter: Optional[str] = None,
    ) -> list[Document]:
        """Return the *k* most similar resolution precedents.

        Uses pgvector cosine distance with optional metadata filtering
        on product and resolution outcome.
        """
        query_vec = self._embeddings.embed_query(query)

        session = SessionLocal()
        try:
            stmt = (
                select(
                    ResolutionEmbedding.content,
                    ResolutionEmbedding.complaint_id,
                    ResolutionEmbedding.product,
                    ResolutionEmbedding.issue,
                    ResolutionEmbedding.company,
                    ResolutionEmbedding.resolution_outcome,
                    ResolutionEmbedding.embedding.cosine_distance(query_vec).label("distance"),
                )
                .order_by("distance")
                .limit(k)
            )

            if product_filter:
                stmt = stmt.where(ResolutionEmbedding.product == product_filter)
            if resolution_filter:
                stmt = stmt.where(ResolutionEmbedding.resolution_outcome == resolution_filter)

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
                            "resolution_outcome": row.resolution_outcome,
                            "distance": float(row.distance),
                        },
                    )
                )

            logger.debug("Resolution search returned %d results (k=%d)", len(docs), k)
            return docs

        finally:
            session.close()

    def search_with_scores(
        self,
        query: str,
        k: int = 5,
        product_filter: Optional[str] = None,
    ) -> list[tuple[Document, float]]:
        """Return the *k* most similar resolutions with cosine distance scores."""
        docs = self.search(query, k=k, product_filter=product_filter)
        return [(doc, doc.metadata.get("distance", 0.0)) for doc in docs]

    # ── Stats ────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Return the total number of resolution embeddings stored."""
        session = SessionLocal()
        try:
            result = session.execute(
                text("SELECT COUNT(*) FROM resolution_embeddings")
            )
            return result.scalar() or 0
        finally:
            session.close()
