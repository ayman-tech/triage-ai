"""Whether Postgres-backed vector RAG is available for this process."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_checked = False
_available = False


def vector_db_available() -> bool:
    """Return True if we can open a SQL session to the configured database.

    When Postgres is not running (typical local notebook/script runs), callers
    should skip ``ComplaintIndex`` / ``ResolutionIndex`` so agents still run
    without RAG context.

    Set ``DISABLE_VECTOR_DB=1`` to force off, or ``FORCE_VECTOR_DB=1`` to skip
    the probe and always use indices (fails fast if DB is down).
    """

    global _checked, _available

    if os.getenv("DISABLE_VECTOR_DB", "").lower() in ("1", "true", "yes"):
        return False
    if os.getenv("FORCE_VECTOR_DB", "").lower() in ("1", "true", "yes"):
        return True
    if _checked:
        return _available

    _checked = True
    try:
        from sqlalchemy import text

        from app.db.session import engine

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        _available = True
    except Exception as exc:
        logger.warning(
            "Postgres unreachable (%s: %s); vector RAG disabled for this run.",
            type(exc).__name__,
            exc,
        )
        _available = False
    return _available
