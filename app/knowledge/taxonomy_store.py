"""Per-company taxonomy store — DB-backed with fallback to mock_company_pack defaults."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

TAXONOMY_TYPES = [
    "product_categories",
    "issue_types",
    "severity_rubric",
    "policy_snippets",
    "routing_matrix",
    "root_cause_controls",
]


def _default_taxonomy(taxonomy_type: str) -> dict | list:
    from app.knowledge.mock_company_pack import (
        PRODUCT_CATEGORIES,
        ISSUE_TYPES,
        SEVERITY_RUBRIC,
        POLICY_SNIPPETS,
        ROUTING_MATRIX,
        ROOT_CAUSE_CONTROLS,
    )
    defaults = {
        "product_categories": PRODUCT_CATEGORIES,
        "issue_types": ISSUE_TYPES,
        "severity_rubric": SEVERITY_RUBRIC,
        "policy_snippets": POLICY_SNIPPETS,
        "routing_matrix": ROUTING_MATRIX,
        "root_cause_controls": ROOT_CAUSE_CONTROLS,
    }
    return defaults.get(taxonomy_type, [])


def load_taxonomy(taxonomy_type: str, company_id: str = "mock_bank") -> dict | list:
    try:
        from app.db.session import SessionLocal
        from app.db.models import CompanyTaxonomy

        session = SessionLocal()
        try:
            row = (
                session.query(CompanyTaxonomy)
                .filter(
                    CompanyTaxonomy.company_id == company_id,
                    CompanyTaxonomy.taxonomy_type == taxonomy_type,
                )
                .first()
            )
            if row:
                return json.loads(row.taxonomy_json)
        finally:
            session.close()
    except Exception as exc:
        logger.warning("load_taxonomy DB error (%s): %s", taxonomy_type, exc)

    return _default_taxonomy(taxonomy_type)


def save_taxonomy(taxonomy_type: str, data: dict | list, company_id: str = "mock_bank") -> None:
    from app.db.session import SessionLocal
    from app.db.models import CompanyTaxonomy
    import uuid

    serialized = json.dumps(data, ensure_ascii=False)
    session = SessionLocal()
    try:
        row = (
            session.query(CompanyTaxonomy)
            .filter(
                CompanyTaxonomy.company_id == company_id,
                CompanyTaxonomy.taxonomy_type == taxonomy_type,
            )
            .first()
        )
        if row:
            row.taxonomy_json = serialized
        else:
            session.add(
                CompanyTaxonomy(
                    id=uuid.uuid4().hex,
                    company_id=company_id,
                    taxonomy_type=taxonomy_type,
                    taxonomy_json=serialized,
                )
            )
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
