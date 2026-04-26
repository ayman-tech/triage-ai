"""Persistence layer for the dynamic company profile stored in PostgreSQL."""

from __future__ import annotations

import json
import logging
from datetime import datetime

from app.db.session import SessionLocal

logger = logging.getLogger(__name__)

_MOCK_BANK_ID = "mock_bank"


def load_company_profile(company_id: str = _MOCK_BANK_ID) -> dict:
    """Return the saved company profile, falling back to the hardcoded default."""
    from app.db.models import CompanyProfile

    session = SessionLocal()
    try:
        row = session.query(CompanyProfile).filter(CompanyProfile.company_id == company_id).first()
        if row is not None:
            return json.loads(row.profile_json)
    except Exception as exc:
        logger.warning("Could not read company profile from DB: %s", exc)
    finally:
        session.close()

    from app.knowledge.mock_company_pack import COMPANY_PROFILE
    return dict(COMPANY_PROFILE)


def save_company_profile(profile: dict, company_id: str = _MOCK_BANK_ID) -> None:
    """Upsert a company profile into the database."""
    from app.db.models import CompanyProfile
    import uuid

    session = SessionLocal()
    try:
        row = session.query(CompanyProfile).filter(CompanyProfile.company_id == company_id).first()
        profile_json = json.dumps(profile)
        if row is None:
            session.add(CompanyProfile(
                id=uuid.uuid4().hex,
                company_id=company_id,
                profile_json=profile_json,
                updated_at=datetime.utcnow(),
            ))
        else:
            row.profile_json = profile_json
            row.updated_at = datetime.utcnow()
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
