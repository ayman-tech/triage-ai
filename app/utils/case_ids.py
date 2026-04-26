"""Public-facing case identifier helpers."""

from __future__ import annotations

import re

from sqlalchemy.orm import Session

from app.db.models import ComplaintCase

PUBLIC_CASE_ID_PREFIX = "CASE"
PUBLIC_CASE_ID_WIDTH = 5
_PUBLIC_CASE_ID_RE = re.compile(rf"^{PUBLIC_CASE_ID_PREFIX}(\d+)$")


def format_public_case_id(number: int) -> str:
    return f"{PUBLIC_CASE_ID_PREFIX}{number:0{PUBLIC_CASE_ID_WIDTH}d}"


def is_public_case_id(value: str | None) -> bool:
    if not value:
        return False
    return bool(_PUBLIC_CASE_ID_RE.fullmatch(str(value).strip().upper()))


def parse_public_case_id(value: str | None) -> int | None:
    if not value:
        return None
    match = _PUBLIC_CASE_ID_RE.fullmatch(str(value).strip().upper())
    if not match:
        return None
    return int(match.group(1))


def next_public_case_id(session: Session) -> str:
    rows = (
        session.query(ComplaintCase.public_case_id)
        .filter(ComplaintCase.public_case_id.isnot(None))
        .all()
    )
    max_number = 0
    for (value,) in rows:
        number = parse_public_case_id(value)
        if number is not None and number > max_number:
            max_number = number
    return format_public_case_id(max_number + 1)


def resolve_case_record(session: Session, identifier: str) -> ComplaintCase | None:
    query_value = (identifier or "").strip()
    if not query_value:
        return None

    row = session.query(ComplaintCase).filter(ComplaintCase.id == query_value).first()
    if row is not None:
        return row

    normalized = query_value.upper()
    return (
        session.query(ComplaintCase)
        .filter(ComplaintCase.public_case_id == normalized)
        .first()
    )


def ensure_case_public_id(session: Session, case: ComplaintCase) -> str:
    if case.public_case_id:
        return case.public_case_id
    public_id = next_public_case_id(session)
    case.public_case_id = public_id
    return public_id
