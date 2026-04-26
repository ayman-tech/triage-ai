"""Schemas for case- and intake-linked uploaded documents."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class DocumentFactSet(BaseModel):
    amounts: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    account_refs: list[str] = Field(default_factory=list)
    parties: list[str] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)


class CaseDocumentRead(BaseModel):
    id: str
    case_id: Optional[str] = None
    intake_session_id: Optional[str] = None
    user_id: Optional[str] = None
    original_filename: str
    mime_type: str
    size_bytes: int
    upload_status: str
    parser_status: str
    extraction_status: str
    document_type: str
    processing_error: Optional[str] = None
    summary_text: Optional[str] = None
    extracted_facts: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class CaseDocumentSummary(BaseModel):
    total_documents: int = 0
    processed_documents: int = 0
    pending_documents: int = 0
    failed_documents: int = 0
    facts: dict[str, Any] = Field(default_factory=dict)

