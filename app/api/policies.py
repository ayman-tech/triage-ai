"""API endpoints for company policy document management."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.knowledge.policy_service import (
    PRODUCT_DISPLAY_NAMES,
    TOPIC_DISPLAY_NAMES,
    delete_policy,
    ingest_policy,
    list_policies,
    search_policies,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/policies", tags=["policies"])


@router.get("")
def get_policies(company_id: str = "mock_bank") -> dict[str, Any]:
    """Return all policies grouped by product."""
    return list_policies(company_id)


@router.get("/meta")
def get_policy_meta() -> dict[str, Any]:
    """Return available products and topics for UI selectors."""
    return {
        "products": [
            {"key": k, "display": v} for k, v in PRODUCT_DISPLAY_NAMES.items() if k != "other"
        ],
        "topics": [
            {"key": k, "display": v} for k, v in TOPIC_DISPLAY_NAMES.items()
        ],
    }


@router.post("/upload")
async def upload_policy(
    product: str = Form(...),
    topic: str = Form(...),
    company_id: str = Form(default="mock_bank"),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Upload a policy document (PDF or image) for a specific product and topic."""
    if product not in PRODUCT_DISPLAY_NAMES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown product: {product}")
    if topic not in TOPIC_DISPLAY_NAMES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown topic: {topic}")

    mime = file.content_type or ""
    suffix = Path(file.filename or "upload.bin").suffix.lower()

    with tempfile.TemporaryDirectory(prefix="policy-upload-") as tmp_dir:
        tmp_path = Path(tmp_dir) / f"upload{suffix}"
        content = await file.read()
        tmp_path.write_bytes(content)

        text = _extract_text(tmp_path, mime, suffix)

    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text could be extracted from the uploaded file.",
        )

    title = Path(file.filename or "policy").stem.replace("_", " ").replace("-", " ").title()

    entry = ingest_policy(
        company_id=company_id,
        product=product,
        topic=topic,
        title=title,
        content_text=text,
        original_filename=file.filename or "upload",
    )
    return entry


@router.delete("/{entry_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_policy(entry_id: str) -> None:
    """Delete a policy entry and its embeddings."""
    found = delete_policy(entry_id)
    if not found:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Policy not found.")


@router.get("/search")
def search(
    q: str,
    product: str | None = None,
    topic: str | None = None,
    company_id: str = "mock_bank",
    k: int = 5,
) -> list[dict[str, Any]]:
    """Semantic search across uploaded policy documents."""
    results = search_policies(q, product=product, topic=topic, company_id=company_id, k=k)
    return [{"content": doc.page_content, **doc.metadata} for doc in results]


def _extract_text(path: Path, mime: str, suffix: str) -> str:
    if mime == "application/pdf" or suffix == ".pdf":
        from app.documents.service import _extract_text_from_pdf
        text, _ = _extract_text_from_pdf(path)
    elif mime.startswith("image/") or suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}:
        from app.documents.service import _extract_text_from_image
        text, _ = _extract_text_from_image(path)
    elif suffix in {".txt", ".md"}:
        text = path.read_text(errors="ignore")
    else:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {mime or suffix}. Upload a PDF, image, or text file.",
        )
    return text
