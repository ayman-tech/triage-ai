"""Persistence and processing helpers for uploaded complaint documents."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterable

from fastapi import HTTPException, UploadFile, status
from app.retrieval.document import Document
from PIL import Image, ImageFilter, ImageOps
from pypdf import PdfReader
from sqlalchemy import select

from app.db.models import CaseDocument, DocumentArtifact, DocumentEmbedding
from app.db.session import SessionLocal
from app.documents.storage import save_upload
from app.retrieval.embeddings import get_embeddings
from app.schemas.document import CaseDocumentRead, CaseDocumentSummary

logger = logging.getLogger(__name__)

_embeddings = None
_TESSERACT_BIN = shutil.which("tesseract")
_PDFTOPPM_BIN = shutil.which("pdftoppm")
_DOCUMENT_GATE_TIMEOUT_SECONDS = float(os.getenv("DOCUMENT_GATE_TIMEOUT_SECONDS", "60"))
_DOCUMENT_GATE_POLL_INTERVAL_SECONDS = float(os.getenv("DOCUMENT_GATE_POLL_INTERVAL_SECONDS", "1"))


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = get_embeddings()
    return _embeddings


def _document_type_for_name(name: str, mime_type: str) -> str:
    lower = name.lower()
    if "statement" in lower:
        return "statement"
    if "notice" in lower:
        return "notice"
    if "screen" in lower or mime_type.startswith("image/"):
        return "screenshot"
    if "letter" in lower or "correspondence" in lower:
        return "correspondence"
    return "unknown"


def _safe_json_load(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _extract_text_from_pdf(path: Path) -> tuple[str, list[dict]]:
    reader = PdfReader(str(path))
    pages: list[dict] = []
    texts: list[str] = []
    for index, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            texts.append(text)
        pages.append({"page": index + 1, "chars": len(text)})
    return "\n\n".join(texts).strip(), pages


def _prepare_image_for_ocr(path: Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("L")
    image = ImageOps.autocontrast(image)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    # Light thresholding tends to improve printed document OCR.
    image = image.point(lambda px: 255 if px > 180 else 0)
    if min(image.size) < 1200:
        image = image.resize((image.width * 2, image.height * 2))
    return image


def _run_tesseract_on_image(image: Image.Image) -> str:
    if not _TESSERACT_BIN:
        raise RuntimeError("tesseract is not installed or not available in PATH.")
    with tempfile.TemporaryDirectory(prefix="ocr-image-") as tmp_dir:
        tmp_path = Path(tmp_dir) / "page.png"
        image.save(tmp_path)
        proc = subprocess.run(
            [
                _TESSERACT_BIN,
                str(tmp_path),
                "stdout",
                "--psm",
                "6",
                "-l",
                "eng",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "tesseract OCR failed.")
        return proc.stdout.strip()


def _extract_text_from_image(path: Path) -> tuple[str, list[dict]]:
    processed = _prepare_image_for_ocr(path)
    text = _run_tesseract_on_image(processed)
    with Image.open(path) as img:
        width, height = img.size
    return text, [{"page": 1, "width": width, "height": height, "ocr_chars": len(text)}]


def _render_pdf_pages_for_ocr(path: Path) -> tuple[list[Path], tempfile.TemporaryDirectory[str]]:
    if not _PDFTOPPM_BIN:
        raise RuntimeError(
            "pdftoppm is not installed; scanned PDF OCR requires Poppler on the host."
        )
    tmp_dir = tempfile.TemporaryDirectory(prefix="ocr-pdf-")
    output_prefix = Path(tmp_dir.name) / "page"
    proc = subprocess.run(
        [
            _PDFTOPPM_BIN,
            "-png",
            "-r",
            "200",
            str(path),
            str(output_prefix),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        tmp_dir.cleanup()
        raise RuntimeError(proc.stderr.strip() or "pdftoppm failed to render PDF pages.")
    rendered = sorted(Path(tmp_dir.name).glob("page-*.png"))
    if not rendered:
        tmp_dir.cleanup()
        raise RuntimeError("No rasterized PDF pages were produced for OCR.")
    return rendered, tmp_dir


def _extract_text_from_scanned_pdf(path: Path) -> tuple[str, list[dict]]:
    rendered_pages, temp_handle = _render_pdf_pages_for_ocr(path)
    texts: list[str] = []
    pages: list[dict] = []
    try:
        for index, page_path in enumerate(rendered_pages):
            processed = _prepare_image_for_ocr(page_path)
            page_text = _run_tesseract_on_image(processed)
            texts.append(page_text)
            pages.append({"page": index + 1, "ocr_chars": len(page_text)})
        return "\n\n".join(part for part in texts if part.strip()).strip(), pages
    finally:
        temp_handle.cleanup()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _extract_facts(text: str) -> dict:
    amounts = list(dict.fromkeys(re.findall(r"(?:USD|\$)\s?\d[\d,]*(?:\.\d{2})?", text)))
    dates = list(
        dict.fromkeys(
            re.findall(
                r"(?:\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b)",
                text,
                re.IGNORECASE,
            )
        )
    )
    account_refs = list(dict.fromkeys(re.findall(r"(?:\*{2,}|\bending in\b)\s*\d{2,4}", text, re.IGNORECASE)))
    parties = list(dict.fromkeys(re.findall(r"\b[A-Z][A-Za-z&]+\s+(?:Bank|Credit|Services|Finance|Card|Loan)\b", text)))
    parties.extend(
        item for item in re.findall(r"\b(?:Dear\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text)
        if item and len(item.split()) <= 3
    )
    parties = list(dict.fromkeys(parties))
    reference_numbers = list(
        dict.fromkeys(
            re.findall(r"\b(?:Order Number|Reference|Ref|Account Number)\s*:\s*([A-Z0-9\-*]+)", text, re.IGNORECASE)
        )
    )
    signals: list[str] = []
    lower = text.lower()
    for needle, label in (
        ("unauthorized", "unauthorized_transaction"),
        ("fraud", "fraud_indicator"),
        ("fee", "fee_dispute"),
        ("refund", "refund_requested"),
        ("late", "timing_issue"),
        ("chargeback", "chargeback_context"),
    ):
        if needle in lower:
            signals.append(label)
    return {
        "amounts": amounts[:20],
        "dates": dates[:20],
        "account_refs": account_refs[:20],
        "parties": parties[:20],
        "reference_numbers": reference_numbers[:20],
        "signals": signals,
    }


def _chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> list[str]:
    norm = _normalize_text(text)
    if not norm:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(norm):
        end = min(len(norm), start + chunk_size)
        chunks.append(norm[start:end])
        if end == len(norm):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _to_read_model(doc: CaseDocument) -> CaseDocumentRead:
    artifact = doc.artifact
    extracted_facts = _safe_json_load(artifact.extracted_json if artifact else None)
    summary_text = None
    if artifact and artifact.normalized_text:
        summary_text = artifact.normalized_text[:300]
    return CaseDocumentRead(
        id=doc.id,
        case_id=doc.case_id,
        intake_session_id=doc.intake_session_id,
        user_id=doc.user_id,
        original_filename=doc.original_filename,
        mime_type=doc.mime_type,
        size_bytes=doc.size_bytes,
        upload_status=doc.upload_status,
        parser_status=doc.parser_status,
        extraction_status=doc.extraction_status,
        document_type=doc.document_type,
        processing_error=doc.processing_error,
        summary_text=summary_text,
        extracted_facts=extracted_facts,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


def create_session_document(*, session_id: str, user_id: str | None, file: UploadFile) -> CaseDocumentRead:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing filename.")
    saved = save_upload(scope_id=session_id, file=file)
    with SessionLocal() as db:
        row = CaseDocument(
            id=saved["document_id"],
            intake_session_id=session_id,
            user_id=user_id,
            original_filename=saved["original_filename"],
            mime_type=saved["mime_type"],
            size_bytes=saved["size_bytes"],
            storage_uri=saved["storage_uri"],
            checksum=saved["checksum"],
            document_type=_document_type_for_name(saved["original_filename"], saved["mime_type"]),
            upload_status="uploaded",
            parser_status="pending",
            extraction_status="pending",
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return _to_read_model(row)


def list_session_documents(session_id: str, user_id: str | None = None) -> list[CaseDocumentRead]:
    with SessionLocal() as db:
        query = db.query(CaseDocument).filter(CaseDocument.intake_session_id == session_id)
        if user_id:
            query = query.filter(CaseDocument.user_id == user_id)
        rows = query.order_by(CaseDocument.created_at.asc()).all()
        return [_to_read_model(row) for row in rows]


def delete_session_document(*, session_id: str, document_id: str, user_id: str | None = None) -> None:
    with SessionLocal() as db:
        query = db.query(CaseDocument).filter(
            CaseDocument.id == document_id,
            CaseDocument.intake_session_id == session_id,
        )
        if user_id:
            query = query.filter(CaseDocument.user_id == user_id)
        row = query.first()
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
        db.delete(row)
        db.commit()
    try:
        path = Path(row.storage_uri)
        if path.exists():
            path.unlink()
        parent = path.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    except Exception:
        logger.warning("Failed to cleanup uploaded file for document_id=%s", document_id, exc_info=True)


def link_session_documents_to_case(*, session_id: str, case_id: str, user_id: str | None = None) -> list[CaseDocumentRead]:
    with SessionLocal() as db:
        query = db.query(CaseDocument).filter(CaseDocument.intake_session_id == session_id)
        if user_id:
            query = query.filter(CaseDocument.user_id == user_id)
        rows = query.all()
        for row in rows:
            row.case_id = case_id
            row.user_id = user_id or row.user_id
            db.query(DocumentEmbedding).filter(DocumentEmbedding.document_id == row.id).update(
                {"case_id": case_id},
                synchronize_session=False,
            )
        db.commit()
        for row in rows:
            db.refresh(row)
        return [_to_read_model(row) for row in rows]


def list_case_documents(case_id: str) -> list[CaseDocumentRead]:
    with SessionLocal() as db:
        rows = (
            db.query(CaseDocument)
            .filter(CaseDocument.case_id == case_id)
            .order_by(CaseDocument.created_at.asc())
            .all()
        )
        return [_to_read_model(row) for row in rows]


def build_case_document_summary(case_id: str) -> CaseDocumentSummary:
    docs = list_case_documents(case_id)
    merged_facts = {
        "amounts": [],
        "dates": [],
        "account_refs": [],
        "parties": [],
        "reference_numbers": [],
        "signals": [],
    }
    for doc in docs:
        facts = doc.extracted_facts or {}
        for key in merged_facts:
            merged_facts[key].extend(facts.get(key, []))
    for key in merged_facts:
        merged_facts[key] = list(dict.fromkeys(merged_facts[key]))
    return CaseDocumentSummary(
        total_documents=len(docs),
        processed_documents=sum(1 for d in docs if d.extraction_status == "processed"),
        pending_documents=sum(1 for d in docs if d.extraction_status in ("pending", "processing")),
        failed_documents=sum(1 for d in docs if d.extraction_status == "failed"),
        facts=merged_facts,
    )


def wait_for_case_documents(case_id: str) -> dict:
    started = time.monotonic()
    initial_docs = list_case_documents(case_id)
    if not initial_docs:
        return {
            "required": False,
            "status": "not_required",
            "waited_seconds": 0.0,
            "total_documents": 0,
            "processed_documents": 0,
            "failed_documents": 0,
        }

    while True:
        docs = list_case_documents(case_id)
        processed = sum(1 for doc in docs if doc.extraction_status == "processed")
        failed = sum(1 for doc in docs if doc.extraction_status == "failed")
        pending = len(docs) - processed - failed
        waited = round(time.monotonic() - started, 2)

        if pending <= 0:
            status = "ready" if processed == len(docs) else "ready_with_failures"
            return {
                "required": True,
                "status": status,
                "waited_seconds": waited,
                "total_documents": len(docs),
                "processed_documents": processed,
                "failed_documents": failed,
            }

        if waited >= _DOCUMENT_GATE_TIMEOUT_SECONDS:
            return {
                "required": True,
                "status": "timed_out",
                "waited_seconds": waited,
                "total_documents": len(docs),
                "processed_documents": processed,
                "failed_documents": failed,
            }

        time.sleep(_DOCUMENT_GATE_POLL_INTERVAL_SECONDS)


def _narrative_facts(text: str) -> dict:
    normalized = _normalize_text(text)
    return _extract_facts(normalized)


def compare_case_to_documents(*, narrative_text: str, document_summary: dict | None) -> dict:
    summary = document_summary or {}
    doc_facts = summary.get("facts") or {}
    if not summary.get("total_documents"):
        return {
            "status": "not_applicable",
            "conflicts": [],
            "verified_facts": {},
        }
    if not summary.get("processed_documents"):
        return {
            "status": "document_evidence_unavailable",
            "conflicts": [],
            "verified_facts": {},
            "narrative_facts": _narrative_facts(narrative_text),
            "document_facts": doc_facts,
        }

    narrative_facts = _narrative_facts(narrative_text)
    conflicts: list[dict] = []
    verified_facts: dict[str, list[str]] = {}

    for key in ("amounts", "dates", "reference_numbers", "account_refs"):
        narrative_vals = [str(v).strip() for v in narrative_facts.get(key, []) if str(v).strip()]
        document_vals = [str(v).strip() for v in doc_facts.get(key, []) if str(v).strip()]
        if document_vals:
            verified_facts[key] = document_vals
        if narrative_vals and document_vals and not set(narrative_vals).intersection(document_vals):
            conflicts.append(
                {
                    "field": key,
                    "narrative": narrative_vals[:10],
                    "documents": document_vals[:10],
                    "reason": f"narrative_{key}_do_not_match_documents",
                }
            )

    narrative_signals = set(narrative_facts.get("signals", []))
    document_signals = set(doc_facts.get("signals", []))
    if narrative_signals and document_signals:
        verified_facts["signals"] = sorted(document_signals)
        if not narrative_signals.intersection(document_signals):
            conflicts.append(
                {
                    "field": "signals",
                    "narrative": sorted(narrative_signals),
                    "documents": sorted(document_signals),
                    "reason": "narrative_signals_do_not_match_documents",
                }
            )

    status = "contradiction" if conflicts else "aligned"
    return {
        "status": status,
        "conflicts": conflicts,
        "verified_facts": verified_facts,
        "narrative_facts": narrative_facts,
        "document_facts": doc_facts,
    }


def _upsert_artifact(*, document_id: str, raw_text: str, normalized_text: str, extracted: dict, confidence: float) -> None:
    with SessionLocal() as db:
        artifact = db.query(DocumentArtifact).filter(DocumentArtifact.document_id == document_id).first()
        if artifact is None:
            artifact = DocumentArtifact(document_id=document_id)
            db.add(artifact)
        artifact.raw_text = raw_text
        artifact.normalized_text = normalized_text
        artifact.extracted_json = json.dumps(extracted, ensure_ascii=False)
        artifact.parser_version = "v1"
        artifact.extraction_version = "v1"
        artifact.confidence = confidence
        db.commit()


def _replace_document_embeddings(*, document_id: str, case_id: str | None, chunks: Iterable[str]) -> None:
    chunk_list = [chunk for chunk in chunks if chunk.strip()]
    if not chunk_list:
        return
    embeddings = _get_embeddings().embed_documents(chunk_list)
    with SessionLocal() as db:
        db.query(DocumentEmbedding).filter(DocumentEmbedding.document_id == document_id).delete()
        rows = [
            DocumentEmbedding(
                document_id=document_id,
                case_id=case_id,
                chunk_index=index,
                content=chunk,
                embedding=vector,
            )
            for index, (chunk, vector) in enumerate(zip(chunk_list, embeddings))
        ]
        db.bulk_save_objects(rows)
        db.commit()


def process_document(document_id: str) -> None:
    with SessionLocal() as db:
        doc = db.query(CaseDocument).filter(CaseDocument.id == document_id).first()
        if doc is None:
            logger.warning("Document %s not found for processing", document_id)
            return
        doc.upload_status = "processing"
        doc.parser_status = "processing"
        doc.extraction_status = "processing"
        doc.processing_error = None
        db.commit()
        storage_uri = doc.storage_uri
        mime_type = doc.mime_type
        case_id = doc.case_id

    path = Path(storage_uri)
    try:
        raw_text = ""
        metadata: dict = {}
        if mime_type == "application/pdf" or path.suffix.lower() == ".pdf":
            raw_text, metadata["pages"] = _extract_text_from_pdf(path)
            if not raw_text:
                raw_text, metadata["ocr_pages"] = _extract_text_from_scanned_pdf(path)
                metadata["ocr_mode"] = "scanned_pdf"
        elif mime_type.startswith("image/") or path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            raw_text, metadata["images"] = _extract_text_from_image(path)
            metadata["ocr_mode"] = "image"
        else:
            raw_text = path.read_text(encoding="utf-8", errors="ignore")

        normalized = _normalize_text(raw_text)
        extracted = _extract_facts(normalized)
        extracted.update(metadata)
        confidence = 0.9 if normalized else 0.2
        _upsert_artifact(
            document_id=document_id,
            raw_text=raw_text,
            normalized_text=normalized,
            extracted=extracted,
            confidence=confidence,
        )
        _replace_document_embeddings(
            document_id=document_id,
            case_id=case_id,
            chunks=_chunk_text(normalized),
        )
        with SessionLocal() as db:
            doc = db.query(CaseDocument).filter(CaseDocument.id == document_id).first()
            if doc is not None:
                doc.upload_status = "processed"
                doc.parser_status = "processed" if normalized or mime_type == "application/pdf" else "processed"
                doc.extraction_status = "processed"
                doc.processing_error = None
                db.commit()
    except Exception as exc:
        logger.exception("Document processing failed for document_id=%s", document_id)
        with SessionLocal() as db:
            doc = db.query(CaseDocument).filter(CaseDocument.id == document_id).first()
            if doc is not None:
                doc.upload_status = "failed"
                doc.parser_status = "failed"
                doc.extraction_status = "failed"
                doc.processing_error = str(exc)
                db.commit()


def search_case_documents(*, case_id: str, query: str, k: int = 3) -> list[Document]:
    if not query.strip():
        return []
    query_vec = _get_embeddings().embed_query(query)
    with SessionLocal() as db:
        rows = db.execute(
            select(
                DocumentEmbedding.document_id,
                DocumentEmbedding.content,
                DocumentEmbedding.chunk_index,
                DocumentEmbedding.embedding.cosine_distance(query_vec).label("distance"),
            )
            .where(DocumentEmbedding.case_id == case_id)
            .order_by("distance")
            .limit(k)
        ).all()
    return [
        Document(
            page_content=row.content,
            metadata={
                "document_id": row.document_id,
                "chunk_index": row.chunk_index,
                "distance": float(row.distance),
            },
        )
        for row in rows
    ]
