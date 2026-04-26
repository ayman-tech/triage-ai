"""Optional Google Document AI adapter for complaint attachments."""

from __future__ import annotations

import os
from pathlib import Path


def document_ai_configured() -> bool:
    return bool(
        os.getenv("GOOGLE_CLOUD_PROJECT")
        and os.getenv("DOCUMENT_AI_LOCATION")
        and os.getenv("DOCUMENT_AI_PROCESSOR_ID")
    )


def process_document_with_document_ai(path: str | Path, mime_type: str) -> dict:
    """Extract text/entities from a local document using Google Document AI.

    This adapter is intentionally optional. The existing local PDF/OCR path
    remains the default unless callers explicitly route documents here.
    """
    if not document_ai_configured():
        raise RuntimeError(
            "Document AI is not configured. Set GOOGLE_CLOUD_PROJECT, "
            "DOCUMENT_AI_LOCATION, and DOCUMENT_AI_PROCESSOR_ID."
        )

    try:
        from google.cloud import documentai
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install google-cloud-documentai to use Document AI.") from exc

    project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ["DOCUMENT_AI_LOCATION"]
    processor_id = os.environ["DOCUMENT_AI_PROCESSOR_ID"]
    client = documentai.DocumentProcessorServiceClient()
    name = client.processor_path(project_id, location, processor_id)

    content = Path(path).read_bytes()
    request = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(content=content, mime_type=mime_type),
    )
    result = client.process_document(request=request)
    doc = result.document
    return {
        "text": doc.text or "",
        "entities": [
            {
                "type": entity.type_,
                "mention_text": entity.mention_text,
                "confidence": entity.confidence,
            }
            for entity in doc.entities
        ],
        "processor": name,
    }
