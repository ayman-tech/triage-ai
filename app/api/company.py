"""API endpoints for managing the dynamic company profile and taxonomy overrides."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.knowledge.company_store import load_company_profile, save_company_profile
from app.knowledge.taxonomy_store import TAXONOMY_TYPES, load_taxonomy, save_taxonomy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/company", tags=["company"])


class CompanyProfileBody(BaseModel):
    display_name: str = ""
    customer_identity: str = ""
    supported_products: list[str] = []
    intake_operator_style: str = ""
    intake_routing_guidance: list[str] = []
    safe_reference_guidance: str = ""


@router.get("/profile")
def get_company_profile() -> dict[str, Any]:
    return load_company_profile()


@router.put("/profile")
def update_company_profile(body: CompanyProfileBody) -> dict[str, Any]:
    profile = body.model_dump()
    save_company_profile(profile)
    return profile


@router.post("/profile/upload")
async def upload_company_profile(file: UploadFile = File(...)) -> dict[str, Any]:
    """Extract company profile from an uploaded PDF or image using Gemini."""
    mime = file.content_type or ""
    suffix = Path(file.filename or "upload.bin").suffix.lower()

    with tempfile.TemporaryDirectory(prefix="company-upload-") as tmp_dir:
        tmp_path = Path(tmp_dir) / f"upload{suffix}"
        content = await file.read()
        tmp_path.write_bytes(content)

        try:
            if mime == "application/pdf" or suffix == ".pdf":
                from app.documents.service import _extract_text_from_pdf
                text, _ = _extract_text_from_pdf(tmp_path)
            elif mime.startswith("image/") or suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}:
                from app.documents.service import _extract_text_from_image
                text, _ = _extract_text_from_image(tmp_path)
            else:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"Unsupported file type: {mime or suffix}. Upload a PDF or image.",
                )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Text extraction failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Could not extract text from file: {exc}",
            )

    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text could be extracted from the uploaded file.",
        )

    profile = _parse_profile_with_gemini(text)
    save_company_profile(profile)
    return profile


_EXTRACTION_PROMPT = """\
You are a configuration assistant. Extract structured company profile information from the document text below.

Return ONLY a valid JSON object with exactly these keys (no extra text, no markdown):
{{
  "display_name": "<company name>",
  "customer_identity": "<short phrase describing the type of business, e.g. 'digital bank'>",
  "supported_products": ["<product 1>", "<product 2>"],
  "intake_operator_style": "<instructions for how the complaints intake agent should speak on behalf of this company>",
  "intake_routing_guidance": ["<guidance item 1>", "<guidance item 2>"],
  "safe_reference_guidance": "<instructions for safely collecting account references from customers>"
}}

If a field cannot be determined from the document, use a sensible empty value ("" or []).

Document text:
{text}
"""


def _parse_profile_with_gemini(text: str) -> dict:
    try:
        from app.agents.llm_factory import default_model_name, get_gemini_client
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM client unavailable: {exc}",
        )

    client = get_gemini_client()
    prompt = _EXTRACTION_PROMPT.format(text=text[:8000])

    try:
        response = client.models.generate_content(
            model=default_model_name(),
            contents=prompt,
        )
        raw = response.text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        profile = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Gemini returned non-JSON response: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not parse company profile from document — Gemini returned unexpected output.",
        )
    except Exception as exc:
        logger.error("Gemini call failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM extraction failed: {exc}",
        )

    # Ensure all expected keys exist with correct types
    defaults: dict[str, Any] = {
        "display_name": "",
        "customer_identity": "",
        "supported_products": [],
        "intake_operator_style": "",
        "intake_routing_guidance": [],
        "safe_reference_guidance": "",
    }
    for key, default in defaults.items():
        if key not in profile:
            profile[key] = default
        if isinstance(default, list) and not isinstance(profile[key], list):
            profile[key] = [profile[key]] if profile[key] else []

    return profile


# ── Taxonomy Upload ───────────────────────────────────────────────────────────

_TAXONOMY_PROMPTS: dict[str, str] = {
    "product_categories": """\
Extract a product category taxonomy from the document below.

Return ONLY a valid JSON array (no markdown, no extra text) where each element has:
  {{"product_category": "<snake_case_id>", "definition": "<one sentence>", "cues": ["<word>", ...]}}

Document:
{text}
""",
    "issue_types": """\
Extract an issue type taxonomy from the document below.

Return ONLY a valid JSON array (no markdown, no extra text) where each element has:
  {{"issue_type": "<snake_case_id>", "definition": "<one sentence>", "cues": ["<word>", ...]}}

Document:
{text}
""",
    "severity_rubric": """\
Extract a severity rubric from the document below.

Return ONLY a valid JSON array (no markdown, no extra text) where each element has:
  {{"level": "<low|medium|high|critical>", "description": "<text>", "cues": ["<word>", ...], "escalation": <true|false>}}

Document:
{text}
""",
    "policy_snippets": """\
Extract compliance policy snippets from the document below.

Return ONLY a valid JSON array (no markdown, no extra text) where each element has:
  {{"policy_id": "<snake_case_id>", "description": "<text>", "cues": ["<word>", ...]}}

Document:
{text}
""",
    "routing_matrix": """\
Extract a routing matrix from the document below.

Return ONLY a valid JSON object (no markdown, no extra text) with this structure:
{{
  "team_by_product_category": {{"<product_category>": "<team_name>", ...}},
  "executive_team": "<team_name>",
  "management_escalation_team": "<team_name>"
}}

Document:
{text}
""",
    "root_cause_controls": """\
Extract root cause control entries from the document below.

Return ONLY a valid JSON array (no markdown, no extra text) where each element has:
  {{
    "root_cause_category": "<human readable label>",
    "root_cause_code": "<snake_case_id>",
    "description": "<text>",
    "business_summary": "<text>",
    "cues": ["<word>", ...],
    "controls_to_check": ["<control name>", ...]
  }}

Document:
{text}
""",
}


@router.post("/taxonomy/upload")
async def upload_taxonomy(
    taxonomy_type: str = Form(...),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Upload a document to override a taxonomy section. Gemini extracts the structured data."""
    if taxonomy_type not in TAXONOMY_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown taxonomy_type '{taxonomy_type}'. Valid: {TAXONOMY_TYPES}",
        )

    mime = file.content_type or ""
    suffix = Path(file.filename or "upload.bin").suffix.lower()

    with tempfile.TemporaryDirectory(prefix="taxonomy-upload-") as tmp_dir:
        tmp_path = Path(tmp_dir) / f"upload{suffix}"
        content = await file.read()
        tmp_path.write_bytes(content)

        try:
            if mime == "application/pdf" or suffix == ".pdf":
                from app.documents.service import _extract_text_from_pdf
                text, _ = _extract_text_from_pdf(tmp_path)
            elif mime.startswith("image/") or suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}:
                from app.documents.service import _extract_text_from_image
                text, _ = _extract_text_from_image(tmp_path)
            elif suffix in {".txt", ".md"}:
                text = tmp_path.read_text(errors="replace")
            else:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"Unsupported file type: {mime or suffix}. Upload PDF, image, or text.",
                )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Text extraction failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Could not extract text: {exc}",
            )

    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text could be extracted from the uploaded file.",
        )

    taxonomy_data = _extract_taxonomy_with_gemini(taxonomy_type, text)
    save_taxonomy(taxonomy_type, taxonomy_data)
    return {"taxonomy_type": taxonomy_type, "data": taxonomy_data}


def _extract_taxonomy_with_gemini(taxonomy_type: str, text: str) -> dict | list:
    try:
        from app.agents.llm_factory import default_model_name, get_gemini_client
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM client unavailable: {exc}",
        )

    client = get_gemini_client()
    prompt_template = _TAXONOMY_PROMPTS[taxonomy_type]
    prompt = prompt_template.format(text=text[:10000])

    try:
        response = client.models.generate_content(
            model=default_model_name(),
            contents=prompt,
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Gemini returned non-JSON for %s: %s", taxonomy_type, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not parse taxonomy from document — Gemini returned unexpected output.",
        )
    except Exception as exc:
        logger.error("Gemini call failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM extraction failed: {exc}",
        )
