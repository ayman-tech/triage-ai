"""Storage abstraction for uploaded complaint documents."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile


UPLOAD_ROOT = Path(os.getenv("UPLOAD_ROOT", "app_data/uploads")).resolve()


def ensure_upload_root() -> Path:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    return UPLOAD_ROOT


def save_upload(*, scope_id: str, file: UploadFile) -> dict:
    """Persist an uploaded file to local storage and return metadata."""
    ensure_upload_root()
    document_id = uuid4().hex
    original_name = Path(file.filename or "upload.bin").name
    suffix = Path(original_name).suffix
    target_dir = UPLOAD_ROOT / scope_id / document_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"original{suffix}"

    hasher = hashlib.sha256()
    size = 0
    with target_path.open("wb") as out:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            hasher.update(chunk)
            size += len(chunk)
    file.file.seek(0)

    return {
        "document_id": document_id,
        "storage_uri": str(target_path),
        "size_bytes": size,
        "checksum": hasher.hexdigest(),
        "original_filename": original_name,
        "mime_type": file.content_type or "application/octet-stream",
    }

