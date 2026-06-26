import logging
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4
from fastapi import UploadFile
from app.config import settings

logger = logging.getLogger(__name__)


class LocalFileStorage:
    async def save_upload(self, upload: UploadFile, content: bytes) -> str:
        suffix = Path(upload.filename or "image.jpg").suffix.lower() or ".jpg"
        filename = f"{uuid4().hex}{suffix}"
        settings.local_storage_dir.mkdir(parents=True, exist_ok=True)
        path = settings.local_storage_dir / filename
        path.write_bytes(content)
        return f"{settings.public_base_url.rstrip('/')}/uploads/{filename}"

    async def delete_url(self, url: str) -> None:
        """Best-effort cleanup for locally stored uploads.

        The API never exposes internal paths; cleanup derives the managed filename from the public URL.
        """
        public_upload_prefix = f"{settings.public_base_url.rstrip('/')}/uploads/"
        if not url.startswith(public_upload_prefix):
            return
        filename = Path(urlparse(url).path).name
        if not filename:
            return
        path = settings.local_storage_dir / filename
        try:
            path.unlink(missing_ok=True)
        except Exception:
            logger.warning(
                "Failed to clean up uploaded object", extra={"filename": filename}, exc_info=True
            )


storage = LocalFileStorage()
