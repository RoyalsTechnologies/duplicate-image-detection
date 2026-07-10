import io
from pathlib import Path

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from app.config import settings
from app.exceptions import BadRequestError

MAX_IMAGE_BYTES = settings.max_upload_size_mb * 1024 * 1024
ALLOWED_IMAGE_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
ALLOWED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
# Groq base64 payloads are limited (~4 MB); keep raw JPEG under this before encoding.
MAX_VISION_API_IMAGE_BYTES = 2_800_000
MAX_VISION_IMAGE_DIMENSION = 1280


async def read_upload_image(image: UploadFile) -> tuple[bytes, str]:
    """Validate and read an uploaded image.

    Returns:
        A tuple of raw image bytes and a MIME type suitable for vision APIs.
    """
    extension = Path(image.filename or "").suffix.lower()
    content_type = image.content_type or ""
    if content_type not in ALLOWED_IMAGE_MIME_TYPES:
        raise BadRequestError("Uploaded file must be a JPEG, PNG, or WebP image")
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise BadRequestError("Uploaded image extension is not allowed")
    data = await image.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise BadRequestError("Image is required")
    if len(data) > MAX_IMAGE_BYTES:
        raise BadRequestError("Image exceeds configured upload size limit")
    try:
        with Image.open(io.BytesIO(data)) as parsed_image:
            parsed_image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise BadRequestError("Uploaded file is not a valid image") from exc
    return data, content_type


def prepare_image_for_vision_api(image_bytes: bytes, mime_type: str) -> tuple[bytes, str]:
    """Resize/compress an image so it fits Groq vision base64 payload limits."""
    if mime_type == "image/jpeg" and len(image_bytes) <= MAX_VISION_API_IMAGE_BYTES:
        return image_bytes, mime_type

    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb_image = image.convert("RGB")
        if max(rgb_image.size) > MAX_VISION_IMAGE_DIMENSION:
            rgb_image.thumbnail(
                (MAX_VISION_IMAGE_DIMENSION, MAX_VISION_IMAGE_DIMENSION),
                Image.Resampling.LANCZOS,
            )

        output = io.BytesIO()
        quality = 90
        while quality >= 50:
            output.seek(0)
            output.truncate(0)
            rgb_image.save(output, format="JPEG", quality=quality, optimize=True)
            if output.tell() <= MAX_VISION_API_IMAGE_BYTES:
                return output.getvalue(), "image/jpeg"
            quality -= 10

        return output.getvalue(), "image/jpeg"
