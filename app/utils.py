import hashlib
import re
from datetime import UTC, datetime

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utcnow() -> datetime:
    return datetime.now(UTC)


def sanitize_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = _CONTROL_CHARS.sub("", value).strip()
    return " ".join(cleaned.split())
