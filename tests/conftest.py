import io
import os

import pytest
from PIL import Image

_POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "3060")
os.environ.setdefault(
    "DATABASE_URL",
    f"postgresql+asyncpg://did:did@localhost:{_POSTGRES_PORT}/did",
)
os.environ.setdefault("IP_WHITELIST_ENABLED", "true")
os.environ.setdefault("ALLOWED_IPS", "127.0.0.1,::1")
os.environ.setdefault("TRUSTED_PROXY_IPS", "10.0.0.1")
os.environ.setdefault("REDIS_URL", "")
os.environ.setdefault("CV_PROVIDER", "local")
os.environ["LOCAL_STORAGE_DIR"] = "./test-uploads"


@pytest.fixture
def image_bytes():
    def _build(
        color: tuple[int, int, int] = (64, 64, 64), size: tuple[int, int] = (64, 64)
    ) -> bytes:
        image = Image.new("RGB", size, color)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    return _build


@pytest.fixture
def local_cv_client():
    from app.computer_vision import LocalComputerVisionClient

    return LocalComputerVisionClient()
