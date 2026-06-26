import io
import logging
import time

from PIL import Image

from app.computer_vision.runtime_env import configure_cv_runtime_env, log_cv_cache_status
from app.computer_vision.base import BaseComputerVisionClient
from app.computer_vision.embedding import EmbeddingComputerVisionClient
from app.computer_vision.local import LocalComputerVisionClient
from app.computer_vision.yolo import YoloV11ComputerVisionClient
from app.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_CV_PROVIDERS = frozenset({"local", "embedding", "yolov11"})


def build_cv_client() -> BaseComputerVisionClient:
    provider = settings.cv_provider.strip().lower()
    if provider not in SUPPORTED_CV_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_CV_PROVIDERS))
        raise ValueError(
            f"Unsupported CV provider {settings.cv_provider!r}. Expected one of: {supported}"
        )

    if provider == "embedding":
        return EmbeddingComputerVisionClient(
            model_name=settings.cv_embedding_model,
            pretrained=settings.cv_embedding_pretrained,
            device=settings.cv_device,
        )
    if provider == "yolov11":
        return YoloV11ComputerVisionClient(
            model_path=settings.cv_yolo_model,
            confidence=settings.cv_yolo_confidence,
            model_name=settings.cv_embedding_model,
            pretrained=settings.cv_embedding_pretrained,
            device=settings.cv_device,
        )
    return LocalComputerVisionClient()


def validate_cv_provider_dependencies() -> None:
    provider = settings.cv_provider.strip().lower()
    if provider == "local":
        return

    import importlib.util
    from pathlib import Path

    if provider in {"embedding", "yolov11"}:
        venv_python = Path("/venv/bin/python")
        if not venv_python.is_file():
            raise RuntimeError(
                f"CV_PROVIDER={settings.cv_provider!r} requires the runtime-cv Docker image "
                f"(missing {venv_python}). Rebuild with: "
                "COMPOSE_PARALLEL_LIMIT=1 docker compose build api && docker compose up -d api"
            )

    missing: list[str] = []
    if provider in {"embedding", "yolov11"} and importlib.util.find_spec("torch") is None:
        missing.append("torch")
    if provider in {"embedding", "yolov11"} and importlib.util.find_spec("open_clip") is None:
        missing.append("open-clip-torch")
    if provider == "yolov11":
        if importlib.util.find_spec("ultralytics") is None:
            missing.append("ultralytics")
        else:
            try:
                import cv2  # noqa: F401
            except ImportError as exc:
                missing.append(f"opencv system libraries ({exc})")

    if not missing:
        return

    extra = "cv-embedding" if provider == "embedding" else "cv-yolo"
    raise RuntimeError(
        f"CV_PROVIDER={settings.cv_provider!r} requires {', '.join(missing)}. "
        f'Install with: pip install -e ".[{extra}]"'
    )


def _warmup_image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(buffer, format="JPEG")
    return buffer.getvalue()


def warm_up_cv_client() -> None:
    """Load neural models at startup so the first report submit is not slow."""
    provider = settings.cv_provider.strip().lower()
    if provider == "local":
        return

    log_cv_cache_status()
    started = time.perf_counter()
    try:
        cv_client.analyze(_warmup_image_bytes())
    except Exception:
        logger.exception(
            "CV warmup failed for %s; the API will still start but the first submit may be slower",
            provider,
        )
        return
    elapsed = time.perf_counter() - started
    logger.info("CV models warmed up for %s in %.1fs", provider, elapsed)


configure_cv_runtime_env()
cv_client = build_cv_client()
