"""Configure process env so baked CV model caches are used at runtime."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from app.computer_vision.constants import MIN_YOLO_WEIGHTS_BYTES
from app.config import settings

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.debug("Could not create CV cache directory %s", path)


def configure_cv_runtime_env() -> None:
    """Point Hugging Face / Ultralytics at the shared model cache directory."""
    cache_root = settings.cv_cache_dir
    hf_home = cache_root / "huggingface"
    hub_cache = hf_home / "hub"
    yolo_config = cache_root / "ultralytics"

    os.environ.setdefault("CV_CACHE_DIR", str(cache_root))
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config))

    for path in (cache_root, hf_home, hub_cache, yolo_config):
        _ensure_dir(path)

    if any(hub_cache.rglob("*.safetensors")):
        os.environ["HF_HUB_OFFLINE"] = "1"
        logger.info("CLIP weights found in %s; HF_HUB_OFFLINE=1", hub_cache)


def log_cv_cache_status() -> None:
    provider = settings.cv_provider.strip().lower()
    if provider == "local":
        return

    cache_root = settings.cv_cache_dir
    hf_hub = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", cache_root / "huggingface" / "hub"))
    yolo_path = Path(settings.cv_yolo_model)

    yolo_bytes = yolo_path.stat().st_size if yolo_path.is_file() else 0
    hf_files = list(hf_hub.rglob("*")) if hf_hub.is_dir() else []
    logger.info(
        "CV cache (%s): yolo=%s (%s bytes) hf_hub=%s (%s files) HF_HOME=%s",
        provider,
        yolo_path,
        yolo_bytes,
        hf_hub,
        len(hf_files),
        os.environ.get("HF_HOME"),
    )
    if provider in {"embedding", "yolov11"} and not hf_files:
        logger.warning(
            "CLIP cache is empty under %s; rebuild with DOCKER_BUILD_TARGET=runtime-cv "
            "(docker compose build api && docker compose up -d api)",
            hf_hub,
        )
    if provider == "yolov11" and yolo_bytes < MIN_YOLO_WEIGHTS_BYTES:
        logger.warning(
            "YOLO weights missing or incomplete at %s (%s bytes); expected >= %s",
            yolo_path,
            yolo_bytes,
            MIN_YOLO_WEIGHTS_BYTES,
        )
