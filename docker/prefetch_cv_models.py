"""Prefetch CV model assets during Docker image build."""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

CACHE_ROOT = Path("/var/cache/did-backend-api")
HF_HOME = CACHE_ROOT / "huggingface"
DEFAULT_YOLO_DEST = CACHE_ROOT / "yolo11n.pt"
MIN_YOLO_BYTES = 1_000_000
YOLO11N_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"


def _extras() -> str:
    return os.environ.get("CV_EXTRAS", "").strip()


def _needs_clip() -> bool:
    return _extras() in {"cv-embedding", "cv-yolo", "cv"}


def _needs_yolo() -> bool:
    return _extras() in {"cv-yolo", "cv"}


def prefetch_clip() -> None:
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HOME / "hub")
    HF_HOME.mkdir(parents=True, exist_ok=True)

    import open_clip

    model_name = os.environ.get("CV_EMBEDDING_MODEL", "ViT-B-32")
    pretrained = os.environ.get("CV_EMBEDDING_PRETRAINED", "openai")
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()

    hub_cache = HF_HOME / "hub"
    cached_files = list(hub_cache.rglob("*")) if hub_cache.exists() else []
    if not cached_files:
        raise SystemExit(f"CLIP prefetch failed: no Hugging Face cache under {hub_cache}")


def prefetch_yolo() -> None:
    """Download YOLO weights directly (avoids importing cv2/ultralytics at build time)."""
    dest = Path(os.environ.get("CV_YOLO_MODEL", str(DEFAULT_YOLO_DEST)))
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading YOLO weights to {dest}...", file=sys.stderr)
    urllib.request.urlretrieve(YOLO11N_URL, dest)
    if dest.stat().st_size < MIN_YOLO_BYTES:
        raise SystemExit(f"YOLO weights look incomplete: {dest.stat().st_size} bytes")


def main() -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if not _needs_clip() and not _needs_yolo():
        print(f"Skipping CV prefetch for CV_EXTRAS={_extras()!r}", file=sys.stderr)
        return

    if _needs_yolo():
        print("Prefetching YOLO weights...", file=sys.stderr)
        prefetch_yolo()
    if _needs_clip():
        print("Prefetching CLIP weights...", file=sys.stderr)
        prefetch_clip()
    print("CV model prefetch complete", file=sys.stderr)


if __name__ == "__main__":
    main()
