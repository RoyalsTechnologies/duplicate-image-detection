"""Configuration for the automated YOLOv11 training pipeline.

Class names MUST match values in `app.models.ReportCategory` so the API can map
detections to report categories via `app/computer_vision/labels.py`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent
RAW_IMAGES_DIR = ML_ROOT / "raw_images"
FLAT_IMAGES_DIR = ML_ROOT / "raw_images_flat"
DATASET_DIR = ML_ROOT / "automated_dataset"
RUNS_DIR = ML_ROOT / "runs"
DEPLOY_DIR = ML_ROOT / "deployment"

# Image-classification pipeline (robust alternative to noisy auto-labeled
# detection). Each scraped query folder already implies its class, so no
# per-image annotation is needed; a "none" negative class lets the model
# reject uploads that are not a concern.
NEGATIVE_IMAGES_DIR = ML_ROOT / "raw_images_negative"
CLS_DATASET_DIR = ML_ROOT / "cls_dataset"
CLS_RUNS_DIR = ML_ROOT / "cls_runs"
CLS_BASE_MODEL = "yolo11n-cls.pt"
CLS_IMAGE_SIZE = 224
CLS_TRAIN_EPOCHS = 40
CLS_VAL_SPLIT = 0.15
NONE_CLASS_NAME = "none"
NEGATIVE_IMAGES_PER_QUERY = 40

# Keep all model/tokenizer downloads inside the ml/ tree so runs work in
# sandboxed or shared environments where the default system caches
# (e.g. /var/cache) are not writable.
CACHE_DIR = ML_ROOT / ".cache"


def _configure_caches() -> None:
    """Point HuggingFace, Torch, and Ultralytics caches at a writable dir."""
    hf_home = CACHE_DIR / "huggingface"
    for key, value in {
        "HF_HOME": str(hf_home),
        "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
        "TRANSFORMERS_CACHE": str(hf_home / "hub"),
        "TORCH_HOME": str(CACHE_DIR / "torch"),
        "XDG_CACHE_HOME": str(CACHE_DIR),
        "YOLO_CONFIG_DIR": str(CACHE_DIR / "ultralytics"),
    }.items():
        os.environ.setdefault(key, value)
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


_configure_caches()

BASE_MODEL = "yolo11n.pt"
IMAGES_PER_QUERY = 50
TRAIN_EPOCHS = 30
IMAGE_SIZE = 640

# Mirrors app.models.ReportCategory (excluding "other"). Keep in sync.
VALID_CLASS_NAMES = frozenset(
    {
        "refuse_dump",
        "blocked_drain",
        "flooding",
        "pothole",
        "pollution",
        "broken_public_facility",
        "sanitation",
    }
)


@dataclass(frozen=True)
class ConcernTarget:
    """A single concern type to scrape, auto-label, and train on.

    Attributes:
        search_query: Query sent to the image scraper.
        caption: Prompt GroundedSAM uses to detect and annotate the object.
        class_name: YOLO class label; must exist in `VALID_CLASS_NAMES`.
    """

    search_query: str
    caption: str
    class_name: str


TARGETS: tuple[ConcernTarget, ...] = (
    ConcernTarget("urban street flooding hazard", "flood water on the street", "flooding"),
    ConcernTarget("flooded road standing water", "flood water on the street", "flooding"),
    ConcernTarget("pothole asphalt road damage", "pothole in the road", "pothole"),
    ConcernTarget("cracked damaged road surface", "pothole in the road", "pothole"),
    ConcernTarget("garbage pile on street", "pile of garbage or rubbish", "refuse_dump"),
    ConcernTarget("illegal dumping trash heap", "pile of garbage or rubbish", "refuse_dump"),
    ConcernTarget("overflowing waste bin", "overflowing garbage bin", "refuse_dump"),
    ConcernTarget("blocked drainage gutter waste", "blocked drain or gutter", "blocked_drain"),
    ConcernTarget("open sewage drain", "blocked drain or gutter", "blocked_drain"),
    ConcernTarget("factory smoke air pollution", "smoke or air pollution", "pollution"),
    ConcernTarget("burning waste smoke", "smoke or air pollution", "pollution"),
    ConcernTarget("broken street light pole", "broken public infrastructure", "broken_public_facility"),
    ConcernTarget("fallen electricity pole", "broken public infrastructure", "broken_public_facility"),
    ConcernTarget("dirty public toilet sanitation", "dirty unsanitary public area", "sanitation"),
    ConcernTarget("open defecation dirty area", "dirty unsanitary public area", "sanitation"),
)


def build_ontology() -> dict[str, str]:
    """Return the caption-to-class mapping consumed by GroundedSAM.

    Each class name maps to exactly one caption. autodistill emits one dataset
    class slot per ontology entry, so allowing several captions to share a class
    name would duplicate that class in `data.yaml` (inflating `nc` and splitting
    detections across duplicate class IDs). The first caption seen for each class
    wins; other captions for the same class are dropped.

    Returns:
        A dict mapping one detection caption to each unique YOLO class name.

    Raises:
        ValueError: If any target class name is not in `VALID_CLASS_NAMES`.
    """
    invalid = sorted({t.class_name for t in TARGETS} - VALID_CLASS_NAMES)
    if invalid:
        raise ValueError(f"Unknown class names (must match ReportCategory): {invalid}")

    ontology: dict[str, str] = {}
    seen_classes: set[str] = set()
    for target in TARGETS:
        if target.class_name in seen_classes:
            continue
        seen_classes.add(target.class_name)
        ontology[target.caption] = target.class_name
    return ontology


# Generic, clearly-non-concern queries used to build the "none" negative class
# so the classifier can reject uploads that are not an environmental/public
# concern (selfies, food, indoor scenes, everyday objects, etc.).
NEGATIVE_QUERIES: tuple[str, ...] = (
    "person selfie portrait",
    "food meal on plate",
    "office desk laptop",
    "living room interior",
    "cat and dog pets",
    "car parked on clean street",
    "product photo white background",
    "landscape mountains nature",
    "statistics infographic chart",
    "educational poster diagram",
)


def query_to_class() -> dict[str, str]:
    """Map each scraped query folder name to its classification class.

    Returns:
        A dict from `search_query` (the `raw_images` subfolder name) to class.
    """
    return {target.search_query: target.class_name for target in TARGETS}
