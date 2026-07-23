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
# detection). Scraped Bing query folders are noisy; organize-clean filters
# with CLIP and writes verified images into class-named folders.
NEGATIVE_IMAGES_DIR = ML_ROOT / "raw_images_negative"
CLEAN_IMAGES_DIR = ML_ROOT / "raw_images_clean"
REJECTED_IMAGES_DIR = ML_ROOT / "raw_images_rejected"
CLS_DATASET_DIR = ML_ROOT / "cls_dataset"
CLS_RUNS_DIR = ML_ROOT / "cls_runs"
CLS_BASE_MODEL = "yolo11n-cls.pt"
CLS_IMAGE_SIZE = 224
CLS_TRAIN_EPOCHS = 40
CLS_VAL_SPLIT = 0.15
NONE_CLASS_NAME = "none"
NEGATIVE_IMAGES_PER_QUERY = 40

# CLIP filter used by the organize-clean phase (open_clip ViT-B-32).
CLEAN_CLIP_MODEL = "ViT-B-32"
CLEAN_CLIP_PRETRAINED = "openai"
# Concern images keep their Bing-folder class label. CLIP only removes junk:
# corrupt files (expected score very low) or images where "none" clearly wins.
CLEAN_CLIP_MIN_EXPECTED = 0.10
CLEAN_CLIP_REJECT_NONE_MARGIN = 0.08
# Negative scrape: reject if any concern class exceeds this AND beats none.
CLEAN_CLIP_MIN_SCORE = 0.22
CLEAN_CLIP_MARGIN = 0.02

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
# Scrape this many candidates per ConcernTarget; CLIP keeps only matches.
IMAGES_PER_QUERY = 60
# Minimum CLIP cosine for the expected class before an image may enter its folder.
DOWNLOAD_CLIP_MIN_EXPECTED = 0.20
# Expected class must beat the best non-concern prompt by at least this margin.
DOWNLOAD_CLIP_MARGIN = 0.03
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

    Strict download rule: images scraped with `search_query` are stored in
    `ml/raw_images/<class_name>/` — folder name is always `class_name`, never
    the Bing query string.

    Attributes:
        search_query: Query sent to the image scraper.
        caption: Prompt GroundedSAM uses to detect and annotate the object.
        class_name: Folder name and YOLO class label; must exist in
            `VALID_CLASS_NAMES`.
    """

    search_query: str
    caption: str
    class_name: str


# Photo-specific Bing queries. Vague phrases ("urban street flooding hazard")
# return documents/product ads; keep these concrete and visual.
TARGETS: tuple[ConcernTarget, ...] = (
    ConcernTarget(
        "photo flooded city street cars underwater",
        "flood water on the street",
        "flooding",
    ),
    ConcernTarget(
        "photograph flooded road standing water vehicles",
        "flood water on the street",
        "flooding",
    ),
    ConcernTarget(
        "close up photo deep pothole asphalt road",
        "pothole in the road",
        "pothole",
    ),
    ConcernTarget(
        "photograph cracked broken asphalt road hole",
        "pothole in the road",
        "pothole",
    ),
    ConcernTarget(
        "photo pile of garbage trash on street",
        "pile of garbage or rubbish",
        "refuse_dump",
    ),
    ConcernTarget(
        "photograph illegal dumping trash heap outdoors",
        "pile of garbage or rubbish",
        "refuse_dump",
    ),
    ConcernTarget(
        "photo overflowing garbage dumpster street",
        "overflowing garbage bin",
        "refuse_dump",
    ),
    ConcernTarget(
        "photo clogged storm drain gutter debris street",
        "blocked drain or gutter",
        "blocked_drain",
    ),
    ConcernTarget(
        "photograph open sewer drain dirty water street",
        "blocked drain or gutter",
        "blocked_drain",
    ),
    ConcernTarget(
        "photo factory chimney black smoke air pollution",
        "smoke or air pollution",
        "pollution",
    ),
    ConcernTarget(
        "photograph burning trash dump thick smoke outdoors",
        "smoke or air pollution",
        "pollution",
    ),
    ConcernTarget(
        "photo broken bent street light pole road",
        "broken public infrastructure",
        "broken_public_facility",
    ),
    ConcernTarget(
        "photograph fallen utility electricity pole street",
        "broken public infrastructure",
        "broken_public_facility",
    ),
    ConcernTarget(
        "photo dirty filthy public toilet latrine outdoors",
        "dirty unsanitary public area",
        "sanitation",
    ),
    ConcernTarget(
        "photograph unsanitary open defecation area outdoors",
        "dirty unsanitary public area",
        "sanitation",
    ),
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
    """Map each ConcernTarget search_query to its class_name.

    Downloads no longer use query-named folders; images live in
    `raw_images/<class_name>/`. This map remains for tooling that still
    references the original Bing query string.

    Returns:
        A dict from `search_query` to `class_name`.
    """
    return {target.search_query: target.class_name for target in TARGETS}


# CLIP text prompts used to sort scraped images into class folders.
# Keep prompts concrete and visual so Bing noise (selfies, ads, game art)
# scores higher on NON_CONCERN_PROMPTS and is rejected.
CLASS_PROMPTS: dict[str, tuple[str, ...]] = {
    "flooding": (
        "flooded road or street with standing water",
        "urban flooding hazard on a road",
        "stagnant flood water covering pavement",
    ),
    "pothole": (
        "pothole in asphalt road surface",
        "cracked damaged road pavement",
        "broken road surface hole",
    ),
    "refuse_dump": (
        "pile of garbage or rubbish on the street",
        "illegal dumping trash heap outdoors",
        "overflowing waste bin with garbage",
    ),
    "blocked_drain": (
        "blocked drainage gutter with waste",
        "open sewage drain in a street",
        "clogged storm drain with debris",
    ),
    "pollution": (
        "factory smoke air pollution",
        "burning waste smoke outdoors",
        "thick smoke air pollution in a city",
    ),
    "broken_public_facility": (
        "broken street light pole",
        "fallen electricity pole on a street",
        "damaged public infrastructure outdoors",
    ),
    "sanitation": (
        "dirty unsanitary public toilet area",
        "filthy public sanitation problem outdoors",
        "dirty public area with sanitation hazard",
    ),
}

NON_CONCERN_PROMPTS: tuple[str, ...] = (
    "selfie portrait of a person",
    "food meal on a plate",
    "pet cat or dog photo",
    "indoor home or office scene",
    "product advertisement or shopping photo",
    "electronic device product photo white background",
    "scanned paper document or form",
    "bank statement or administrative certificate",
    "wallpaper texture or interior decor sample",
    "rally race car motorsport splash",
    "video game cover or digital artwork",
    "nature landscape without damage",
    "DIY craft project or fridge magnets",
    "educational infographic or chart",
    "statistics or data visualization diagram",
    "poster or meme with text",
    "random personal photo",
)
