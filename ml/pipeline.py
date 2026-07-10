"""End-to-end pipeline: scrape images, auto-label, train YOLOv11, export weights.

Phases run independently so you can iterate without repeating slow steps:

    python ml/pipeline.py --phase all
    python ml/pipeline.py --phase download
    python ml/pipeline.py --phase label
    python ml/pipeline.py --phase train
    python ml/pipeline.py --phase export

Heavy dependencies (torch, autodistill, GroundedSAM) are imported lazily inside
each phase so a single-phase run only needs that phase's packages.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Allow running as a script: `python ml/pipeline.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ml.pipeline")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def download_images() -> Path:
    """Scrape candidate images for every configured search query.

    Returns:
        The directory containing per-query subfolders of downloaded images.
    """
    from bing_image_downloader import downloader

    config.RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    queries = sorted({target.search_query for target in config.TARGETS})
    logger.info("Downloading %d queries (%d images each)", len(queries), config.IMAGES_PER_QUERY)

    for query in queries:
        # bing_image_downloader creates a subfolder named exactly the query.
        target_dir = config.RAW_IMAGES_DIR / query
        if target_dir.is_dir() and any(target_dir.iterdir()):
            logger.info("Skipping already-downloaded query: %s", query)
            continue
        downloader.download(
            query,
            limit=config.IMAGES_PER_QUERY,
            output_dir=str(config.RAW_IMAGES_DIR),
            adult_filter_off=True,
            force_replace=False,
            timeout=60,
            verbose=False,
        )
    return config.RAW_IMAGES_DIR


def flatten_images() -> Path:
    """Convert all scraped images into one flat folder of JPEGs for labeling.

    Auto-labeling assigns classes from the ontology, not the source folder, so a
    single flat directory is sufficient. autodistill's `label()` globs a single
    extension (default ``.jpg``), so every image is normalized to RGB JPEG to
    avoid silently dropping ``.png``/``.jpeg``/``.webp`` files.

    Returns:
        The flat directory of uniquely named ``.jpg`` images.
    """
    from PIL import Image, UnidentifiedImageError

    if config.FLAT_IMAGES_DIR.exists():
        shutil.rmtree(config.FLAT_IMAGES_DIR)
    config.FLAT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for image_path in sorted(config.RAW_IMAGES_DIR.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        destination = config.FLAT_IMAGES_DIR / f"img_{count:05d}.jpg"
        try:
            with Image.open(image_path) as image:
                image.convert("RGB").save(destination, format="JPEG", quality=90)
        except (UnidentifiedImageError, OSError):
            logger.warning("Skipping unreadable image: %s", image_path)
            continue
        count += 1

    logger.info("Flattened %d images into %s", count, config.FLAT_IMAGES_DIR)
    if count == 0:
        raise RuntimeError("No images found to label; run the download phase first.")
    return config.FLAT_IMAGES_DIR


def _write_data_yaml() -> Path:
    """Write a `data.yaml` for the accumulated dataset.

    The class list is generated from the ontology (deterministic order) rather
    than autodistill's per-batch output, so class IDs stay stable across
    resumed batches.

    Returns:
        Path to the written `data.yaml`.
    """
    import yaml

    class_names = list(config.build_ontology().values())
    data = {
        "names": class_names,
        "nc": len(class_names),
        "train": str(config.DATASET_DIR / "train" / "images"),
        "val": str(config.DATASET_DIR / "valid" / "images"),
    }
    data_yaml = config.DATASET_DIR / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False))
    return data_yaml


def _merge_split(batch_output: Path) -> None:
    """Move a batch's train/valid images and labels into the final dataset."""
    for split in ("train", "valid"):
        for kind in ("images", "labels"):
            source = batch_output / split / kind
            if not source.is_dir():
                continue
            destination = config.DATASET_DIR / split / kind
            destination.mkdir(parents=True, exist_ok=True)
            for item in source.iterdir():
                shutil.move(str(item), str(destination / item.name))


def auto_label(batch_size: int = 20) -> Path:
    """Auto-annotate the flat image folder with GroundedSAM, resumably.

    Images are labeled in small batches and merged into the dataset after each
    batch, with completed filenames recorded in a manifest. A rerun skips
    already-labeled images, so an interruption (sleep, reboot, crash) costs at
    most the current batch instead of the whole run.

    Args:
        batch_size: Number of images to label per committed batch.

    Returns:
        Path to the generated `data.yaml` describing the YOLO dataset.
    """
    import tempfile

    from autodistill.detection import CaptionOntology
    from autodistill_grounded_sam import GroundedSAM

    if not config.FLAT_IMAGES_DIR.is_dir() or not any(config.FLAT_IMAGES_DIR.iterdir()):
        flatten_images()

    config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    manifest = config.DATASET_DIR / ".labeled.txt"
    done: set[str] = set(manifest.read_text().split()) if manifest.is_file() else set()

    images = sorted(config.FLAT_IMAGES_DIR.glob("*.jpg"))
    pending = [image for image in images if image.name not in done]
    logger.info(
        "Auto-labeling: %d pending, %d already labeled (batch size %d)",
        len(pending),
        len(done),
        batch_size,
    )

    ontology = CaptionOntology(config.build_ontology())
    model: GroundedSAM | None = None

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        if model is None:
            # Loading GroundingDINO + SAM is expensive; do it once, lazily.
            model = GroundedSAM(ontology=ontology)

        with tempfile.TemporaryDirectory() as batch_in, tempfile.TemporaryDirectory() as batch_out:
            batch_in_dir = Path(batch_in)
            for image in batch:
                shutil.copy(image, batch_in_dir / image.name)
            model.label(
                input_folder=str(batch_in_dir),
                extension=".jpg",
                output_folder=batch_out,
            )
            _merge_split(Path(batch_out))

        with manifest.open("a") as handle:
            handle.writelines(f"{image.name}\n" for image in batch)
        logger.info("Committed %d/%d images", min(start + batch_size, len(pending)), len(pending))

    data_yaml = _write_data_yaml()
    if not data_yaml.is_file():
        raise RuntimeError(f"Auto-labeling did not produce {data_yaml}")
    return data_yaml


def train() -> Path:
    """Train YOLOv11 on the auto-labeled dataset.

    Returns:
        Path to the best checkpoint produced by training.
    """
    from ultralytics import YOLO

    data_yaml = config.DATASET_DIR / "data.yaml"
    if not data_yaml.is_file():
        raise RuntimeError(f"Missing {data_yaml}; run the label phase first.")

    logger.info("Training %s for %d epochs", config.BASE_MODEL, config.TRAIN_EPOCHS)
    model = YOLO(config.BASE_MODEL)
    model.train(
        data=str(data_yaml),
        epochs=config.TRAIN_EPOCHS,
        imgsz=config.IMAGE_SIZE,
        project=str(config.RUNS_DIR),
        name="train_run",
        exist_ok=True,
    )
    best_weights = config.RUNS_DIR / "train_run" / "weights" / "best.pt"
    if not best_weights.is_file():
        raise RuntimeError(f"Training finished but {best_weights} was not created.")
    return best_weights


def export(best_weights: Path | None = None) -> Path:
    """Copy the trained weights to the deployment folder and API cache.

    Args:
        best_weights: Optional explicit checkpoint path. Defaults to the latest
            training run's best checkpoint.

    Returns:
        The deployment weights path.
    """
    best_weights = best_weights or (config.RUNS_DIR / "train_run" / "weights" / "best.pt")
    if not best_weights.is_file():
        raise RuntimeError(f"No weights to export at {best_weights}; run the train phase first.")

    config.DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    deploy_path = config.DEPLOY_DIR / "best.pt"
    shutil.copy2(best_weights, deploy_path)
    logger.info("Exported weights to %s", deploy_path)

    # Optionally drop weights where the API expects them (CV_YOLO_MODEL).
    api_model_path = os.environ.get("CV_YOLO_MODEL")
    if api_model_path:
        destination = Path(api_model_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, destination)
        logger.info("Copied weights to API model path %s", destination)

    return deploy_path


def run_all() -> None:
    """Run every phase in order."""
    download_images()
    flatten_images()
    auto_label()
    best = train()
    export(best)
    logger.info("Pipeline run successfully complete.")


# ---------------------------------------------------------------------------
# Image-classification pipeline (robust alternative to detection)
# ---------------------------------------------------------------------------


def download_negatives() -> Path:
    """Scrape generic non-concern images for the classifier's "none" class.

    Returns:
        The directory containing per-query subfolders of negative images.
    """
    from bing_image_downloader import downloader

    config.NEGATIVE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %d negative queries", len(config.NEGATIVE_QUERIES))
    for query in config.NEGATIVE_QUERIES:
        target_dir = config.NEGATIVE_IMAGES_DIR / query
        if target_dir.is_dir() and any(target_dir.iterdir()):
            logger.info("Skipping already-downloaded negative query: %s", query)
            continue
        downloader.download(
            query,
            limit=config.NEGATIVE_IMAGES_PER_QUERY,
            output_dir=str(config.NEGATIVE_IMAGES_DIR),
            adult_filter_off=True,
            force_replace=False,
            timeout=60,
            verbose=False,
        )
    return config.NEGATIVE_IMAGES_DIR


def _copy_as_jpeg(source: Path, destination: Path) -> bool:
    """Convert an image to RGB JPEG at `destination`. Returns success."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(source) as image:
            image.convert("RGB").save(destination, format="JPEG", quality=90)
        return True
    except (UnidentifiedImageError, OSError):
        logger.warning("Skipping unreadable image: %s", source)
        return False


def _split_into_class_dir(images: list[Path], class_name: str) -> tuple[int, int]:
    """Copy images into train/val class subfolders. Returns (train, val) counts.

    A deterministic hash of the filename assigns each image to the validation
    split so reruns are stable.
    """
    import hashlib

    train_count = 0
    val_count = 0
    for image in images:
        digest = hashlib.md5(image.name.encode()).hexdigest()  # noqa: S324 (non-crypto use)
        fraction = int(digest[:8], 16) / 0xFFFFFFFF
        split = "val" if fraction < config.CLS_VAL_SPLIT else "train"
        destination_dir = config.CLS_DATASET_DIR / split / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        unique = f"{class_name}_{digest[:10]}.jpg"
        if _copy_as_jpeg(image, destination_dir / unique):
            if split == "val":
                val_count += 1
            else:
                train_count += 1
    return train_count, val_count


def prepare_classification() -> Path:
    """Build a YOLO classification dataset from scraped, query-grouped images.

    Each `raw_images/<query>/` folder maps to its concern class, and negative
    images become the "none" class. No per-image annotation is required.

    Returns:
        The classification dataset root (contains train/ and val/ class dirs).
    """
    if config.CLS_DATASET_DIR.exists():
        shutil.rmtree(config.CLS_DATASET_DIR)

    query_class = config.query_to_class()
    totals: dict[str, tuple[int, int]] = {}

    for query, class_name in query_class.items():
        query_dir = config.RAW_IMAGES_DIR / query
        if not query_dir.is_dir():
            logger.warning("Missing scraped folder for query %r; skipping", query)
            continue
        images = [p for p in sorted(query_dir.rglob("*")) if p.is_file()]
        train_count, val_count = _split_into_class_dir(images, class_name)
        prev = totals.get(class_name, (0, 0))
        totals[class_name] = (prev[0] + train_count, prev[1] + val_count)

    if config.NEGATIVE_IMAGES_DIR.is_dir():
        negatives = [p for p in sorted(config.NEGATIVE_IMAGES_DIR.rglob("*")) if p.is_file()]
        totals[config.NONE_CLASS_NAME] = _split_into_class_dir(
            negatives, config.NONE_CLASS_NAME
        )
    else:
        logger.warning(
            "No negative images at %s; the classifier will lack a 'none' class "
            "and cannot reject non-concern uploads. Run the 'negatives' phase.",
            config.NEGATIVE_IMAGES_DIR,
        )

    logger.info("Classification dataset built at %s", config.CLS_DATASET_DIR)
    for class_name, (train_count, val_count) in sorted(totals.items()):
        logger.info("  %-24s train=%d val=%d", class_name, train_count, val_count)

    total_images = sum(t + v for t, v in totals.values())
    if total_images == 0:
        raise RuntimeError("No images available; run 'download' and 'negatives' first.")
    return config.CLS_DATASET_DIR


def train_classifier() -> Path:
    """Train a YOLOv11 classification model on the prepared dataset.

    Returns:
        Path to the best checkpoint produced by training.
    """
    from ultralytics import YOLO

    if not (config.CLS_DATASET_DIR / "train").is_dir():
        raise RuntimeError(
            f"Missing {config.CLS_DATASET_DIR}/train; run the 'prepare-cls' phase first."
        )

    logger.info(
        "Training classifier %s for %d epochs", config.CLS_BASE_MODEL, config.CLS_TRAIN_EPOCHS
    )
    model = YOLO(config.CLS_BASE_MODEL)
    model.train(
        data=str(config.CLS_DATASET_DIR),
        epochs=config.CLS_TRAIN_EPOCHS,
        imgsz=config.CLS_IMAGE_SIZE,
        project=str(config.CLS_RUNS_DIR),
        name="cls_run",
        exist_ok=True,
    )
    best_weights = config.CLS_RUNS_DIR / "cls_run" / "weights" / "best.pt"
    if not best_weights.is_file():
        raise RuntimeError(f"Training finished but {best_weights} was not created.")
    return best_weights


def export_classifier(best_weights: Path | None = None) -> Path:
    """Copy the trained classifier weights to the deployment folder and API cache.

    Args:
        best_weights: Optional explicit checkpoint path. Defaults to the latest
            classification run's best checkpoint.

    Returns:
        The deployment weights path.
    """
    best_weights = best_weights or (config.CLS_RUNS_DIR / "cls_run" / "weights" / "best.pt")
    if not best_weights.is_file():
        raise RuntimeError(
            f"No classifier weights at {best_weights}; run the 'train-cls' phase first."
        )

    config.DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    deploy_path = config.DEPLOY_DIR / "best_cls.pt"
    shutil.copy2(best_weights, deploy_path)
    logger.info("Exported classifier weights to %s", deploy_path)

    api_model_path = os.environ.get("CV_CLASSIFIER_MODEL")
    if api_model_path:
        destination = Path(api_model_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, destination)
        logger.info("Copied classifier weights to API model path %s", destination)

    return deploy_path


def run_classifier_all() -> None:
    """Run the full classification pipeline: scrape negatives, prep, train, export."""
    download_images()
    download_negatives()
    prepare_classification()
    best = train_classifier()
    export_classifier(best)
    logger.info("Classification pipeline run successfully complete.")


PHASES = {
    "download": lambda: (download_images(), flatten_images()),
    "label": auto_label,
    "train": train,
    "export": export,
    "all": run_all,
    "negatives": download_negatives,
    "prepare-cls": prepare_classification,
    "train-cls": train_classifier,
    "export-cls": export_classifier,
    "classifier": run_classifier_all,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated YOLOv11 training pipeline")
    parser.add_argument(
        "--phase",
        choices=sorted(PHASES.keys()),
        default="all",
        help="Which phase to run (default: all).",
    )
    args = parser.parse_args()
    PHASES[args.phase]()


if __name__ == "__main__":
    main()
