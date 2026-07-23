"""End-to-end pipeline: scrape images, auto-label, train YOLOv11, export weights.

Phases run independently so you can iterate without repeating slow steps:

    python -m ml.pipeline --phase download
    python -m ml.pipeline --phase organize-clean   # CLIP → class folders
    python -m ml.pipeline --phase prepare-cls
    python -m ml.pipeline --phase train-cls
    python -m ml.pipeline --phase export-cls
    python -m ml.pipeline --phase classifier       # full classification path

Detection path:

    python -m ml.pipeline --phase label
    python -m ml.pipeline --phase train
    python -m ml.pipeline --phase export

Heavy dependencies (torch, open_clip, autodistill, GroundedSAM) are imported
lazily inside each phase so a single-phase run only needs that phase's packages.
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



def _fetch_search_images(query: str, limit: int, dest_dir: Path) -> int:
    """Download up to `limit` images for `query` into `dest_dir` via DuckDuckGo.

    Returns the number of files written. Uses DDGS image search instead of Bing,
    which returned unrelated documents/product ads for concern queries.
    """
    import hashlib
    import urllib.request

    from ddgs import DDGS

    dest_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=limit))
    except Exception as exc:
        logger.warning("Image search failed for %r: %s", query, exc)
        return 0

    for index, result in enumerate(results):
        url = result.get("image") or result.get("thumbnail")
        if not url:
            continue
        digest = hashlib.md5(f"{query}:{url}".encode()).hexdigest()[:12]
        # Extension from URL when possible; always convert later via _copy_as_jpeg.
        suffix = Path(url.split("?", 1)[0]).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            suffix = ".jpg"
        destination = dest_dir / f"cand_{digest}{suffix}"
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; did-backend-api/1.0)"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read()
            if len(data) < 2048:
                continue
            destination.write_bytes(data)
            written += 1
        except Exception:
            continue
    return written


def download_images() -> Path:
    """Download ConcernTarget images into `raw_images/<class_name>/`.

    Strict rules:
    1. Folder name is exactly `ConcernTarget.class_name` (never the search query).
    2. Images are scraped with DuckDuckGo using `search_query`.
    3. An image is only saved if CLIP scores it as that class (beats non-concern
       prompts). Documents, ads, and unrelated scrapes are discarded.

    Returns:
        The directory containing per-class subfolders of verified images.
    """
    import hashlib
    import tempfile

    import torch

    invalid = sorted({t.class_name for t in config.TARGETS} - config.VALID_CLASS_NAMES)
    if invalid:
        raise ValueError(f"ConcernTarget class_name must be in VALID_CLASS_NAMES: {invalid}")

    config.RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Downloading %d ConcernTargets (%d candidates each); CLIP keeps only class matches",
        len(config.TARGETS),
        config.IMAGES_PER_QUERY,
    )

    model, preprocess, tokenizer = _load_clip_for_cleaning()
    prompts, labels = _build_clean_prompt_table()
    with torch.no_grad():
        text_tokens = tokenizer(prompts)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    for target in config.TARGETS:
        class_dir = config.RAW_IMAGES_DIR / target.class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        query_slug = hashlib.md5(target.search_query.encode()).hexdigest()[:12]
        marker = class_dir / f".downloaded_{query_slug}"
        if marker.is_file():
            logger.info(
                "Skipping already-downloaded target: %s -> %s/",
                target.search_query,
                target.class_name,
            )
            continue

        with tempfile.TemporaryDirectory() as tmp:
            scraped = Path(tmp) / "candidates"
            fetched = _fetch_search_images(target.search_query, config.IMAGES_PER_QUERY, scraped)
            logger.info(
                "  fetched %d candidates for %r",
                fetched,
                target.search_query,
            )

            kept = 0
            rejected = 0
            for image_path in sorted(scraped.rglob("*")):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                scores = _clip_label_scores(
                    image_path,
                    model=model,
                    preprocess=preprocess,
                    text_features=text_features,
                    labels=labels,
                )
                expected_score = scores.get(target.class_name, 0.0)
                none_score = scores.get(config.NONE_CLASS_NAME, 0.0)
                if expected_score < config.DOWNLOAD_CLIP_MIN_EXPECTED or none_score >= (
                    expected_score + config.DOWNLOAD_CLIP_MARGIN
                ):
                    rejected += 1
                    continue

                digest = hashlib.md5(
                    f"{target.search_query}:{image_path.name}".encode()
                ).hexdigest()[:12]
                destination = class_dir / f"{target.class_name}_{digest}.jpg"
                if _copy_as_jpeg(image_path, destination):
                    kept += 1

            logger.info(
                "  %s <- %r: kept %d, rejected %d (CLIP gate)",
                target.class_name,
                target.search_query,
                kept,
                rejected,
            )
            if kept == 0:
                logger.warning(
                    "No verified images for target %r -> %s/; try a better search_query",
                    target.search_query,
                    target.class_name,
                )
                continue
            marker.write_text(
                f"{target.search_query}\nkept={kept}\nrejected={rejected}\n",
                encoding="utf-8",
            )

    for class_name in sorted(config.VALID_CLASS_NAMES):
        class_dir = config.RAW_IMAGES_DIR / class_name
        count = (
            sum(1 for p in class_dir.iterdir() if p.is_file() and not p.name.startswith("."))
            if class_dir.is_dir()
            else 0
        )
        logger.info("Class folder %-24s %d verified images", class_name + "/", count)

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
    """Scrape non-concern images into `raw_images_negative/none/`.

    Same strict folder rule as concerns: images land in a class-named folder
    (`none`), not in search-query-named folders.

    Returns:
        The directory containing the `none/` class folder.
    """
    import hashlib
    import tempfile

    none_dir = config.NEGATIVE_IMAGES_DIR / config.NONE_CLASS_NAME
    none_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Downloading %d negative queries into %s/",
        len(config.NEGATIVE_QUERIES),
        none_dir,
    )

    for query in config.NEGATIVE_QUERIES:
        query_slug = hashlib.md5(query.encode()).hexdigest()[:12]
        marker = none_dir / f".downloaded_{query_slug}"
        if marker.is_file():
            logger.info("Skipping already-downloaded negative query: %s", query)
            continue

        with tempfile.TemporaryDirectory() as tmp:
            scraped = Path(tmp) / "candidates"
            fetched = _fetch_search_images(query, config.NEGATIVE_IMAGES_PER_QUERY, scraped)
            copied = 0
            for image_path in sorted(scraped.rglob("*")):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                digest = hashlib.md5(f"{query}:{image_path.name}".encode()).hexdigest()[:12]
                destination = none_dir / f"{config.NONE_CLASS_NAME}_{digest}.jpg"
                if _copy_as_jpeg(image_path, destination):
                    copied += 1

            logger.info("  none <- %r: fetched %d, saved %d", query, fetched, copied)
            if copied:
                marker.write_text(f"{query}\n{copied}\n", encoding="utf-8")

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


def _load_clip_for_cleaning() -> tuple[object, object, object]:
    """Load open_clip model/preprocess/tokenizer for the organize-clean phase."""
    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError(
            "organize-clean requires open-clip-torch. "
            "Install with: pip install -r ml/requirements.txt"
        ) from exc

    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLEAN_CLIP_MODEL,
        pretrained=config.CLEAN_CLIP_PRETRAINED,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(config.CLEAN_CLIP_MODEL)
    return model, preprocess, tokenizer


def _build_clean_prompt_table() -> tuple[list[str], list[str]]:
    """Return (prompts, labels) for CLIP zero-shot sorting.

    Labels are either a class name from `VALID_CLASS_NAMES` or `none`.
    """
    prompts: list[str] = []
    labels: list[str] = []
    for class_name, class_prompts in config.CLASS_PROMPTS.items():
        if class_name not in config.VALID_CLASS_NAMES:
            raise ValueError(f"CLASS_PROMPTS has unknown class: {class_name}")
        for prompt in class_prompts:
            prompts.append(prompt)
            labels.append(class_name)
    for prompt in config.NON_CONCERN_PROMPTS:
        prompts.append(prompt)
        labels.append(config.NONE_CLASS_NAME)
    return prompts, labels


def _clip_label_scores(
    image_path: Path,
    *,
    model: object,
    preprocess: object,
    text_features: object,
    labels: list[str],
) -> dict[str, float]:
    """Return max cosine similarity per label (concern classes + none)."""
    import torch
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(image_path) as image:
            tensor = preprocess(image.convert("RGB")).unsqueeze(0)
    except (UnidentifiedImageError, OSError):
        return {config.NONE_CLASS_NAME: 1.0}

    with torch.no_grad():
        image_features = model.encode_image(tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarities = (image_features @ text_features.T).squeeze(0)

    best_by_label: dict[str, float] = {}
    for index, label in enumerate(labels):
        score = float(similarities[index].item())
        if label not in best_by_label or score > best_by_label[label]:
            best_by_label[label] = score
    return best_by_label


def _classify_image_with_clip(
    image_path: Path,
    *,
    model: object,
    preprocess: object,
    text_features: object,
    labels: list[str],
) -> tuple[str, float, float]:
    """Return (predicted_label, concern_score, none_score) via cosine similarity.

    Scores are max cosine similarity per label group (not softmax), which is
    more stable when many prompts compete.
    """
    best_by_label = _clip_label_scores(
        image_path,
        model=model,
        preprocess=preprocess,
        text_features=text_features,
        labels=labels,
    )

    none_score = best_by_label.get(config.NONE_CLASS_NAME, 0.0)
    concern_scores = {
        label: score
        for label, score in best_by_label.items()
        if label != config.NONE_CLASS_NAME
    }
    if not concern_scores:
        return config.NONE_CLASS_NAME, 0.0, none_score

    predicted = max(concern_scores, key=concern_scores.get)
    concern_score = concern_scores[predicted]
    if concern_score < none_score + config.CLEAN_CLIP_MARGIN:
        return config.NONE_CLASS_NAME, concern_score, none_score
    return predicted, concern_score, none_score


def organize_clean_images(*, rebuild: bool = True) -> Path:
    """Filter scraped Bing images with CLIP and store them in class folders.

    Reads `raw_images/<query>/` (folder name maps to class via `query_to_class()`)
    and optional `raw_images_negative/`. CLIP validates each image against the
    **expected** class prompts and non-concern prompts — junk is rejected even
    if it came from the right Bing folder.

    - `raw_images_clean/<class_name>/` for accepted concern images
    - `raw_images_clean/none/` for clearly non-concern negatives (from negatives)
    - `raw_images_rejected/` for Bing noise that does not match a concern

    Folder names match `ReportCategory` values (plus `none`), not Bing queries.

    Args:
        rebuild: When True (default), wipe previous clean/rejected dirs first.

    Returns:
        Path to `raw_images_clean`.
    """
    import hashlib

    import torch

    if not config.RAW_IMAGES_DIR.is_dir() or not any(config.RAW_IMAGES_DIR.iterdir()):
        raise RuntimeError(
            f"No scraped images at {config.RAW_IMAGES_DIR}; run --phase download first."
        )

    if rebuild:
        if config.CLEAN_IMAGES_DIR.exists():
            shutil.rmtree(config.CLEAN_IMAGES_DIR)
        if config.REJECTED_IMAGES_DIR.exists():
            shutil.rmtree(config.REJECTED_IMAGES_DIR)

    config.CLEAN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.REJECTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    model, preprocess, tokenizer = _load_clip_for_cleaning()
    prompts, labels = _build_clean_prompt_table()
    with torch.no_grad():
        text_tokens = tokenizer(prompts)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Strict rule: folder name IS the class (from ConcernTarget.class_name).
    sources: list[tuple[Path, str]] = []
    for class_dir in sorted(p for p in config.RAW_IMAGES_DIR.iterdir() if p.is_dir()):
        expected_class = class_dir.name
        if expected_class not in config.VALID_CLASS_NAMES:
            logger.warning(
                "Skipping folder not in VALID_CLASS_NAMES (must match ConcernTarget): %s",
                expected_class,
            )
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if (
                image_path.is_file()
                and not image_path.name.startswith(".")
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                sources.append((image_path, expected_class))

    none_dir = config.NEGATIVE_IMAGES_DIR / config.NONE_CLASS_NAME
    if none_dir.is_dir():
        for image_path in sorted(none_dir.rglob("*")):
            if (
                image_path.is_file()
                and not image_path.name.startswith(".")
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                sources.append((image_path, config.NONE_CLASS_NAME))
    elif config.NEGATIVE_IMAGES_DIR.is_dir():
        # Legacy layout: query-named subfolders under raw_images_negative/.
        for image_path in sorted(config.NEGATIVE_IMAGES_DIR.rglob("*")):
            if (
                image_path.is_file()
                and not image_path.name.startswith(".")
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                sources.append((image_path, config.NONE_CLASS_NAME))

    accepted: dict[str, int] = {}
    rejected = 0

    for image_path, expected_class in sources:
        scores = _clip_label_scores(
            image_path,
            model=model,
            preprocess=preprocess,
            text_features=text_features,
            labels=labels,
        )
        none_score = scores.get(config.NONE_CLASS_NAME, 0.0)

        if expected_class == config.NONE_CLASS_NAME:
            top_concern = max(
                (
                    (label, score)
                    for label, score in scores.items()
                    if label != config.NONE_CLASS_NAME
                ),
                key=lambda item: item[1],
                default=(config.NONE_CLASS_NAME, 0.0),
            )
            concern_label, concern_score = top_concern
            if (
                concern_score >= config.CLEAN_CLIP_MIN_SCORE
                and concern_score >= none_score + config.CLEAN_CLIP_MARGIN
            ):
                dest_dir = config.REJECTED_IMAGES_DIR
                rejected += 1
                digest = hashlib.md5(str(image_path).encode()).hexdigest()[:12]
                destination = dest_dir / f"rej_{digest}.jpg"
                _copy_as_jpeg(image_path, destination)
                continue
            class_name = config.NONE_CLASS_NAME
        else:
            expected_score = scores.get(expected_class, 0.0)
            if expected_score < config.CLEAN_CLIP_MIN_EXPECTED:
                dest_dir = config.REJECTED_IMAGES_DIR
                rejected += 1
                digest = hashlib.md5(str(image_path).encode()).hexdigest()[:12]
                destination = dest_dir / f"rej_{digest}.jpg"
                _copy_as_jpeg(image_path, destination)
                continue
            if none_score >= expected_score + config.CLEAN_CLIP_REJECT_NONE_MARGIN:
                dest_dir = config.REJECTED_IMAGES_DIR
                rejected += 1
                digest = hashlib.md5(str(image_path).encode()).hexdigest()[:12]
                destination = dest_dir / f"rej_{digest}.jpg"
                _copy_as_jpeg(image_path, destination)
                continue
            class_name = expected_class

        dest_dir = config.CLEAN_IMAGES_DIR / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.md5(str(image_path).encode()).hexdigest()[:12]
        destination = dest_dir / f"{class_name}_{digest}.jpg"
        if _copy_as_jpeg(image_path, destination):
            accepted[class_name] = accepted.get(class_name, 0) + 1

    logger.info("Organized cleaned images into %s", config.CLEAN_IMAGES_DIR)
    for class_name, count in sorted(accepted.items()):
        logger.info("  %-24s %d", class_name, count)
    logger.info("  %-24s %d", "rejected", rejected)

    if sum(accepted.values()) == 0:
        raise RuntimeError(
            "CLIP filter accepted zero images. Lower CLEAN_CLIP_MIN_EXPECTED or "
            "CLEAN_CLIP_REJECT_NONE_MARGIN, inspect raw_images_rejected/, "
            "and re-run download with better queries."
        )
    return config.CLEAN_IMAGES_DIR


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
    """Build a YOLO classification dataset from CLIP-cleaned class folders.

    Prefers `raw_images_clean/<class>/` (from `--phase organize-clean`). Falls
    back to raw Bing query folders only if clean folders are missing, with a
    warning that labels will be noisy.

    Returns:
        The classification dataset root (contains train/ and val/ class dirs).
    """
    if config.CLS_DATASET_DIR.exists():
        shutil.rmtree(config.CLS_DATASET_DIR)

    totals: dict[str, tuple[int, int]] = {}
    clean_ready = (
        config.CLEAN_IMAGES_DIR.is_dir()
        and any(p.is_dir() and any(p.iterdir()) for p in config.CLEAN_IMAGES_DIR.iterdir())
    )

    if clean_ready:
        logger.info("Building cls_dataset from cleaned class folders: %s", config.CLEAN_IMAGES_DIR)
        for class_dir in sorted(p for p in config.CLEAN_IMAGES_DIR.iterdir() if p.is_dir()):
            class_name = class_dir.name
            if class_name != config.NONE_CLASS_NAME and class_name not in config.VALID_CLASS_NAMES:
                logger.warning("Skipping unknown clean class folder: %s", class_name)
                continue
            images = [p for p in sorted(class_dir.rglob("*")) if p.is_file()]
            totals[class_name] = _split_into_class_dir(images, class_name)
    else:
        logger.warning(
            "No cleaned images at %s; falling back to raw_images/<class_name>/ folders "
            "(from ConcernTarget downloads). Run: python -m ml.pipeline --phase organize-clean",
            config.CLEAN_IMAGES_DIR,
        )
        for class_name in sorted(config.VALID_CLASS_NAMES):
            class_dir = config.RAW_IMAGES_DIR / class_name
            if not class_dir.is_dir():
                logger.warning("Missing class folder %s/; skipping", class_name)
                continue
            images = [
                p
                for p in sorted(class_dir.rglob("*"))
                if p.is_file() and not p.name.startswith(".")
            ]
            totals[class_name] = _split_into_class_dir(images, class_name)

        none_dir = config.NEGATIVE_IMAGES_DIR / config.NONE_CLASS_NAME
        if none_dir.is_dir():
            negatives = [
                p for p in sorted(none_dir.rglob("*")) if p.is_file() and not p.name.startswith(".")
            ]
            totals[config.NONE_CLASS_NAME] = _split_into_class_dir(
                negatives, config.NONE_CLASS_NAME
            )
        elif config.NEGATIVE_IMAGES_DIR.is_dir():
            negatives = [
                p
                for p in sorted(config.NEGATIVE_IMAGES_DIR.rglob("*"))
                if p.is_file() and not p.name.startswith(".")
            ]
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
        raise RuntimeError(
            "No images available; run 'download', 'negatives', and 'organize-clean' first."
        )
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
    """Run the full classification pipeline: scrape, clean, prep, train, export."""
    download_images()
    download_negatives()
    organize_clean_images()
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
    "organize-clean": organize_clean_images,
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
