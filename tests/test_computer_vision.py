import pytest

from app.computer_vision import (
    DetectedObject,
    cosine_similarity,
    perceptual_hash_similarity,
)
from app.computer_vision.base import EMBEDDING_DIM
from app.models import ReportCategory


def test_perceptual_hash_similarity_identical_and_opposite() -> None:
    assert perceptual_hash_similarity("ffffffffffffffff", "ffffffffffffffff") == 1.0
    assert perceptual_hash_similarity("0000000000000000", "ffffffffffffffff") == 0.0
    assert perceptual_hash_similarity(None, "ffffffffffffffff") == 0.0


def test_cosine_similarity_handles_empty_and_identical_vectors() -> None:
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_local_client_identical_images_have_high_visual_similarity(
    local_cv_client, image_bytes
) -> None:
    data = image_bytes((20, 80, 180))
    comparison = local_cv_client.compare_images(data, data)
    assert comparison.visual_similarity >= 0.99
    assert len(local_cv_client.image_embedding(data)) == EMBEDDING_DIM


def test_local_client_different_images_have_lower_visual_similarity(
    local_cv_client, image_bytes
) -> None:
    blue = image_bytes((20, 80, 180))
    brown = image_bytes((130, 85, 35))
    comparison = local_cv_client.compare_images(blue, brown)
    assert comparison.visual_similarity < 0.95


def test_local_client_detects_water_like_concern(local_cv_client, image_bytes) -> None:
    data = image_bytes((30, 90, 190))
    labels = {item.label for item in local_cv_client.detect_objects(data)}
    assert {"stagnant_water", "flooding"} & labels
    assert local_cv_client.recognize_concern_type(data) == ReportCategory.flooding


def test_local_client_analyze_returns_hash_embedding_objects_and_category(
    local_cv_client, image_bytes
) -> None:
    analysis = local_cv_client.analyze(image_bytes((30, 90, 190)))
    assert analysis.perceptual_hash is not None
    assert len(analysis.embedding) == EMBEDDING_DIM
    assert analysis.detected_objects
    assert analysis.inferred_category is not None
    assert analysis.category_confidence > 0


def test_base_client_category_mapping(local_cv_client) -> None:
    assert (
        local_cv_client.category_from_detected_objects(
            [
                DetectedObject("unknown_environmental_issue", 0.81),
            ]
        )
        == ReportCategory.other
    )

    assert (
        local_cv_client.category_from_detected_objects(
            [
                DetectedObject("garbage", 0.81),
            ]
        )
        == ReportCategory.refuse_dump
    )

    cases = {
        "open_gutter": ReportCategory.blocked_drain,
        "sewage overflow": ReportCategory.blocked_drain,
        "illegal-dumping": ReportCategory.refuse_dump,
        "oil_spill": ReportCategory.pollution,
        "fallen_tree": ReportCategory.broken_public_facility,
        "open_defecation": ReportCategory.sanitation,
        "road_erosion": ReportCategory.pothole,
        "standing_water": ReportCategory.flooding,
        "bottle": ReportCategory.refuse_dump,
    }
    for label, expected in cases.items():
        category = local_cv_client.category_from_detected_objects([DetectedObject(label, 0.9)])
        assert category == expected


def test_local_client_returns_empty_detection_for_invalid_image(local_cv_client) -> None:
    assert local_cv_client.detect_objects(b"not-an-image") == []
    assert local_cv_client.recognize_concern_type(b"not-an-image") is None
