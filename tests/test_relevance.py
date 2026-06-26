import pytest

from app.computer_vision.base import DetectedObject, ImageAnalysis
from app.computer_vision.local import LocalComputerVisionClient
from app.exceptions import BadRequestError
from app.models import ReportCategory
from app.services import ReportService


def _analysis(**overrides) -> ImageAnalysis:
    defaults = {
        "perceptual_hash": "abc",
        "embedding": [0.0] * 512,
        "detected_objects": [],
        "inferred_category": None,
        "category_confidence": 0.0,
    }
    defaults.update(overrides)
    return ImageAnalysis(**defaults)


def test_assess_relevance_accepts_mapped_detection(local_cv_client) -> None:
    analysis = _analysis(
        detected_objects=[DetectedObject("garbage", 0.82)],
        inferred_category=ReportCategory.refuse_dump,
        category_confidence=0.82,
    )
    assessment = local_cv_client.assess_relevance(analysis)
    assert assessment.is_relevant is True
    assert assessment.reason == "concern_object_detected"


def test_assess_relevance_rejects_unmapped_detection(local_cv_client) -> None:
    analysis = _analysis(
        detected_objects=[DetectedObject("person", 0.91)],
        inferred_category=ReportCategory.other,
        category_confidence=0.91,
    )
    assessment = local_cv_client.assess_relevance(analysis)
    assert assessment.is_relevant is False


def test_assess_relevance_rejects_empty_detection(local_cv_client) -> None:
    assessment = local_cv_client.assess_relevance(_analysis())
    assert assessment.is_relevant is False


def test_water_like_image_is_relevant(local_cv_client, image_bytes) -> None:
    analysis = local_cv_client.analyze(image_bytes((30, 90, 190)))
    assessment = local_cv_client.assess_relevance(analysis)
    assert assessment.is_relevant is True


def test_neutral_image_is_not_relevant(local_cv_client, image_bytes) -> None:
    analysis = local_cv_client.analyze(image_bytes((200, 120, 160)))
    assessment = local_cv_client.assess_relevance(analysis)
    assert assessment.is_relevant is False


def test_service_rejects_irrelevant_image(monkeypatch, image_bytes) -> None:
    monkeypatch.setattr("app.services.settings.cv_reject_irrelevant_images", True)
    service = ReportService(db=None)  # type: ignore[arg-type]
    irrelevant = image_bytes((200, 120, 160))
    analysis = LocalComputerVisionClient().analyze(irrelevant)
    with pytest.raises(BadRequestError, match="environmental or public concern"):
        service._ensure_relevant_concern_image(irrelevant, analysis)


def test_service_allows_relevant_image_when_enabled(monkeypatch, image_bytes) -> None:
    monkeypatch.setattr("app.services.settings.cv_reject_irrelevant_images", True)
    service = ReportService(db=None)  # type: ignore[arg-type]
    water = image_bytes((30, 90, 190))
    analysis = LocalComputerVisionClient().analyze(water)
    service._ensure_relevant_concern_image(water, analysis)
