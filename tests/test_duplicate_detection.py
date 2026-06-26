from types import SimpleNamespace
import pytest
from app.duplicate_detection import DuplicateDetector, perceptual_hash_similarity
from app.models import DuplicateStatus, ReportCategory


class FakeRepo:
    def __init__(self, candidates):
        self.candidates = candidates

    async def find_duplicate_candidates(self, **kwargs):
        self.kwargs = kwargs
        return self.candidates


def candidate(
    report_id=1, sha="abc", phash="ffffffffffffffff", vector_similarity=0.9, distance_meters=40
):
    report = SimpleNamespace(id=report_id, image_sha256=sha, perceptual_hash=phash)
    return SimpleNamespace(
        report=report, vector_similarity=vector_similarity, distance_meters=distance_meters
    )


def test_perceptual_hash_similarity() -> None:
    assert perceptual_hash_similarity("ffffffffffffffff", "ffffffffffffffff") == 1.0
    assert perceptual_hash_similarity("0000000000000000", "ffffffffffffffff") == 0.0


@pytest.mark.asyncio
async def test_far_similar_image_is_new() -> None:
    detector = DuplicateDetector(FakeRepo([candidate(distance_meters=150, vector_similarity=0.99)]))
    decision = await detector.evaluate(
        category=ReportCategory.sanitation,
        latitude=5.0,
        longitude=-0.1,
        image_sha256="new",
        perceptual_hash="ffffffffffffffff",
        embedding=[0.0] * 512,
    )
    assert decision.status == DuplicateStatus.new


@pytest.mark.asyncio
async def test_same_category_close_high_similarity_is_duplicate() -> None:
    detector = DuplicateDetector(FakeRepo([candidate(distance_meters=40, vector_similarity=0.9)]))
    decision = await detector.evaluate(
        category=ReportCategory.sanitation,
        latitude=5.0,
        longitude=-0.1,
        image_sha256="new",
        perceptual_hash="ffffffffffffffff",
        embedding=[0.0] * 512,
    )
    assert decision.status == DuplicateStatus.duplicate


@pytest.mark.asyncio
async def test_medium_similarity_is_possible_duplicate() -> None:
    detector = DuplicateDetector(
        FakeRepo([candidate(distance_meters=70, vector_similarity=0.72, phash="0000000000000000")])
    )
    decision = await detector.evaluate(
        category=ReportCategory.sanitation,
        latitude=5.0,
        longitude=-0.1,
        image_sha256="new",
        perceptual_hash="ffffffffffffffff",
        embedding=[0.0] * 512,
    )
    assert decision.status == DuplicateStatus.possible_duplicate
    assert decision.requires_review is True
