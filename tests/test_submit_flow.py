from types import SimpleNamespace
import pytest
from app.computer_vision import DetectedObject, ImageAnalysis
from app.models import DuplicateStatus, ReportCategory
from app.schemas import ReportCreate
from app.services import ReportService


class FakeDb:
    def __init__(self):
        self.committed = False
        self.rolled_back = False

    async def commit(self):
        self.committed = True

    async def rollback(self):
        self.rolled_back = True


class FakeReviews:
    async def create(self, review):
        return review


class FakeAudit:
    async def log(self, *args, **kwargs):
        return None


class FakeReports:
    def __init__(self, exact=None, fail_create=False):
        self.exact = exact
        self.fail_create = fail_create
        self.created = None

    async def find_exact_duplicate_by_sha256(self, **kwargs):
        return self.exact

    async def find_duplicate_candidates(self, **kwargs):
        return []

    async def create(self, report):
        if self.fail_create:
            raise RuntimeError("db failed")
        report.id = 99
        self.created = report
        return report


@pytest.mark.asyncio
async def test_exact_sha_duplicate_is_checked_before_storage(monkeypatch):
    service = ReportService.__new__(ReportService)
    service.db = FakeDb()
    exact = SimpleNamespace(
        id=7,
        image_url="http://localhost:8000/uploads/existing.jpg",
        perceptual_hash="ffffffffffffffff",
        image_embedding=[0.1] * 512,
        detected_objects=[{"label": "rubbish", "confidence": 0.9}],
        cv_inferred_category="refuse_dump",
        cv_confidence_score=0.9,
    )
    service.reports = FakeReports(exact=exact)
    service.reviews = FakeReviews()
    service.audit = FakeAudit()

    async def fake_read_image(_image):
        return b"same-image"

    async def fake_rate_limit(_ip):
        return None

    async def fail_save(*args, **kwargs):
        raise AssertionError("storage should not be called for exact duplicate")

    monkeypatch.setattr(service, "_read_image", fake_read_image)
    monkeypatch.setattr(service, "_enforce_upload_rate_limit", fake_rate_limit)

    async def fake_cache_lookup(_sha):
        return None

    monkeypatch.setattr("app.services.get_cached_report_for_image_hash", fake_cache_lookup)
    monkeypatch.setattr("app.services.storage.save_upload", fail_save)

    report = await service.submit_report(
        ReportCreate(
            title="Dump site", category=ReportCategory.refuse_dump, latitude=5.0, longitude=-0.1
        ),
        SimpleNamespace(filename="image.jpg"),
        "127.0.0.1",
        "req-1",
    )

    assert report.duplicate_status == DuplicateStatus.duplicate
    assert report.duplicate_of_report_id == 7
    assert report.image_url == exact.image_url
    assert service.db.committed is True


@pytest.mark.asyncio
async def test_new_upload_is_deleted_when_db_write_fails(monkeypatch):
    service = ReportService.__new__(ReportService)
    service.db = FakeDb()
    service.reports = FakeReports(exact=None, fail_create=True)
    service.reviews = FakeReviews()
    service.audit = FakeAudit()
    deleted = []

    async def fake_read_image(_image):
        return b"new-image"

    async def fake_rate_limit(_ip):
        return None

    async def fake_save(*args, **kwargs):
        return "http://localhost:8000/uploads/new.jpg"

    async def fake_delete(url):
        deleted.append(url)

    monkeypatch.setattr(service, "_read_image", fake_read_image)
    monkeypatch.setattr(service, "_enforce_upload_rate_limit", fake_rate_limit)

    async def fake_cache_lookup(_sha):
        return None

    monkeypatch.setattr("app.services.get_cached_report_for_image_hash", fake_cache_lookup)
    monkeypatch.setattr("app.services.storage.save_upload", fake_save)
    monkeypatch.setattr("app.services.storage.delete_url", fake_delete)
    monkeypatch.setattr(
        "app.services.cv_client.analyze",
        lambda _bytes: ImageAnalysis(
            perceptual_hash="abc",
            embedding=[0.0] * 512,
            detected_objects=[DetectedObject("garbage", 0.9)],
            inferred_category=ReportCategory.refuse_dump,
            category_confidence=0.9,
        ),
    )
    monkeypatch.setattr(
        "app.services.cv_client.assess_relevance",
        lambda analysis, image_bytes=None: SimpleNamespace(is_relevant=True),
    )

    with pytest.raises(RuntimeError):
        await service.submit_report(
            ReportCreate(
                title="Dump site", category=ReportCategory.refuse_dump, latitude=5.0, longitude=-0.1
            ),
            SimpleNamespace(filename="image.jpg"),
            "127.0.0.1",
            "req-1",
        )

    assert service.db.rolled_back is True
    assert deleted == ["http://localhost:8000/uploads/new.jpg"]
