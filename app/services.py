import io
from pathlib import Path
from fastapi import UploadFile
from geoalchemy2.functions import ST_MakePoint, ST_SetSRID
from PIL import Image, UnidentifiedImageError
from sqlalchemy.ext.asyncio import AsyncSession
from app.cache import cache_image_hash, get_cached_report_for_image_hash, increment_rate_limit
from app.computer_vision import cv_client
from app.config import settings
from app.duplicate_detection import DuplicateDetector
from app.exceptions import BadRequestError, NotFoundError
from app.models import DuplicateReview, DuplicateStatus, Report, ReportStatus, ReviewStatus
from app.repositories import AuditRepository, ReportRepository, ReviewRepository
from app.schemas import (
    DuplicateReviewResolve,
    ReportCreate,
    ReportStatusUpdate,
    SupportingEvidenceCreate,
)
from app.storage import storage
from app.utils import sanitize_text, sha256_bytes, utcnow

MAX_IMAGE_BYTES = settings.max_upload_size_mb * 1024 * 1024
ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class ReportService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.reports = ReportRepository(db)
        self.reviews = ReviewRepository(db)
        self.audit = AuditRepository(db)

    async def submit_report(
        self,
        payload: ReportCreate,
        image: UploadFile,
        source_ip: str,
        request_id: str | None = None,
    ) -> Report:
        await self._enforce_upload_rate_limit(source_ip)
        image_bytes = await self._read_image(image)
        image_sha256 = sha256_bytes(image_bytes)
        cached_report_id = await get_cached_report_for_image_hash(image_sha256)
        exact_duplicate = await self._find_exact_duplicate_before_storage(
            image_sha256=image_sha256,
            payload=payload,
            cached_report_id=cached_report_id,
        )

        image_url: str | None = None
        saved_new_image = False
        try:
            if exact_duplicate is not None:
                image_url = exact_duplicate.image_url
                perceptual_hash = exact_duplicate.perceptual_hash
                embedding = (
                    exact_duplicate.image_embedding
                    if exact_duplicate.image_embedding is not None
                    else cv_client.image_embedding(image_bytes)
                )
                detected_objects = exact_duplicate.detected_objects
                cv_inferred_category = exact_duplicate.cv_inferred_category
                cv_confidence_score = exact_duplicate.cv_confidence_score
                decision_status = DuplicateStatus.duplicate
                duplicate_of_report_id = exact_duplicate.id
                confidence_score = 1.0
                decision_evidence: dict[str, object] = {
                    "exact_sha256_match": True,
                    "checked_before_storage": True,
                    "matched_report_id": exact_duplicate.id,
                }
                requires_review = False
                decision_candidate = None
            else:
                analysis = cv_client.analyze(image_bytes)
                self._ensure_relevant_concern_image(image_bytes, analysis)
                perceptual_hash = analysis.perceptual_hash
                embedding = analysis.embedding
                image_url = await storage.save_upload(image, image_bytes)
                saved_new_image = True
                decision = await DuplicateDetector(self.reports).evaluate(
                    category=payload.category,
                    latitude=payload.latitude,
                    longitude=payload.longitude,
                    image_sha256=image_sha256,
                    perceptual_hash=perceptual_hash,
                    embedding=embedding,
                )
                detected_objects = [obj.__dict__ for obj in analysis.detected_objects]
                cv_inferred_category = (
                    analysis.inferred_category.value if analysis.inferred_category else None
                )
                cv_confidence_score = analysis.category_confidence
                decision_status = decision.status
                duplicate_of_report_id = decision.duplicate_of_report_id
                confidence_score = decision.confidence_score
                decision_evidence = decision.evidence
                requires_review = decision.requires_review
                decision_candidate = decision.candidate

            report = Report(
                title=sanitize_text(payload.title) or payload.title,
                description=sanitize_text(payload.description),
                category=payload.category,
                latitude=payload.latitude,
                longitude=payload.longitude,
                location_point=ST_SetSRID(ST_MakePoint(payload.longitude, payload.latitude), 4326),
                image_url=image_url,
                image_sha256=image_sha256,
                perceptual_hash=perceptual_hash,
                image_embedding=embedding,
                detected_objects=detected_objects,
                cv_inferred_category=cv_inferred_category,
                cv_confidence_score=cv_confidence_score,
                duplicate_status=decision_status,
                duplicate_of_report_id=duplicate_of_report_id,
                confidence_score=confidence_score,
                status=ReportStatus.pending,
                source_ip=source_ip,
            )
            await self.reports.create(report)
            if requires_review and decision_candidate is not None:
                await self.reviews.create(
                    DuplicateReview(
                        report_id=report.id,
                        candidate_report_id=decision_candidate.id,
                        confidence_score=confidence_score or 0,
                    )
                )
            if report.duplicate_status == DuplicateStatus.new:
                await cache_image_hash(image_sha256, report.id)
            await self.audit.log(
                source_ip,
                "report.created",
                report.id,
                {
                    "duplicate_status": report.duplicate_status.value,
                    "cached_exact_report_id": cached_report_id,
                    "duplicate_evidence": decision_evidence,
                    "cv_inferred_category": report.cv_inferred_category,
                    "detected_objects": report.detected_objects,
                    "stored_new_image": saved_new_image,
                },
                request_id,
            )
            await self.db.commit()
            return report
        except Exception:
            await self.db.rollback()
            if saved_new_image and image_url is not None:
                await storage.delete_url(image_url)
            raise

    async def _find_exact_duplicate_before_storage(
        self,
        *,
        image_sha256: str,
        payload: ReportCreate,
        cached_report_id: int | None,
    ) -> Report | None:
        if cached_report_id is not None:
            cached_match = await self.reports.find_exact_duplicate_by_sha256(
                image_sha256=image_sha256,
                category=payload.category,
                latitude=payload.latitude,
                longitude=payload.longitude,
                radius_meters=settings.duplicate_possible_distance_meters,
                candidate_report_id=cached_report_id,
            )
            if cached_match is not None:
                return cached_match
        return await self.reports.find_exact_duplicate_by_sha256(
            image_sha256=image_sha256,
            category=payload.category,
            latitude=payload.latitude,
            longitude=payload.longitude,
            radius_meters=settings.duplicate_possible_distance_meters,
        )

    async def list_reports(self, limit: int, offset: int) -> list[Report]:
        return await self.reports.list_reports(limit, offset)

    async def get_report(self, report_id: int) -> Report:
        report = await self.reports.get(report_id)
        if report is None:
            raise NotFoundError("Report not found")
        return report

    async def update_status(
        self,
        report_id: int,
        payload: ReportStatusUpdate,
        source_ip: str,
        request_id: str | None = None,
    ) -> Report:
        report = await self.get_report(report_id)
        old_status = report.status.value
        report.status = payload.status
        await self.audit.log(
            source_ip,
            "report.status_updated",
            report.id,
            {
                "old": {"status": old_status},
                "new": {"status": payload.status.value},
                "notes": sanitize_text(payload.notes),
            },
            request_id,
        )
        await self.db.commit()
        await self.db.refresh(report)
        return report

    async def add_supporting_evidence(
        self,
        report_id: int,
        payload: SupportingEvidenceCreate,
        image: UploadFile,
        source_ip: str,
        request_id: str | None = None,
    ) -> Report:
        await self._enforce_upload_rate_limit(source_ip)
        parent = await self.get_report(report_id)
        image_bytes = await self._read_image(image)
        image_url: str | None = None
        try:
            image_url = await storage.save_upload(image, image_bytes)
            supporting_analysis = cv_client.analyze(image_bytes)
            report = Report(
                title=sanitize_text(payload.title) or payload.title,
                description=sanitize_text(payload.description),
                category=parent.category,
                latitude=payload.latitude,
                longitude=payload.longitude,
                location_point=ST_SetSRID(ST_MakePoint(payload.longitude, payload.latitude), 4326),
                image_url=image_url,
                image_sha256=sha256_bytes(image_bytes),
                perceptual_hash=supporting_analysis.perceptual_hash,
                image_embedding=supporting_analysis.embedding,
                detected_objects=[obj.__dict__ for obj in supporting_analysis.detected_objects],
                cv_inferred_category=supporting_analysis.inferred_category.value
                if supporting_analysis.inferred_category
                else None,
                cv_confidence_score=supporting_analysis.category_confidence,
                duplicate_status=DuplicateStatus.supporting_evidence,
                duplicate_of_report_id=parent.id,
                confidence_score=1.0,
                status=parent.status,
                source_ip=source_ip,
            )
            await self.reports.create(report)
            await self.audit.log(
                source_ip,
                "report.supporting_evidence_added",
                report.id,
                {"parent_report_id": parent.id},
                request_id,
            )
            await self.db.commit()
            return report
        except Exception:
            await self.db.rollback()
            if image_url is not None:
                await storage.delete_url(image_url)
            raise

    async def duplicates_for(self, report_id: int) -> list[Report]:
        await self.get_report(report_id)
        return await self.reports.duplicates_for(report_id)

    def _ensure_relevant_concern_image(self, image_bytes: bytes, analysis) -> None:
        if not settings.cv_reject_irrelevant_images:
            return
        assessment = cv_client.assess_relevance(analysis, image_bytes)
        if assessment.is_relevant:
            return
        raise BadRequestError(
            "Image does not appear to show an environmental or public concern. "
            "Please upload a clear photo of the issue you are reporting."
        )

    async def _enforce_upload_rate_limit(self, source_ip: str) -> None:
        attempts = await increment_rate_limit(f"upload-rate:{source_ip}", ttl_seconds=60)
        if attempts > settings.upload_rate_limit_per_minute:
            raise BadRequestError("Too many upload requests; try again later")

    async def _read_image(self, image: UploadFile) -> bytes:
        extension = Path(image.filename or "").suffix.lower()
        if image.content_type not in ALLOWED_IMAGE_MIME_TYPES:
            raise BadRequestError("Uploaded file must be a JPEG, PNG, or WebP image")
        if extension not in ALLOWED_IMAGE_EXTENSIONS:
            raise BadRequestError("Uploaded image extension is not allowed")
        data = await image.read(MAX_IMAGE_BYTES + 1)
        if not data:
            raise BadRequestError("Image is required")
        if len(data) > MAX_IMAGE_BYTES:
            raise BadRequestError("Image exceeds configured upload size limit")
        try:
            with Image.open(io.BytesIO(data)) as parsed_image:
                parsed_image.verify()
        except (UnidentifiedImageError, OSError):
            raise BadRequestError("Uploaded file is not a valid image")
        return data


class DuplicateReviewService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.reports = ReportRepository(db)
        self.reviews = ReviewRepository(db)
        self.audit = AuditRepository(db)

    async def list_open(self) -> list[DuplicateReview]:
        return await self.reviews.list_open()

    async def resolve(
        self,
        review_id: int,
        payload: DuplicateReviewResolve,
        source_ip: str,
        request_id: str | None = None,
    ) -> DuplicateReview:
        review = await self.reviews.get(review_id)
        if review is None:
            raise NotFoundError("Duplicate review not found")
        report = await self.reports.get(review.report_id)
        if report is None:
            raise NotFoundError("Reviewed report not found")
        old_values = {
            "duplicate_status": report.duplicate_status.value,
            "duplicate_of_report_id": report.duplicate_of_report_id,
        }
        report.duplicate_status = payload.resolution
        if payload.resolution in {DuplicateStatus.duplicate, DuplicateStatus.supporting_evidence}:
            report.duplicate_of_report_id = (
                payload.duplicate_of_report_id or review.candidate_report_id
            )
        elif payload.resolution == DuplicateStatus.new:
            report.duplicate_of_report_id = None
        review.status = ReviewStatus.resolved
        review.resolution = payload.resolution.value
        review.notes = sanitize_text(payload.notes)
        review.resolved_at = utcnow()
        await self.audit.log(
            source_ip,
            "duplicate_review.resolved",
            report.id,
            {
                "review_id": review.id,
                "old": old_values,
                "new": {
                    "duplicate_status": payload.resolution.value,
                    "duplicate_of_report_id": report.duplicate_of_report_id,
                },
            },
            request_id,
        )
        await self.db.commit()
        await self.db.refresh(review)
        return review

    async def merge(
        self,
        review_id: int,
        target_report_id: int,
        notes: str | None,
        source_ip: str,
        request_id: str | None = None,
    ) -> DuplicateReview:
        review = await self.reviews.get(review_id)
        if review is None:
            raise NotFoundError("Duplicate review not found")
        report = await self.reports.get(review.report_id)
        target = await self.reports.get(target_report_id)
        if report is None or target is None:
            raise NotFoundError("Report not found")
        old_values = {
            "duplicate_status": report.duplicate_status.value,
            "duplicate_of_report_id": report.duplicate_of_report_id,
        }
        report.duplicate_status = DuplicateStatus.supporting_evidence
        report.duplicate_of_report_id = target.id
        report.confidence_score = review.confidence_score
        review.status = ReviewStatus.resolved
        review.resolution = "merged"
        review.notes = sanitize_text(notes)
        review.resolved_at = utcnow()
        await self.audit.log(
            source_ip,
            "duplicate_review.merged",
            report.id,
            {
                "review_id": review.id,
                "target_report_id": target.id,
                "old": old_values,
                "new": {
                    "duplicate_status": report.duplicate_status.value,
                    "duplicate_of_report_id": target.id,
                },
            },
            request_id,
        )
        await self.db.commit()
        await self.db.refresh(review)
        return review
