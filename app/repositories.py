from dataclasses import dataclass
from datetime import timedelta
from sqlalchemy import Select, and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import settings
from app.models import AuditLog, DuplicateReview, Report, ReportCategory, ReportStatus, ReviewStatus
from app.utils import utcnow


@dataclass(frozen=True)
class DuplicateCandidateRow:
    report: Report
    vector_similarity: float
    distance_meters: float


ACTIVE_DUPLICATE_STATUSES = (ReportStatus.pending, ReportStatus.verified, ReportStatus.in_progress)


class ReportRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, report: Report) -> Report:
        self.db.add(report)
        await self.db.flush()
        await self.db.refresh(report)
        return report

    async def get(self, report_id: int) -> Report | None:
        return await self.db.get(Report, report_id)

    async def list_reports(self, limit: int = 50, offset: int = 0) -> list[Report]:
        result = await self.db.scalars(
            select(Report).order_by(Report.created_at.desc()).limit(limit).offset(offset)
        )
        return list(result)

    async def find_exact_duplicate_by_sha256(
        self,
        *,
        image_sha256: str,
        category: ReportCategory,
        latitude: float,
        longitude: float,
        radius_meters: int,
        candidate_report_id: int | None = None,
    ) -> Report | None:
        since = utcnow() - timedelta(hours=settings.duplicate_time_window_hours)
        point = func.ST_SetSRID(func.ST_MakePoint(longitude, latitude), 4326)
        stmt = select(Report).where(
            and_(
                Report.image_sha256 == image_sha256,
                Report.category == category,
                Report.status.in_(ACTIVE_DUPLICATE_STATUSES),
                Report.created_at >= since,
                func.ST_DWithin(
                    func.Geography(Report.location_point), func.Geography(point), radius_meters
                ),
            )
        )
        if candidate_report_id is not None:
            stmt = stmt.where(Report.id == candidate_report_id)
        result = await self.db.scalars(stmt.order_by(Report.created_at.desc()).limit(1))
        return result.first()

    async def find_duplicate_candidates(
        self,
        *,
        category: ReportCategory,
        latitude: float,
        longitude: float,
        embedding: list[float],
        radius_meters: int,
        exclude_id: int | None = None,
    ) -> list[DuplicateCandidateRow]:
        since = utcnow() - timedelta(hours=settings.duplicate_time_window_hours)
        point = func.ST_SetSRID(func.ST_MakePoint(longitude, latitude), 4326)
        distance = func.ST_Distance(func.Geography(Report.location_point), func.Geography(point))
        # Location, category, status, and time filters run before vector ordering to avoid full-table image comparison.
        stmt: Select[tuple[Report, float, float]] = select(
            Report,
            (1 - Report.image_embedding.cosine_distance(embedding)).label("vector_similarity"),
            distance.label("distance_meters"),
        ).where(
            and_(
                Report.category == category,
                Report.status.in_(ACTIVE_DUPLICATE_STATUSES),
                Report.created_at >= since,
                func.ST_DWithin(
                    func.Geography(Report.location_point), func.Geography(point), radius_meters
                ),
                Report.image_embedding.is_not(None),
            )
        )
        if exclude_id is not None:
            stmt = stmt.where(Report.id != exclude_id)
        stmt = stmt.order_by(Report.image_embedding.cosine_distance(embedding)).limit(25)
        rows = (await self.db.execute(stmt)).all()
        return [
            DuplicateCandidateRow(
                report=row[0],
                vector_similarity=float(row[1] or 0.0),
                distance_meters=float(row[2] or 0.0),
            )
            for row in rows
        ]

    async def duplicates_for(self, report_id: int) -> list[Report]:
        report = await self.get(report_id)
        if report is None:
            return []
        ids = [report_id]
        if report.duplicate_of_report_id:
            ids.append(report.duplicate_of_report_id)
        result = await self.db.scalars(
            select(Report).where(
                (Report.duplicate_of_report_id.in_(ids))
                | (Report.id == report.duplicate_of_report_id)
            )
        )
        return list(result)


class ReviewRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, review: DuplicateReview) -> DuplicateReview:
        self.db.add(review)
        await self.db.flush()
        await self.db.refresh(review)
        return review

    async def list_open(self) -> list[DuplicateReview]:
        result = await self.db.scalars(
            select(DuplicateReview)
            .where(DuplicateReview.status == ReviewStatus.open)
            .order_by(DuplicateReview.created_at.desc())
        )
        return list(result)

    async def get(self, review_id: int) -> DuplicateReview | None:
        return await self.db.get(DuplicateReview, review_id)


class AuditRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def log(
        self,
        actor_ip: str,
        action: str,
        report_id: int | None = None,
        details: dict | None = None,
        request_id: str | None = None,
    ) -> None:
        self.db.add(
            AuditLog(
                actor_ip=actor_ip,
                action=action,
                report_id=report_id,
                details=details,
                request_id=request_id,
            )
        )
