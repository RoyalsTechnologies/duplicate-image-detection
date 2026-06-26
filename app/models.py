import enum
from datetime import datetime
from typing import Any
from geoalchemy2 import Geometry
from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ReportCategory(str, enum.Enum):
    refuse_dump = "refuse_dump"
    blocked_drain = "blocked_drain"
    flooding = "flooding"
    pothole = "pothole"
    pollution = "pollution"
    broken_public_facility = "broken_public_facility"
    sanitation = "sanitation"
    other = "other"


class DuplicateStatus(str, enum.Enum):
    new = "new"
    duplicate = "duplicate"
    possible_duplicate = "possible_duplicate"
    supporting_evidence = "supporting_evidence"


class ReportStatus(str, enum.Enum):
    pending = "pending"
    verified = "verified"
    in_progress = "in_progress"
    resolved = "resolved"
    rejected = "rejected"


class ReviewStatus(str, enum.Enum):
    open = "open"
    resolved = "resolved"


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(160))
    description: Mapped[str | None] = mapped_column(Text)
    category: Mapped[ReportCategory] = mapped_column(Enum(ReportCategory, name="report_category"))
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    location_point: Mapped[Any] = mapped_column(Geometry("POINT", srid=4326, spatial_index=True))
    image_url: Mapped[str] = mapped_column(Text)
    image_sha256: Mapped[str] = mapped_column(String(64), index=True)
    perceptual_hash: Mapped[str | None] = mapped_column(String(32))
    image_embedding: Mapped[list[float] | None] = mapped_column(Vector(512))
    detected_objects: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    cv_inferred_category: Mapped[str | None] = mapped_column(String(64))
    cv_confidence_score: Mapped[float | None] = mapped_column(Float)
    duplicate_status: Mapped[DuplicateStatus] = mapped_column(
        Enum(DuplicateStatus, name="duplicate_status"), default=DuplicateStatus.new
    )
    duplicate_of_report_id: Mapped[int | None] = mapped_column(
        ForeignKey("reports.id", ondelete="SET NULL")
    )
    confidence_score: Mapped[float | None] = mapped_column(Float)
    status: Mapped[ReportStatus] = mapped_column(
        Enum(ReportStatus, name="report_status"), default=ReportStatus.pending
    )
    source_ip: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    duplicate_of: Mapped["Report | None"] = relationship(remote_side=[id])


class DuplicateReview(Base):
    __tablename__ = "duplicate_reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    report_id: Mapped[int] = mapped_column(ForeignKey("reports.id", ondelete="CASCADE"))
    candidate_report_id: Mapped[int] = mapped_column(ForeignKey("reports.id", ondelete="CASCADE"))
    confidence_score: Mapped[float] = mapped_column(Float)
    status: Mapped[ReviewStatus] = mapped_column(
        Enum(ReviewStatus, name="review_status"), default=ReviewStatus.open
    )
    resolution: Mapped[str | None] = mapped_column(String(64))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    actor_ip: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(100))
    report_id: Mapped[int | None] = mapped_column(ForeignKey("reports.id", ondelete="SET NULL"))
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    request_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
