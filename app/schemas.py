from datetime import datetime
from typing import Generic, TypeVar
from pydantic import BaseModel, Field
from app.models import DuplicateStatus, ReportCategory, ReportStatus, ReviewStatus

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    success: bool = True
    message: str
    data: T | None = None


class ApiErrorResponse(BaseModel):
    success: bool = False
    message: str
    errors: list[dict[str, object] | str] = Field(default_factory=list)


class ReportCreate(BaseModel):
    title: str = Field(min_length=3, max_length=160)
    description: str | None = Field(default=None, max_length=5000)
    category: ReportCategory
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class ReportRead(BaseModel):
    id: int
    title: str
    description: str | None
    category: ReportCategory
    latitude: float
    longitude: float
    image_url: str
    image_sha256: str
    perceptual_hash: str | None
    detected_objects: list[dict[str, object]] | None = None
    cv_inferred_category: str | None = None
    cv_confidence_score: float | None = None
    duplicate_status: DuplicateStatus
    duplicate_of_report_id: int | None
    confidence_score: float | None
    status: ReportStatus
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ReportStatusUpdate(BaseModel):
    status: ReportStatus
    notes: str | None = Field(default=None, max_length=2000)


class SupportingEvidenceCreate(BaseModel):
    title: str = Field(min_length=3, max_length=160)
    description: str | None = Field(default=None, max_length=5000)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class DuplicateCandidate(BaseModel):
    report: ReportRead
    confidence_score: float


class DuplicateReviewRead(BaseModel):
    id: int
    report_id: int
    candidate_report_id: int
    confidence_score: float
    status: ReviewStatus
    resolution: str | None
    notes: str | None
    created_at: datetime
    resolved_at: datetime | None

    model_config = {"from_attributes": True}


class DuplicateReviewResolve(BaseModel):
    resolution: DuplicateStatus
    duplicate_of_report_id: int | None = None
    notes: str | None = Field(default=None, max_length=2000)


class DuplicateReviewMerge(BaseModel):
    target_report_id: int
    notes: str | None = Field(default=None, max_length=2000)
