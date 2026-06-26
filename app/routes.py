from typing import Annotated
from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import settings
from app.database import get_db
from app.models import ReportCategory
from app.schemas import (
    ApiResponse,
    DuplicateReviewMerge,
    DuplicateReviewRead,
    DuplicateReviewResolve,
    ReportCreate,
    ReportRead,
    ReportStatusUpdate,
    SupportingEvidenceCreate,
)
from app.services import DuplicateReviewService, ReportService

router = APIRouter(prefix=settings.api_v1_prefix)


@router.post("/reports", response_model=ApiResponse[ReportRead], status_code=201)
async def create_report(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    title: Annotated[str, Form()],
    category: Annotated[ReportCategory, Form()],
    latitude: Annotated[float, Form()],
    longitude: Annotated[float, Form()],
    image: Annotated[UploadFile, File()],
    description: Annotated[str | None, Form()] = None,
) -> ApiResponse[ReportRead]:
    payload = ReportCreate(
        title=title,
        description=description,
        category=category,
        latitude=latitude,
        longitude=longitude,
    )
    report = await ReportService(db).submit_report(
        payload, image, request.state.client_ip, request.state.request_id
    )
    return ApiResponse(
        message="Report created successfully", data=ReportRead.model_validate(report)
    )


@router.get("/reports", response_model=ApiResponse[list[ReportRead]])
async def list_reports(
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> ApiResponse[list[ReportRead]]:
    reports = await ReportService(db).list_reports(limit, offset)
    return ApiResponse(
        message="Reports retrieved successfully",
        data=[ReportRead.model_validate(report) for report in reports],
    )


@router.get("/reports/{report_id}", response_model=ApiResponse[ReportRead])
async def get_report(
    report_id: int, db: Annotated[AsyncSession, Depends(get_db)]
) -> ApiResponse[ReportRead]:
    report = await ReportService(db).get_report(report_id)
    return ApiResponse(
        message="Report retrieved successfully", data=ReportRead.model_validate(report)
    )


@router.patch("/reports/{report_id}/status", response_model=ApiResponse[ReportRead])
async def update_report_status(
    report_id: int,
    payload: ReportStatusUpdate,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ApiResponse[ReportRead]:
    report = await ReportService(db).update_status(
        report_id, payload, request.state.client_ip, request.state.request_id
    )
    return ApiResponse(
        message="Report status updated successfully", data=ReportRead.model_validate(report)
    )


@router.post(
    "/reports/{report_id}/supporting-evidence",
    response_model=ApiResponse[ReportRead],
    status_code=201,
)
async def add_supporting_evidence(
    report_id: int,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    title: Annotated[str, Form()],
    latitude: Annotated[float, Form()],
    longitude: Annotated[float, Form()],
    image: Annotated[UploadFile, File()],
    description: Annotated[str | None, Form()] = None,
) -> ApiResponse[ReportRead]:
    payload = SupportingEvidenceCreate(
        title=title, description=description, latitude=latitude, longitude=longitude
    )
    report = await ReportService(db).add_supporting_evidence(
        report_id, payload, image, request.state.client_ip, request.state.request_id
    )
    return ApiResponse(
        message="Supporting evidence added successfully", data=ReportRead.model_validate(report)
    )


@router.get("/reports/{report_id}/duplicates", response_model=ApiResponse[list[ReportRead]])
async def get_report_duplicates(
    report_id: int, db: Annotated[AsyncSession, Depends(get_db)]
) -> ApiResponse[list[ReportRead]]:
    reports = await ReportService(db).duplicates_for(report_id)
    return ApiResponse(
        message="Duplicate reports retrieved successfully",
        data=[ReportRead.model_validate(report) for report in reports],
    )


@router.get("/duplicate-reviews", response_model=ApiResponse[list[DuplicateReviewRead]])
async def list_duplicate_reviews(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ApiResponse[list[DuplicateReviewRead]]:
    reviews = await DuplicateReviewService(db).list_open()
    return ApiResponse(
        message="Duplicate reviews retrieved successfully",
        data=[DuplicateReviewRead.model_validate(review) for review in reviews],
    )


@router.patch(
    "/duplicate-reviews/{review_id}/resolve", response_model=ApiResponse[DuplicateReviewRead]
)
async def resolve_duplicate_review(
    review_id: int,
    payload: DuplicateReviewResolve,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ApiResponse[DuplicateReviewRead]:
    review = await DuplicateReviewService(db).resolve(
        review_id, payload, request.state.client_ip, request.state.request_id
    )
    return ApiResponse(
        message="Duplicate review resolved successfully",
        data=DuplicateReviewRead.model_validate(review),
    )


@router.patch(
    "/duplicate-reviews/{review_id}/merge", response_model=ApiResponse[DuplicateReviewRead]
)
async def merge_duplicate_review(
    review_id: int,
    payload: DuplicateReviewMerge,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ApiResponse[DuplicateReviewRead]:
    review = await DuplicateReviewService(db).merge(
        review_id,
        payload.target_report_id,
        payload.notes,
        request.state.client_ip,
        request.state.request_id,
    )
    return ApiResponse(
        message="Reports merged successfully", data=DuplicateReviewRead.model_validate(review)
    )
