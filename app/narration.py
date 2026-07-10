from dataclasses import asdict

from fastapi import UploadFile

from app.cache import increment_rate_limit
from app.computer_vision import cv_client
from app.config import settings
from app.exceptions import BadRequestError
from app.images import read_upload_image
from app.llm.groq_vision import narrate_image_with_groq
from app.models import ReportCategory


class ImageNarrationService:
    """Generate natural-language scene descriptions via Groq vision."""

    async def describe_image(
        self,
        image: UploadFile,
        *,
        category: ReportCategory | None = None,
        source_ip: str,
        skip_relevance_check: bool = False,
    ) -> tuple[str, bool]:
        await self._enforce_narration_rate_limit(source_ip)
        image_bytes, mime_type = await read_upload_image(image)

        detected_objects: list[dict[str, object]] | None = None
        if not skip_relevance_check and settings.cv_reject_irrelevant_images:
            analysis = cv_client.analyze(image_bytes)
            assessment = cv_client.assess_relevance(analysis, image_bytes)
            if not assessment.is_relevant:
                raise BadRequestError(
                    "Image does not appear to show an environmental or public concern. "
                    "Please upload a clear photo of the issue you are reporting."
                )
            detected_objects = [asdict(obj) for obj in analysis.detected_objects]
        elif settings.groq_narration_available:
            analysis = cv_client.analyze(image_bytes)
            detected_objects = [asdict(obj) for obj in analysis.detected_objects]

        if not settings.groq_narration_available:
            return "", False

        narration = await narrate_image_with_groq(
            image_bytes,
            mime_type=mime_type,
            category=category.value if category else None,
            detected_objects=detected_objects,
        )
        if narration:
            return narration, True
        return "", False

    async def narrate_if_missing(
        self,
        image_bytes: bytes,
        *,
        mime_type: str,
        description: str | None,
        category: ReportCategory,
        detected_objects: list[dict[str, object]] | None,
    ) -> str | None:
        if description and description.strip():
            return description
        if not settings.groq_narration_available:
            return description

        narration = await narrate_image_with_groq(
            image_bytes,
            mime_type=mime_type,
            category=category.value,
            detected_objects=detected_objects,
        )
        return narration or description

    async def _enforce_narration_rate_limit(self, source_ip: str) -> None:
        attempts = await increment_rate_limit(f"narration-rate:{source_ip}", ttl_seconds=60)
        if attempts > settings.groq_narration_rate_limit_per_minute:
            raise BadRequestError("Too many description requests; try again later")
