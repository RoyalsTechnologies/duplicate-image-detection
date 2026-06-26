from dataclasses import dataclass, field
from app.computer_vision import perceptual_hash_similarity
from app.config import settings
from app.models import DuplicateStatus, Report, ReportCategory
from app.repositories import DuplicateCandidateRow, ReportRepository


@dataclass(frozen=True)
class DuplicateDecision:
    status: DuplicateStatus
    duplicate_of_report_id: int | None
    confidence_score: float | None
    candidate: Report | None
    requires_review: bool = False
    evidence: dict[str, float | str | bool] = field(default_factory=dict)


class DuplicateDetector:
    def __init__(self, repo: ReportRepository):
        self.repo = repo

    async def evaluate(
        self,
        *,
        category: ReportCategory,
        latitude: float,
        longitude: float,
        image_sha256: str,
        perceptual_hash: str | None,
        embedding: list[float],
    ) -> DuplicateDecision:
        candidates = await self.repo.find_duplicate_candidates(
            category=category,
            latitude=latitude,
            longitude=longitude,
            embedding=embedding,
            radius_meters=settings.duplicate_possible_distance_meters,
        )
        if not candidates:
            return DuplicateDecision(DuplicateStatus.new, None, None, None)

        best_decision = DuplicateDecision(DuplicateStatus.new, None, None, None)
        for candidate in candidates:
            decision = self._classify_candidate(candidate, image_sha256, perceptual_hash)
            if decision.status == DuplicateStatus.duplicate:
                return decision
            if decision.status == DuplicateStatus.possible_duplicate and (
                best_decision.confidence_score is None
                or (decision.confidence_score or 0.0) > (best_decision.confidence_score or 0.0)
            ):
                best_decision = decision
        return best_decision

    def _classify_candidate(
        self,
        candidate: DuplicateCandidateRow,
        image_sha256: str,
        perceptual_hash: str | None,
    ) -> DuplicateDecision:
        phash_similarity = perceptual_hash_similarity(
            perceptual_hash, candidate.report.perceptual_hash
        )
        visual_similarity = max(candidate.vector_similarity, phash_similarity)
        exact_image = candidate.report.image_sha256 == image_sha256
        same_physical_problem = self.same_physical_problem_likely(
            candidate.distance_meters, visual_similarity
        )
        confidence = round(visual_similarity, 4)
        evidence: dict[str, float | str | bool] = {
            "distance_meters": round(candidate.distance_meters, 2),
            "vector_similarity": round(candidate.vector_similarity, 4),
            "perceptual_hash_similarity": round(phash_similarity, 4),
            "visual_similarity": confidence,
            "same_physical_problem_likely": same_physical_problem,
        }

        if exact_image and candidate.distance_meters <= settings.duplicate_possible_distance_meters:
            return DuplicateDecision(
                DuplicateStatus.duplicate,
                candidate.report.id,
                1.0,
                candidate.report,
                evidence=evidence,
            )

        if (
            candidate.distance_meters <= settings.duplicate_exact_distance_meters
            and visual_similarity >= settings.duplicate_similarity_threshold
            and phash_similarity >= settings.perceptual_hash_possible_similarity
        ):
            return DuplicateDecision(
                DuplicateStatus.duplicate,
                candidate.report.id,
                confidence,
                candidate.report,
                evidence=evidence,
            )

        if (
            candidate.distance_meters <= settings.duplicate_possible_distance_meters
            and visual_similarity >= settings.possible_duplicate_similarity_threshold
        ):
            return DuplicateDecision(
                DuplicateStatus.possible_duplicate,
                candidate.report.id,
                confidence,
                candidate.report,
                requires_review=True,
                evidence=evidence,
            )

        return DuplicateDecision(
            DuplicateStatus.new, None, confidence, candidate.report, evidence=evidence
        )

    @staticmethod
    def same_physical_problem_likely(distance_meters: float, visual_similarity: float) -> bool:
        return (
            distance_meters <= settings.duplicate_exact_distance_meters
            and visual_similarity >= settings.duplicate_similarity_threshold
        ) or (
            distance_meters <= settings.duplicate_possible_distance_meters
            and visual_similarity >= settings.possible_duplicate_similarity_threshold
        )
