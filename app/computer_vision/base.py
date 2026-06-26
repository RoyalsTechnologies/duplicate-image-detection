import io
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image

from app.computer_vision.labels import CONCERN_LABEL_TO_CATEGORY, normalize_label
from app.models import ReportCategory

EMBEDDING_DIM = 512


@dataclass(frozen=True)
class DetectedObject:
    label: str
    confidence: float


@dataclass(frozen=True)
class ImageAnalysis:
    perceptual_hash: str | None
    embedding: list[float]
    detected_objects: list[DetectedObject]
    inferred_category: ReportCategory | None
    category_confidence: float


@dataclass(frozen=True)
class RelevanceAssessment:
    is_relevant: bool
    score: float
    reason: str


@dataclass(frozen=True)
class ImageComparison:
    perceptual_hash_similarity: float
    vector_similarity: float
    visual_similarity: float


def perceptual_hash_similarity(left: str | None, right: str | None) -> float:
    if not left or not right:
        return 0.0
    try:
        left_int = int(left, 16)
        right_int = int(right, 16)
    except ValueError:
        return 0.0
    distance = (left_int ^ right_int).bit_count()
    return max(0.0, 1.0 - (distance / 64.0))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(size))
    left_norm = math.sqrt(sum(left[i] * left[i] for i in range(size)))
    right_norm = math.sqrt(sum(right[i] * right[i] for i in range(size)))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


class BaseComputerVisionClient(ABC):
    """Shared CV contract used by report submission and duplicate detection."""

    @abstractmethod
    def image_embedding(self, image_bytes: bytes) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def detect_objects(self, image_bytes: bytes) -> list[DetectedObject]:
        raise NotImplementedError

    def analyze(self, image_bytes: bytes) -> ImageAnalysis:
        detected_objects = self.detect_objects(image_bytes)
        return ImageAnalysis(
            perceptual_hash=self.perceptual_hash(image_bytes),
            embedding=self.image_embedding(image_bytes),
            detected_objects=detected_objects,
            inferred_category=self.recognize_concern_type(image_bytes, detected_objects),
            category_confidence=self.category_confidence(detected_objects),
        )

    def perceptual_hash(self, image_bytes: bytes) -> str | None:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("L").resize((8, 8))
            pixels = list(image.getdata())
            avg = sum(pixels) / len(pixels)
            bits = "".join("1" if pixel > avg else "0" for pixel in pixels)
            return f"{int(bits, 2):016x}"
        except Exception:
            return None

    def recognize_concern_type(
        self,
        image_bytes: bytes,
        detected_objects: list[DetectedObject] | None = None,
    ) -> ReportCategory | None:
        objects = (
            detected_objects if detected_objects is not None else self.detect_objects(image_bytes)
        )
        return self.category_from_detected_objects(objects)

    def category_from_detected_objects(
        self, objects: list[DetectedObject]
    ) -> ReportCategory | None:
        if not objects:
            return None
        for detected in objects:
            category = CONCERN_LABEL_TO_CATEGORY.get(normalize_label(detected.label))
            if category is not None:
                return category
        return ReportCategory.other

    def category_confidence(self, objects: list[DetectedObject]) -> float:
        return objects[0].confidence if objects else 0.0

    def assess_relevance(
        self, analysis: ImageAnalysis, image_bytes: bytes | None = None
    ) -> RelevanceAssessment:
        from app.config import settings

        min_confidence = settings.cv_relevance_min_detection_confidence
        for detected in analysis.detected_objects:
            if detected.confidence < min_confidence:
                continue
            category = CONCERN_LABEL_TO_CATEGORY.get(normalize_label(detected.label))
            if category is not None:
                return RelevanceAssessment(
                    is_relevant=True,
                    score=detected.confidence,
                    reason="concern_object_detected",
                )

        return RelevanceAssessment(
            is_relevant=False,
            score=analysis.category_confidence,
            reason="no_concern_detected",
        )

    def compare_images(self, left_bytes: bytes, right_bytes: bytes) -> ImageComparison:
        return self.compare_features(
            self.perceptual_hash(left_bytes),
            self.image_embedding(left_bytes),
            self.perceptual_hash(right_bytes),
            self.image_embedding(right_bytes),
        )

    def compare_features(
        self,
        left_hash: str | None,
        left_embedding: list[float],
        right_hash: str | None,
        right_embedding: list[float],
    ) -> ImageComparison:
        phash_similarity = perceptual_hash_similarity(left_hash, right_hash)
        vector_similarity = cosine_similarity(left_embedding, right_embedding)
        normalized_vector_similarity = (vector_similarity + 1.0) / 2.0
        combined_similarity = (0.35 * phash_similarity) + (0.65 * normalized_vector_similarity)
        return ImageComparison(
            perceptual_hash_similarity=phash_similarity,
            vector_similarity=vector_similarity,
            visual_similarity=max(0.0, min(1.0, combined_similarity)),
        )
