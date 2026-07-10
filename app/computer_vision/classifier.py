"""YOLOv11 image-classification CV client.

Unlike the detection client, this predicts a single concern category per image
and abstains (returns no detections) when the top class is the negative "none"
class or falls below a confidence threshold. Abstaining makes the API reject
uploads that are not an environmental or public concern.
"""

import io
from pathlib import Path
from typing import Any

from PIL import Image

from app.computer_vision.base import DetectedObject, ImageAnalysis, RelevanceAssessment
from app.computer_vision.constants import MIN_YOLO_WEIGHTS_BYTES
from app.computer_vision.embedding import EmbeddingComputerVisionClient
from app.computer_vision.labels import CONCERN_LABEL_TO_CATEGORY, normalize_label

NONE_CLASS_LABELS = frozenset({"none", "other", "background", "negative"})


class YoloClassifierComputerVisionClient(EmbeddingComputerVisionClient):
    """YOLOv11 classification with CLIP embeddings for duplicate matching."""

    def __init__(
        self,
        *,
        model_path: str = "yolo11n-cls.pt",
        confidence: float = 0.35,
        clip_verify_below: float = 0.75,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cpu",
    ) -> None:
        super().__init__(model_name=model_name, pretrained=pretrained, device=device)
        self.model_path = model_path
        self.confidence = confidence
        self.clip_verify_below = clip_verify_below
        self._cls_model: Any | None = None

    def _load_model(self) -> Any:
        if self._cls_model is not None:
            return self._cls_model

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            message = (
                "YOLO classifier CV provider requires optional dependencies. "
                'Install with: pip install -e ".[cv-yolo]"'
            )
            if "libxcb" in str(exc) or "libGL" in str(exc):
                message += (
                    " OpenCV also needs system libraries in Docker "
                    "(libxcb1, libgl1, libglib2.0-0). Rebuild the API image."
                )
            raise RuntimeError(message) from exc

        path = Path(self.model_path)
        if path.is_file() and path.stat().st_size < MIN_YOLO_WEIGHTS_BYTES:
            path.unlink(missing_ok=True)
        if not path.is_file():
            raise RuntimeError(
                f"Classifier weights not found at {self.model_path}. Train them with "
                "`python -m ml.pipeline --phase classifier` and mount best_cls.pt there."
            )
        self._cls_model = YOLO(self.model_path)
        return self._cls_model

    def _classify(self, image_bytes: bytes) -> tuple[str, float]:
        model = self._load_model()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = model.predict(source=image, verbose=False)[0]
        probs = result.probs
        top_index = int(probs.top1)
        confidence = float(probs.top1conf)
        label = str((result.names or {}).get(top_index, top_index))
        return label, confidence

    def detect_objects(self, image_bytes: bytes) -> list[DetectedObject]:
        try:
            label, confidence = self._classify(image_bytes)
        except RuntimeError:
            raise
        except Exception:
            return super().detect_objects(image_bytes)

        if normalize_label(label) in NONE_CLASS_LABELS or confidence < self.confidence:
            return []
        return [DetectedObject(label=label, confidence=confidence)]

    def assess_relevance(
        self, analysis: ImageAnalysis, image_bytes: bytes | None = None
    ) -> RelevanceAssessment:
        """Accept confident classifier hits; verify borderline ones with CLIP."""
        for detected in analysis.detected_objects:
            category = CONCERN_LABEL_TO_CATEGORY.get(normalize_label(detected.label))
            if category is None:
                continue

            if (
                image_bytes is not None
                and detected.confidence < self.clip_verify_below
            ):
                try:
                    clip_assessment = self._assess_relevance_with_clip(image_bytes)
                    if not clip_assessment.is_relevant:
                        return RelevanceAssessment(
                            is_relevant=False,
                            score=clip_assessment.score,
                            reason="clip_overruled_borderline_classifier",
                        )
                except Exception:
                    pass

            return RelevanceAssessment(
                is_relevant=True,
                score=detected.confidence,
                reason="concern_classified",
            )

        if image_bytes is not None:
            clip_assessment = self._clip_fallback_relevance(image_bytes)
            if clip_assessment is not None and clip_assessment.is_relevant:
                return RelevanceAssessment(
                    is_relevant=True,
                    score=clip_assessment.score,
                    reason="clip_concern_when_classifier_abstained",
                )

        return RelevanceAssessment(
            is_relevant=False,
            score=analysis.category_confidence,
            reason="no_concern_classified",
        )

    def _clip_fallback_relevance(self, image_bytes: bytes) -> RelevanceAssessment | None:
        try:
            return self._assess_relevance_with_clip(image_bytes)
        except Exception:
            return None
