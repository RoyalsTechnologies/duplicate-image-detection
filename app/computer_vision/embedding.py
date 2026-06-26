import io
from typing import Any

from PIL import Image

from app.computer_vision.base import (
    EMBEDDING_DIM,
    DetectedObject,
    ImageAnalysis,
    RelevanceAssessment,
)
from app.computer_vision.local import LocalComputerVisionClient
from app.computer_vision.relevance import CONCERN_PROMPTS, NON_CONCERN_PROMPTS


class EmbeddingComputerVisionClient(LocalComputerVisionClient):
    """CLIP-based semantic image embeddings for duplicate detection."""

    def __init__(
        self,
        *,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self._clip_bundle: tuple[Any, Any] | None = None

    def _load_clip(self) -> tuple[Any, Any]:
        if self._clip_bundle is not None:
            return self._clip_bundle

        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError(
                "Embedding CV provider requires optional dependencies. "
                'Install with: pip install -e ".[cv-embedding]"'
            ) from exc

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
        )
        model.eval()
        model.to(self.device)
        self._clip_bundle = (model, preprocess)
        return self._clip_bundle

    def image_embedding(self, image_bytes: bytes) -> list[float]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Embedding CV provider requires optional dependencies. "
                'Install with: pip install -e ".[cv-embedding]"'
            ) from exc

        try:
            model, preprocess = self._load_clip()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = model.encode_image(tensor)
                features = features / features.norm(dim=-1, keepdim=True)
            values = features.squeeze(0).detach().cpu().tolist()
            return self._fit_embedding_dim(values)
        except RuntimeError:
            raise
        except Exception:
            return super().image_embedding(image_bytes)

    def _fit_embedding_dim(self, values: list[float]) -> list[float]:
        if len(values) == EMBEDDING_DIM:
            return values
        if len(values) > EMBEDDING_DIM:
            return values[:EMBEDDING_DIM]
        return values + [0.0] * (EMBEDDING_DIM - len(values))

    def detect_objects(self, image_bytes: bytes) -> list[DetectedObject]:
        return super().detect_objects(image_bytes)

    def assess_relevance(
        self, analysis: ImageAnalysis, image_bytes: bytes | None = None
    ) -> RelevanceAssessment:
        object_assessment = super().assess_relevance(analysis, image_bytes)
        if object_assessment.is_relevant or image_bytes is None:
            return object_assessment
        try:
            return self._assess_relevance_with_clip(image_bytes)
        except Exception:
            return object_assessment

    def _assess_relevance_with_clip(self, image_bytes: bytes) -> RelevanceAssessment:
        import open_clip
        import torch

        from app.config import settings

        model, preprocess = self._load_clip()
        tokenizer = open_clip.get_tokenizer(self.model_name)
        prompts = list(CONCERN_PROMPTS) + list(NON_CONCERN_PROMPTS)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(self.device)
        text_tensor = tokenizer(prompts).to(self.device)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_features @ text_features.T).softmax(dim=-1).squeeze(0)

        concern_count = len(CONCERN_PROMPTS)
        concern_score = float(logits[:concern_count].max().item())
        non_concern_score = float(logits[concern_count:].max().item())
        is_relevant = (
            concern_score >= settings.cv_relevance_threshold and concern_score > non_concern_score
        )
        return RelevanceAssessment(
            is_relevant=is_relevant,
            score=concern_score,
            reason="clip_concern_match" if is_relevant else "clip_not_a_concern",
        )
