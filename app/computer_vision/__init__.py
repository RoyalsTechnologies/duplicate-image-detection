from app.computer_vision.base import (
    BaseComputerVisionClient,
    DetectedObject,
    EMBEDDING_DIM,
    ImageAnalysis,
    ImageComparison,
    cosine_similarity,
    perceptual_hash_similarity,
)
from app.computer_vision.embedding import EmbeddingComputerVisionClient
from app.computer_vision.factory import SUPPORTED_CV_PROVIDERS, build_cv_client, cv_client
from app.computer_vision.labels import CONCERN_LABEL_TO_CATEGORY
from app.computer_vision.local import LocalComputerVisionClient
from app.computer_vision.yolo import YoloV11ComputerVisionClient

# Backward-compatible alias used by existing tests and imports.
ComputerVisionClient = LocalComputerVisionClient

__all__ = [
    "BaseComputerVisionClient",
    "ComputerVisionClient",
    "DetectedObject",
    "EMBEDDING_DIM",
    "EmbeddingComputerVisionClient",
    "ImageAnalysis",
    "ImageComparison",
    "LocalComputerVisionClient",
    "SUPPORTED_CV_PROVIDERS",
    "YoloV11ComputerVisionClient",
    "build_cv_client",
    "CONCERN_LABEL_TO_CATEGORY",
    "cosine_similarity",
    "cv_client",
    "perceptual_hash_similarity",
]
