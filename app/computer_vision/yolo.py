import io
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from app.computer_vision.base import DetectedObject
from app.computer_vision.embedding import EmbeddingComputerVisionClient

MIN_YOLO_WEIGHTS_BYTES = 1_000_000  # yolo11n.pt is ~5.5MB


def _looks_like_corrupt_yolo_checkpoint(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "corrupted" in message
        or "central directory" in message
        or "zip archive" in message
        or "pytorchstreamreader" in message
    )


class YoloV11ComputerVisionClient(EmbeddingComputerVisionClient):
    """YOLOv11 object detection with CLIP embeddings for duplicate matching."""

    def __init__(
        self,
        *,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.25,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cpu",
    ) -> None:
        super().__init__(model_name=model_name, pretrained=pretrained, device=device)
        self.model_path = model_path
        self.confidence = confidence
        self._yolo_model: Any | None = None

    def _resolve_model_path(self) -> str:
        path = Path(self.model_path)
        if path.is_file() and path.stat().st_size < MIN_YOLO_WEIGHTS_BYTES:
            path.unlink(missing_ok=True)
        return self.model_path

    def _download_yolo_weights(self, dest: Path) -> Any:
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        if dest.is_absolute():
            dest.parent.mkdir(parents=True, exist_ok=True)
            source = Path(getattr(model, "ckpt_path", "yolo11n.pt"))
            if source.is_file() and source.resolve() != dest.resolve():
                shutil.copy2(source, dest)
            return YOLO(str(dest))
        return model

    def _load_yolo(self) -> Any:
        if self._yolo_model is not None:
            return self._yolo_model

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            message = (
                "YOLO CV provider requires optional dependencies. "
                'Install with: pip install -e ".[cv-yolo]"'
            )
            if "libxcb" in str(exc) or "libGL" in str(exc):
                message += (
                    " OpenCV also needs system libraries in Docker "
                    "(libxcb1, libgl1, libglib2.0-0). Rebuild the API image."
                )
            raise RuntimeError(message) from exc

        model_path = self._resolve_model_path()
        try:
            self._yolo_model = YOLO(model_path)
        except RuntimeError as exc:
            if not _looks_like_corrupt_yolo_checkpoint(exc):
                raise
            path = Path(model_path)
            path.unlink(missing_ok=True)
            self._yolo_model = self._download_yolo_weights(path)
        return self._yolo_model

    def detect_objects(self, image_bytes: bytes) -> list[DetectedObject]:
        try:
            model = self._load_yolo()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            results = model.predict(source=image, conf=self.confidence, verbose=False)
        except RuntimeError:
            raise
        except Exception:
            return super().detect_objects(image_bytes)

        detections: list[DetectedObject] = []
        for result in results:
            if result.boxes is None:
                continue
            names = result.names or {}
            for box in result.boxes:
                class_id = int(box.cls.item())
                label = str(names.get(class_id, class_id))
                score = float(box.conf.item())
                detections.append(DetectedObject(label=label, confidence=score))

        detections.sort(key=lambda item: item.confidence, reverse=True)
        if detections:
            return detections[:5]
        return super().detect_objects(image_bytes)
