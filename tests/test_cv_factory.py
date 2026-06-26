import sys
from unittest.mock import MagicMock

import pytest

from app.computer_vision import (
    DetectedObject,
    EmbeddingComputerVisionClient,
    LocalComputerVisionClient,
    SUPPORTED_CV_PROVIDERS,
    YoloV11ComputerVisionClient,
    build_cv_client,
)
from app.computer_vision.base import EMBEDDING_DIM
from app.computer_vision.factory import build_cv_client as factory_build_cv_client
from app.config import get_settings
from app.models import ReportCategory


@pytest.fixture
def cv_settings():
    import app.computer_vision.factory as factory_module

    original_provider = factory_module.settings.cv_provider
    yield factory_module.settings
    factory_module.settings.cv_provider = original_provider
    get_settings.cache_clear()


@pytest.mark.parametrize(
    ("provider", "expected_type"),
    [
        ("local", LocalComputerVisionClient),
        ("embedding", EmbeddingComputerVisionClient),
        ("yolov11", YoloV11ComputerVisionClient),
    ],
)
def test_build_cv_client_returns_configured_provider(
    cv_settings, provider: str, expected_type: type
) -> None:
    cv_settings.cv_provider = provider
    client = build_cv_client()
    assert isinstance(client, expected_type)


def test_build_cv_client_rejects_unknown_provider(cv_settings) -> None:
    cv_settings.cv_provider = "unknown-provider"
    with pytest.raises(ValueError, match="Unsupported CV provider"):
        build_cv_client()


def test_supported_providers_include_expected_values() -> None:
    assert SUPPORTED_CV_PROVIDERS == {"local", "embedding", "yolov11"}


def test_embedding_client_inherits_local_detection(image_bytes) -> None:
    client = EmbeddingComputerVisionClient()
    data = image_bytes((30, 90, 190))
    assert client.detect_objects(data)
    assert client.recognize_concern_type(data) == ReportCategory.flooding


def test_embedding_client_fits_embedding_dimensions() -> None:
    client = EmbeddingComputerVisionClient()
    assert len(client._fit_embedding_dim([0.1] * EMBEDDING_DIM)) == EMBEDDING_DIM
    assert len(client._fit_embedding_dim([0.2] * 600)) == EMBEDDING_DIM
    assert client._fit_embedding_dim([0.3] * 128) == [0.3] * 128 + [0.0] * (EMBEDDING_DIM - 128)


def test_embedding_client_falls_back_to_local_embedding_when_clip_fails(
    image_bytes, monkeypatch
) -> None:
    client = EmbeddingComputerVisionClient()
    local_embedding = LocalComputerVisionClient().image_embedding(image_bytes((20, 80, 180)))

    monkeypatch.setitem(sys.modules, "torch", MagicMock())

    def fail_clip() -> tuple[object, object]:
        raise OSError("clip unavailable")

    monkeypatch.setattr(client, "_load_clip", fail_clip)
    assert client.image_embedding(image_bytes((20, 80, 180))) == local_embedding


def test_embedding_client_raises_helpful_error_when_open_clip_missing(monkeypatch) -> None:
    client = EmbeddingComputerVisionClient()
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "open_clip":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match='pip install -e ".\\[cv-embedding\\]"'):
        client._load_clip()


def test_yolo_client_maps_predictions_to_detected_objects(image_bytes, monkeypatch) -> None:
    client = YoloV11ComputerVisionClient()

    class FakeCls:
        def item(self) -> int:
            return 0

    class FakeConf:
        def item(self) -> float:
            return 0.91

    class FakeBox:
        cls = FakeCls()
        conf = FakeConf()

    class FakeResult:
        boxes = [FakeBox()]
        names = {0: "bottle"}

    class FakeYolo:
        def predict(self, **kwargs):
            return [FakeResult()]

    monkeypatch.setattr(client, "_load_yolo", lambda: FakeYolo())
    detections = client.detect_objects(image_bytes((130, 85, 35)))
    assert detections == [DetectedObject("bottle", 0.91)]
    assert client.category_from_detected_objects(detections) == ReportCategory.refuse_dump


def test_yolo_client_falls_back_to_local_detection_on_non_runtime_errors(
    image_bytes, monkeypatch
) -> None:
    client = YoloV11ComputerVisionClient()

    class BrokenYolo:
        def predict(self, **kwargs):
            raise ValueError("model failed")

    monkeypatch.setattr(client, "_load_yolo", lambda: BrokenYolo())
    local_detections = LocalComputerVisionClient().detect_objects(image_bytes((30, 90, 190)))
    assert client.detect_objects(image_bytes((30, 90, 190))) == local_detections


def test_yolo_client_reraises_runtime_errors_from_prediction(image_bytes, monkeypatch) -> None:
    client = YoloV11ComputerVisionClient()

    class BrokenYolo:
        def predict(self, **kwargs):
            raise RuntimeError("model failed")

    monkeypatch.setattr(client, "_load_yolo", lambda: BrokenYolo())
    with pytest.raises(RuntimeError, match="model failed"):
        client.detect_objects(image_bytes((30, 90, 190)))


def test_yolo_client_removes_partial_weights_before_load(tmp_path) -> None:
    weights = tmp_path / "yolo11n.pt"
    weights.write_bytes(b"partial")

    client = YoloV11ComputerVisionClient(model_path=str(weights))
    assert client._resolve_model_path() == str(weights)
    assert not weights.exists()


def test_yolo_client_raises_helpful_error_when_ultralytics_missing(monkeypatch) -> None:
    client = YoloV11ComputerVisionClient()
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ultralytics":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match='pip install -e ".\\[cv-yolo\\]"'):
        client._load_yolo()


def test_factory_build_cv_client_uses_settings_device_and_model_names(cv_settings) -> None:
    cv_settings.cv_provider = "yolov11"
    cv_settings.cv_device = "cpu"
    cv_settings.cv_yolo_model = "custom.pt"
    cv_settings.cv_yolo_confidence = 0.4
    cv_settings.cv_embedding_model = "ViT-B-32"
    cv_settings.cv_embedding_pretrained = "openai"

    client = factory_build_cv_client()
    assert isinstance(client, YoloV11ComputerVisionClient)
    assert client.device == "cpu"
    assert client.model_path == "custom.pt"
    assert client.confidence == 0.4
    assert client.model_name == "ViT-B-32"
    assert client.pretrained == "openai"
