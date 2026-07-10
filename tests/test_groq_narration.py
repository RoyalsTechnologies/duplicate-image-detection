import asyncio
import io
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from PIL import Image

from app.images import prepare_image_for_vision_api
from app.llm.groq_vision import _extract_message_content, narrate_image_with_groq


def _tiny_jpeg_bytes() -> bytes:
    image = Image.new("RGB", (32, 32), color=(64, 128, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def groq_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.llm.groq_vision.settings.groq_api_key", "gsk_test_key")
    monkeypatch.setattr("app.llm.groq_vision.settings.groq_vision_enabled", True)


def test_narrate_image_with_groq_returns_none_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.llm.groq_vision.settings.groq_api_key", None)

    result = asyncio.run(narrate_image_with_groq(b"image-bytes", mime_type="image/png"))
    assert result is None


def test_narrate_image_with_groq_returns_caption(groq_settings: None) -> None:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(
        200,
        request=request,
        json={
            "choices": [
                {
                    "message": {
                        "content": "Standing water covers the road near a blocked drain."
                    }
                }
            ]
        },
    )

    with patch("app.llm.groq_vision.httpx.AsyncClient") as client_cls:
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.__aexit__.return_value = None
        client.post = AsyncMock(return_value=response)
        client_cls.return_value = client

        result = asyncio.run(
            narrate_image_with_groq(
                _tiny_jpeg_bytes(),
                mime_type="image/jpeg",
                category="flooding",
                detected_objects=[{"label": "flooding", "confidence": 0.82}],
            )
        )

    assert result == "Standing water covers the road near a blocked drain."
    request = client.post.await_args
    assert request is not None
    payload = request.kwargs["json"]
    assert payload["model"] == "meta-llama/llama-4-scout-17b-16e-instruct"
    assert "flooding" in payload["messages"][0]["content"][0]["text"]


def test_narrate_image_with_groq_returns_none_on_http_error(groq_settings: None) -> None:
    with patch("app.llm.groq_vision.httpx.AsyncClient") as client_cls:
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.__aexit__.return_value = None
        client.post = AsyncMock(side_effect=httpx.HTTPError("network down"))
        client_cls.return_value = client

        result = asyncio.run(narrate_image_with_groq(_tiny_jpeg_bytes()))
        assert result is None


def test_extract_message_content_handles_text_blocks() -> None:
    content = [{"type": "text", "text": "Flooded street."}]
    assert _extract_message_content(content) == "Flooded street."


def test_prepare_image_for_vision_api_compresses_large_png() -> None:
    image = Image.new("RGB", (3000, 2000), color=(20, 120, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    prepared, mime_type = prepare_image_for_vision_api(buffer.getvalue(), "image/png")

    assert mime_type == "image/jpeg"
    assert len(prepared) < 2_800_000
