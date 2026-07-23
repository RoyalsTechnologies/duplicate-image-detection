"""Groq vision API client for environmental concern image narration."""

from __future__ import annotations

import base64
import logging
import re
from typing import Any

import httpx

from app.config import settings
from app.images import prepare_image_for_vision_api

logger = logging.getLogger(__name__)

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

NARRATION_PROMPT = """You describe photos of environmental and public infrastructure concerns.
Write 2-4 factual sentences about what is visible: hazards, damage, waste, water, smoke, blocked drains, etc.
Do not guess location, cause, or severity beyond what the image shows.
If the image is unclear or not a real-world outdoor/public scene, say so briefly."""


def _build_user_prompt(
    *,
    category: str | None = None,
    detected_objects: list[dict[str, Any]] | None = None,
) -> str:
    parts = [NARRATION_PROMPT]
    if category:
        parts.append(f"Reporter-selected category: {category.replace('_', ' ')}.")
    if detected_objects:
        hints = ", ".join(
            f"{item.get('label', 'object')} ({float(item.get('confidence', 0)):.0%})"
            for item in detected_objects[:5]
            if item.get("label")
        )
        if hints:
            parts.append(f"On-device vision hints (may be incomplete): {hints}.")
    return "\n".join(parts)


def _extract_message_content(content: object) -> str:
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        text_parts = [
            str(block.get("text", "")).strip()
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
        ]
        text = " ".join(text_parts).strip()
    else:
        text = str(content).strip()
    # Qwen thinking models may still emit <think>…</think> blocks.
    return _THINK_BLOCK_RE.sub("", text).strip()


async def narrate_image_with_groq(
    image_bytes: bytes,
    *,
    mime_type: str = "image/jpeg",
    category: str | None = None,
    detected_objects: list[dict[str, Any]] | None = None,
) -> str | None:
    """Return a short natural-language description of an image, or None on failure."""
    if not settings.groq_narration_available:
        return None

    try:
        vision_bytes, vision_mime_type = prepare_image_for_vision_api(image_bytes, mime_type)
    except Exception:
        logger.warning("Could not prepare image for Groq vision", exc_info=True)
        return None

    encoded = base64.b64encode(vision_bytes).decode("ascii")
    payload = {
        "model": settings.groq_vision_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _build_user_prompt(
                            category=category,
                            detected_objects=detected_objects,
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{vision_mime_type};base64,{encoded}"},
                    },
                ],
            }
        ],
        "temperature": 0.2,
        "max_completion_tokens": settings.groq_vision_max_tokens,
        # Disable Qwen reasoning so the UI gets a clean caption, not <think> text.
        "reasoning_effort": "none",
    }

    try:
        async with httpx.AsyncClient(timeout=settings.groq_vision_timeout_seconds) as client:
            response = await client.post(
                GROQ_CHAT_COMPLETIONS_URL,
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                json=payload,
            )
            if response.is_error:
                logger.warning(
                    "Groq image narration failed: %s %s",
                    response.status_code,
                    response.text[:300],
                )
                return None
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            text = _extract_message_content(content)
            return text or None
    except Exception:
        logger.warning("Groq image narration failed", exc_info=True)
        return None
