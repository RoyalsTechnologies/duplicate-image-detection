import logging
from redis.asyncio import Redis
from app.config import settings

logger = logging.getLogger(__name__)
redis_client: Redis | None = (
    Redis.from_url(settings.redis_url, decode_responses=True) if settings.redis_url else None
)


async def get_cached_report_for_image_hash(image_sha256: str) -> int | None:
    if redis_client is None:
        return None
    try:
        value = await redis_client.get(f"image-sha256:{image_sha256}")
        return int(value) if value else None
    except Exception:
        logger.warning("Redis read failed; continuing without cache", exc_info=True)
        return None


async def cache_image_hash(image_sha256: str, report_id: int, ttl_seconds: int = 86_400) -> None:
    if redis_client is None:
        return
    try:
        await redis_client.set(f"image-sha256:{image_sha256}", str(report_id), ex=ttl_seconds)
    except Exception:
        logger.warning("Redis write failed; continuing without cache", exc_info=True)


async def increment_rate_limit(key: str, ttl_seconds: int = 60) -> int:
    if redis_client is None:
        return 1
    try:
        value = await redis_client.incr(key)
        if value == 1:
            await redis_client.expire(key, ttl_seconds)
        return int(value)
    except Exception:
        logger.warning("Redis rate limit failed open", exc_info=True)
        return 1
