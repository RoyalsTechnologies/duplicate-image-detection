import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.computer_vision.factory import validate_cv_provider_dependencies, warm_up_cv_client
from app.config import settings
from app.exceptions import register_exception_handlers
from app.middleware import IPWhitelistMiddleware
from app.routes import router
from app.ui_routes import router as ui_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_cv_provider_dependencies()
    logger.info("CV provider ready: %s", settings.cv_provider)

    async def _warmup_in_background() -> None:
        try:
            await asyncio.to_thread(warm_up_cv_client)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Unexpected CV warmup task failure")

    warmup_task = asyncio.create_task(_warmup_in_background())
    yield
    warmup_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await warmup_task


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(IPWhitelistMiddleware)
register_exception_handlers(app)
app.include_router(ui_router)
app.include_router(router)
settings.local_storage_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.local_storage_dir), name="uploads")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
