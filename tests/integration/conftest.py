import io
import os
from collections.abc import AsyncIterator

import httpx
import pytest
from alembic import command
from alembic.config import Config
from PIL import Image
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# Integration tests use the configured database and reset schema once per session.
os.environ.setdefault(
    "INTEGRATION_DATABASE_URL",
    "postgresql+asyncpg://did:did@localhost:5432/did",
)
os.environ["DATABASE_URL"] = os.environ["INTEGRATION_DATABASE_URL"]
os.environ.setdefault("REDIS_URL", "")

from app.config import get_settings

get_settings.cache_clear()
settings = get_settings()

from app import database  # noqa: E402

database.engine = create_async_engine(settings.database_url, pool_pre_ping=True)
database.AsyncSessionLocal = async_sessionmaker(
    database.engine,
    expire_on_commit=False,
    class_=AsyncSession,
)

from app.main import app  # noqa: E402

pytestmark = pytest.mark.integration

TRUNCATE_TABLES = "TRUNCATE audit_logs, duplicate_reviews, reports RESTART IDENTITY CASCADE"
RESET_SCHEMA_STATEMENTS = (
    "DROP SCHEMA IF EXISTS public CASCADE",
    "CREATE SCHEMA public",
    "GRANT ALL ON SCHEMA public TO public",
    "GRANT ALL ON SCHEMA public TO did",
    "CREATE EXTENSION IF NOT EXISTS postgis",
    "CREATE EXTENSION IF NOT EXISTS vector",
)


def _sync_database_url() -> str:
    return settings.database_url.replace("+asyncpg", "+psycopg")


def _image_bytes(
    color: tuple[int, int, int] = (30, 90, 190), size: tuple[int, int] = (64, 64)
) -> bytes:
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _reset_schema() -> None:
    engine = create_engine(_sync_database_url(), isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as connection:
            for statement in RESET_SCHEMA_STATEMENTS:
                connection.execute(text(statement))
    finally:
        engine.dispose()


def _truncate_tables() -> None:
    engine = create_engine(_sync_database_url())
    try:
        with engine.begin() as connection:
            connection.execute(text(TRUNCATE_TABLES))
    finally:
        engine.dispose()


@pytest.fixture(scope="session")
def postgres_available() -> None:
    engine = create_engine(_sync_database_url())
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:
        pytest.skip(
            "Postgres is not available for integration tests. "
            "Start it with: docker compose up -d postgres"
        )
    finally:
        engine.dispose()


@pytest.fixture(scope="session")
def migrated_db(postgres_available: None) -> None:
    _reset_schema()

    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", settings.database_url)
    command.upgrade(alembic_cfg, "head")


@pytest.fixture(scope="session")
def integration_engine(migrated_db: None) -> AsyncEngine:
    return database.engine


@pytest.fixture
async def clean_db(integration_engine: AsyncEngine) -> AsyncIterator[None]:
    await database.engine.dispose()
    _truncate_tables()
    yield
    _truncate_tables()
    await database.engine.dispose()


@pytest.fixture
async def api_client(clean_db: None, tmp_path) -> AsyncIterator[httpx.AsyncClient]:
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    settings.local_storage_dir = upload_dir

    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.fixture
def sample_image() -> bytes:
    return _image_bytes()


@pytest.fixture
def report_form() -> dict[str, str | float]:
    return {
        "title": "Flooded gutter on Main Street",
        "description": "Standing water blocking the drain",
        "category": "flooding",
        "latitude": "5.6037",
        "longitude": "-0.1870",
    }


async def submit_report(
    client: httpx.AsyncClient,
    image_bytes: bytes,
    *,
    filename: str = "report.png",
    content_type: str = "image/png",
    **form_overrides: str | float,
) -> dict:
    form = {
        "title": "Flooded gutter on Main Street",
        "description": "Standing water blocking the drain",
        "category": "flooding",
        "latitude": "5.6037",
        "longitude": "-0.1870",
    }
    form.update({key: str(value) for key, value in form_overrides.items()})
    response = await client.post(
        "/api/v1/reports",
        data=form,
        files={"image": (filename, image_bytes, content_type)},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["success"] is True
    return body["data"]
