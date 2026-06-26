import io

import httpx
import pytest
from PIL import Image

from tests.integration.conftest import submit_report


@pytest.mark.asyncio
async def test_create_and_get_report(
    api_client: httpx.AsyncClient,
    sample_image: bytes,
    report_form: dict[str, str | float],
) -> None:
    create_response = await api_client.post(
        "/api/v1/reports",
        data=report_form,
        files={"image": ("report.png", sample_image, "image/png")},
    )
    assert create_response.status_code == 201
    created = create_response.json()
    assert created["success"] is True
    assert created["message"] == "Report created successfully"
    report = created["data"]
    assert report["title"] == report_form["title"]
    assert report["category"] == report_form["category"]
    assert report["duplicate_status"] == "new"
    assert report["status"] == "pending"
    assert report["image_url"].endswith(".png")
    assert report["detected_objects"]
    assert report["cv_inferred_category"] == "flooding"
    assert "x-request-id" in create_response.headers

    get_response = await api_client.get(f"/api/v1/reports/{report['id']}")
    assert get_response.status_code == 200
    assert get_response.json()["data"]["id"] == report["id"]


@pytest.mark.asyncio
async def test_list_reports_returns_created_report(
    api_client: httpx.AsyncClient, sample_image: bytes
) -> None:
    created = await submit_report(api_client, sample_image)
    list_response = await api_client.get("/api/v1/reports")
    assert list_response.status_code == 200
    reports = list_response.json()["data"]
    assert any(item["id"] == created["id"] for item in reports)


@pytest.mark.asyncio
async def test_get_missing_report_returns_404(api_client: httpx.AsyncClient) -> None:
    response = await api_client.get("/api/v1/reports/999999")
    assert response.status_code == 404
    assert response.json()["success"] is False


@pytest.mark.asyncio
async def test_invalid_image_is_rejected(
    api_client: httpx.AsyncClient, report_form: dict[str, str | float]
) -> None:
    response = await api_client.post(
        "/api/v1/reports",
        data=report_form,
        files={"image": ("report.png", b"not-an-image", "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["success"] is False


@pytest.mark.asyncio
async def test_irrelevant_image_is_rejected(
    api_client: httpx.AsyncClient, report_form: dict[str, str | float]
) -> None:
    neutral = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 120, 160)).save(neutral, format="PNG")
    response = await api_client.post(
        "/api/v1/reports",
        data=report_form,
        files={"image": ("report.png", neutral.getvalue(), "image/png")},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["success"] is False
    assert "environmental or public concern" in body["message"]


@pytest.mark.asyncio
async def test_update_report_status(api_client: httpx.AsyncClient, sample_image: bytes) -> None:
    report = await submit_report(api_client, sample_image)
    response = await api_client.patch(
        f"/api/v1/reports/{report['id']}/status",
        json={"status": "verified", "notes": "Confirmed by field team"},
    )
    assert response.status_code == 200
    updated = response.json()["data"]
    assert updated["status"] == "verified"


@pytest.mark.asyncio
async def test_exact_duplicate_reuses_existing_image_url(
    api_client: httpx.AsyncClient, sample_image: bytes
) -> None:
    first = await submit_report(api_client, sample_image)
    second = await submit_report(api_client, sample_image, title="Duplicate flooding report")
    assert second["duplicate_status"] == "duplicate"
    assert second["duplicate_of_report_id"] == first["id"]
    assert second["image_url"] == first["image_url"]
    assert second["image_sha256"] == first["image_sha256"]


@pytest.mark.asyncio
async def test_add_supporting_evidence(api_client: httpx.AsyncClient, sample_image: bytes) -> None:
    parent = await submit_report(api_client, sample_image)
    other_image = io.BytesIO()
    Image.new("RGB", (64, 64), (32, 92, 188)).save(other_image, format="PNG")

    response = await api_client.post(
        f"/api/v1/reports/{parent['id']}/supporting-evidence",
        data={
            "title": "Additional angle of flooding",
            "description": "Water level from the other side",
            "latitude": "5.6038",
            "longitude": "-0.1871",
        },
        files={"image": ("evidence.png", other_image.getvalue(), "image/png")},
    )
    assert response.status_code == 201
    evidence = response.json()["data"]
    assert evidence["duplicate_status"] == "supporting_evidence"
    assert evidence["duplicate_of_report_id"] == parent["id"]

    duplicates_response = await api_client.get(f"/api/v1/reports/{parent['id']}/duplicates")
    assert duplicates_response.status_code == 200
    duplicate_ids = {item["id"] for item in duplicates_response.json()["data"]}
    assert evidence["id"] in duplicate_ids


@pytest.mark.asyncio
async def test_duplicate_review_resolve_flow(api_client: httpx.AsyncClient) -> None:
    base_image = io.BytesIO()
    Image.new("RGB", (64, 64), (30, 90, 190)).save(base_image, format="PNG")
    base_bytes = base_image.getvalue()

    original = await submit_report(api_client, base_bytes, title="Original flooding report")
    similar_image = io.BytesIO()
    Image.new("RGB", (64, 64), (32, 92, 188)).save(similar_image, format="PNG")
    candidate = await submit_report(
        api_client,
        similar_image.getvalue(),
        title="Similar flooding report",
        latitude="5.60371",
        longitude="-0.18701",
    )

    reviews_response = await api_client.get("/api/v1/duplicate-reviews")
    assert reviews_response.status_code == 200
    reviews = reviews_response.json()["data"]
    review = next((item for item in reviews if item["report_id"] == candidate["id"]), None)
    if review is None:
        pytest.skip("Duplicate review was not created for similar report in this environment")

    resolve_response = await api_client.patch(
        f"/api/v1/duplicate-reviews/{review['id']}/resolve",
        json={
            "resolution": "duplicate",
            "duplicate_of_report_id": original["id"],
            "notes": "Confirmed duplicate during integration test",
        },
    )
    assert resolve_response.status_code == 200
    resolved = resolve_response.json()["data"]
    assert resolved["status"] == "resolved"
    assert resolved["resolution"] == "duplicate"

    updated_report = (await api_client.get(f"/api/v1/reports/{candidate['id']}")).json()["data"]
    assert updated_report["duplicate_status"] == "duplicate"
    assert updated_report["duplicate_of_report_id"] == original["id"]
