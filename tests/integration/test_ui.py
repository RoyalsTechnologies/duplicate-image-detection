from fastapi.testclient import TestClient

from app.main import app


def test_home_page_is_public() -> None:
    client = TestClient(app, client=("198.51.100.9", 12345))
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Report an environmental concern" in response.text
    assert 'id="report-form"' in response.text


def test_home_page_accessible_from_allowed_ip() -> None:
    client = TestClient(app, client=("127.0.0.1", 12345))
    response = client.get("/")
    assert response.status_code == 200
