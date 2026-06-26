from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware import IPWhitelistMiddleware

app = FastAPI()
app.add_middleware(IPWhitelistMiddleware)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/v1/protected")
def protected():
    return {"ok": True}


def test_health_is_public() -> None:
    client = TestClient(app, client=("198.51.100.9", 12345))
    response = client.get("/health")
    assert response.status_code == 200


def test_blocked_ip_gets_403_shape() -> None:
    client = TestClient(app, client=("198.51.100.9", 12345))
    response = client.get("/api/v1/protected")
    assert response.status_code == 403
    assert response.json() == {"success": False, "message": "Access denied", "errors": []}


def test_allowed_ip_access() -> None:
    client = TestClient(app, client=("127.0.0.1", 12345))
    response = client.get("/api/v1/protected")
    assert response.status_code == 200
