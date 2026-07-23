from starlette.datastructures import Headers
from starlette.requests import Request
from app.middleware import get_client_ip


class Client:
    def __init__(self, host: str):
        self.host = host
        self.port = 12345


def make_request(host: str, headers: dict[str, str] | None = None) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/reports",
            "headers": Headers(headers or {}).raw,
            "client": (host, 12345),
            "server": ("testserver", 80),
            "scheme": "http",
            "query_string": b"",
        }
    )


def test_ignores_forwarded_headers_from_untrusted_clients() -> None:
    request = make_request("198.51.100.10", {"x-forwarded-for": "127.0.0.1"})
    assert get_client_ip(request) == "198.51.100.10"


def test_trusts_forwarded_headers_only_from_trusted_proxy(monkeypatch) -> None:
    monkeypatch.setattr("app.middleware.settings.trusted_proxy_ips", "10.0.0.1")
    request = make_request("10.0.0.1", {"x-forwarded-for": "197.253.123.104, 10.0.0.1"})
    assert get_client_ip(request) == "197.253.123.104"


def test_whitelist_disabled_permits_all_clients(monkeypatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.middleware import IPWhitelistMiddleware

    monkeypatch.setattr("app.middleware.settings.ip_whitelist_enabled", False)
    monkeypatch.setattr("app.middleware.settings.allowed_ips", "127.0.0.1")

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(IPWhitelistMiddleware)

    @app.get("/api/v1/reports")
    def reports():
        return {"ok": True}

    client = TestClient(app, client=("198.51.100.10", 12345))
    response = client.get("/api/v1/reports")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_whitelist_enabled_with_empty_allowed_ips_denies(monkeypatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.middleware import IPWhitelistMiddleware

    monkeypatch.setattr("app.middleware.settings.ip_whitelist_enabled", True)
    monkeypatch.setattr("app.middleware.settings.allowed_ips", "")

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(IPWhitelistMiddleware)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/v1/reports")
    def reports():
        return {"ok": True}

    client = TestClient(app, client=("127.0.0.1", 12345))
    assert client.get("/health").status_code == 200
    denied = client.get("/api/v1/reports")
    assert denied.status_code == 403
    assert denied.json()["message"] == "Access denied"


def test_whitelist_enabled_enforces_allowed_ips(monkeypatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.middleware import IPWhitelistMiddleware

    monkeypatch.setattr("app.middleware.settings.ip_whitelist_enabled", True)
    monkeypatch.setattr("app.middleware.settings.allowed_ips", "127.0.0.1")

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(IPWhitelistMiddleware)

    @app.get("/docs")
    def docs():
        return {"ok": True}

    @app.get("/api/v1/reports")
    def reports():
        return {"ok": True}

    blocked = TestClient(app, client=("198.51.100.10", 12345))
    assert blocked.get("/docs").status_code == 200
    denied = blocked.get("/api/v1/reports")
    assert denied.status_code == 403
    assert denied.json()["message"] == "Access denied"

    allowed = TestClient(app, client=("127.0.0.1", 12345))
    assert allowed.get("/api/v1/reports").status_code == 200


def test_allows_docker_host_despite_external_x_forwarded_for(monkeypatch) -> None:
    """ngrok on the host forwards X-Forwarded-For; Docker connects as 172.19.0.1."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.middleware import IPWhitelistMiddleware

    monkeypatch.setattr("app.middleware.settings.ip_whitelist_enabled", True)
    monkeypatch.setattr("app.middleware.settings.allowed_ips", "172.16.0.0/12")
    monkeypatch.setattr("app.middleware.settings.trusted_proxy_ips", "172.16.0.0/12")

    app = FastAPI()
    app.add_middleware(IPWhitelistMiddleware)

    @app.post("/api/v1/reports")
    def reports():
        return {"ok": True}

    client = TestClient(app, client=("172.19.0.1", 57832))
    response = client.post("/api/v1/reports", headers={"x-forwarded-for": "3.125.223.134"})
    assert response.status_code == 200
