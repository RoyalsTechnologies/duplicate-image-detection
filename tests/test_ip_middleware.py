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


def test_health_imports() -> None:
    from app.main import app

    assert app.title


def test_ignores_forwarded_headers_from_untrusted_clients() -> None:
    request = make_request("198.51.100.10", {"x-forwarded-for": "127.0.0.1"})
    assert get_client_ip(request) == "198.51.100.10"


def test_trusts_forwarded_headers_only_from_trusted_proxy() -> None:
    request = make_request("10.0.0.1", {"x-forwarded-for": "197.253.123.104, 10.0.0.1"})
    assert get_client_ip(request) == "197.253.123.104"


def test_allows_docker_host_despite_external_x_forwarded_for(monkeypatch) -> None:
    """ngrok on the host forwards X-Forwarded-For; Docker connects as 172.19.0.1."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.middleware import IPWhitelistMiddleware

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
