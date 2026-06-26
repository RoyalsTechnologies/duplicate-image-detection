import ipaddress
import logging
from uuid import uuid4
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from app.config import settings

logger = logging.getLogger(__name__)


def _in_ranges(ip: str, ranges: list[str]) -> bool:
    try:
        parsed = ipaddress.ip_address(ip)
        return any(parsed in ipaddress.ip_network(item, strict=False) for item in ranges)
    except ValueError:
        return False


def get_client_ip(request: Request) -> str:
    direct_client_ip = request.client.host if request.client else ""
    if direct_client_ip and _in_ranges(direct_client_ip, settings.trusted_proxy_ranges):
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()
    return direct_client_ip


class IPWhitelistMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        request_id = request.headers.get("x-request-id") or uuid4().hex
        request.state.request_id = request_id
        if request.url.path in {"/health", "/"}:
            response = await call_next(request)
            response.headers["x-request-id"] = request_id
            return response

        direct_ip = request.client.host if request.client else ""
        client_ip = get_client_ip(request)
        allowed = settings.allowed_ip_ranges and (
            _in_ranges(direct_ip, settings.allowed_ip_ranges)
            or _in_ranges(client_ip, settings.allowed_ip_ranges)
        )
        if not allowed:
            logger.warning(
                "Rejected request from non-whitelisted IP",
                extra={
                    "source_ip": client_ip,
                    "direct_ip": direct_ip,
                    "route": request.url.path,
                    "request_id": request_id,
                },
            )
            return JSONResponse(
                status_code=403,
                content={"success": False, "message": "Access denied", "errors": []},
                headers={"x-request-id": request_id},
            )
        request.state.client_ip = client_ip
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response
