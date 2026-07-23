import ipaddress
import logging
from uuid import uuid4
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from app.config import settings

logger = logging.getLogger(__name__)

# Exact paths that skip IP whitelisting (health checks + OpenAPI UI).
WHITELIST_EXEMPT_PATHS = frozenset({"/", "/health", "/openapi.json"})
# Prefixes for the interactive docs UIs (and their static assets).
WHITELIST_EXEMPT_PREFIXES = ("/docs", "/redoc")


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
        path = request.url.path
        if path in WHITELIST_EXEMPT_PATHS or path.startswith(WHITELIST_EXEMPT_PREFIXES):
            response = await call_next(request)
            response.headers["x-request-id"] = request_id
            return response

        direct_ip = request.client.host if request.client else ""
        client_ip = get_client_ip(request)
        request.state.client_ip = client_ip or direct_ip

        # Optional: enable with IP_WHITELIST_ENABLED=true and list CIDRs in ALLOWED_IPS.
        # When enabled with an empty ALLOWED_IPS, deny non-exempt routes (fail closed).
        if settings.ip_whitelist_enabled:
            if not settings.allowed_ip_ranges:
                logger.warning(
                    "IP_WHITELIST_ENABLED is true but ALLOWED_IPS is empty; "
                    "denying non-exempt clients"
                )
                allowed = False
            else:
                allowed = _in_ranges(direct_ip, settings.allowed_ip_ranges) or _in_ranges(
                    client_ip, settings.allowed_ip_ranges
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

        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response
