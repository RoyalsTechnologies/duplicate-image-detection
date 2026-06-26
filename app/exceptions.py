import logging
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class NotFoundError(Exception):
    pass


class BadRequestError(Exception):
    pass


class ConflictError(Exception):
    pass


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _error(
    status_code: int, message: str, errors: list | None = None, request_id: str | None = None
) -> JSONResponse:
    headers = {"x-request-id": request_id} if request_id else None
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "message": message, "errors": errors or []},
        headers=headers,
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(NotFoundError)
    async def not_found_handler(request: Request, exc: NotFoundError) -> JSONResponse:
        return _error(404, str(exc), request_id=_request_id(request))

    @app.exception_handler(BadRequestError)
    async def bad_request_handler(request: Request, exc: BadRequestError) -> JSONResponse:
        return _error(400, str(exc), request_id=_request_id(request))

    @app.exception_handler(ConflictError)
    async def conflict_handler(request: Request, exc: ConflictError) -> JSONResponse:
        return _error(409, str(exc), request_id=_request_id(request))

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(
            400, "Validation failed", errors=exc.errors(), request_id=_request_id(request)
        )

    @app.exception_handler(Exception)
    async def unexpected_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unexpected server error",
            extra={"route": request.url.path, "request_id": _request_id(request)},
        )
        return _error(500, "Internal server error", request_id=_request_id(request))
