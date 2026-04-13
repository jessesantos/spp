"""Custom exceptions and FastAPI handlers returning structured JSON."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class DomainError(Exception):
    """Base class for domain-level errors (mapped to HTTP 4xx)."""

    status_code: int = 400

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code


class NotFoundError(DomainError):
    status_code = 404


class UpstreamError(DomainError):
    status_code = 502


def _error_body(status: int, code: str, message: str, request_id: str | None) -> dict[str, object]:
    body: dict[str, object] = {"error": {"code": code, "message": message, "status": status}}
    if request_id:
        body["error"]["request_id"] = request_id  # type: ignore[index]
    return body


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    async def _domain_error(request: Request, exc: DomainError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(
                exc.status_code,
                exc.__class__.__name__,
                exc.message,
                getattr(request.state, "request_id", None),
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(
                exc.status_code,
                "HTTPException",
                str(exc.detail),
                getattr(request.state, "request_id", None),
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "ValidationError",
                    "message": "invalid request",
                    "status": 422,
                    "details": exc.errors(),
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )
