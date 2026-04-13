"""FastAPI application entrypoint - wiring middleware, routers, metrics."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.api.errors import register_exception_handlers
from app.api.routes import router as api_router
from app.api.websocket import router as ws_router
from app.infra.config import settings
from app.infra.logging import configure_logging, get_logger
from app.infra.middleware import RequestIDMiddleware, SecureHeadersMiddleware
from app.infra.rate_limit import build_limiter

logger = get_logger(__name__)


RECONCILE_INTERVAL_SECONDS: int = 3600


def _run_alembic_upgrade() -> None:
    """Run ``alembic upgrade head`` programmatically; best-effort."""
    try:
        from alembic import command
        from alembic.config import Config

        ini_path = Path(__file__).resolve().parents[1] / "alembic.ini"
        if not ini_path.exists():
            return
        cfg = Config(str(ini_path))
        cfg.set_main_option("sqlalchemy.url", settings.database_url)
        command.upgrade(cfg, "head")
    except Exception as exc:  # noqa: BLE001 - DB may still be booting
        logger.warning("alembic.upgrade_failed", error=str(exc))


async def _reconcile_loop() -> None:
    from app.infra.dependencies import reconciliation_service

    while True:
        try:
            await asyncio.sleep(RECONCILE_INTERVAL_SECONDS)
            checked, resolved = await reconciliation_service().reconcile()
            logger.info("spp.reconcile", checked=checked, resolved=resolved)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("spp.reconcile_failed", error=str(exc))


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    configure_logging(settings.log_level)
    logger.info("spp.startup", env=settings.env)
    await asyncio.to_thread(_run_alembic_upgrade)
    reconcile_task = asyncio.create_task(_reconcile_loop())
    try:
        yield
    finally:
        reconcile_task.cancel()
        try:
            await reconcile_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        logger.info("spp.shutdown")


app = FastAPI(
    title="SPP API",
    version="3.0.0",
    description="Stock Price Predictor - LSTM + Claude sentiment",
    lifespan=lifespan,
)

limiter = build_limiter(default=settings.rate_limit_default)
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(SecureHeadersMiddleware)
app.add_middleware(RequestIDMiddleware)

register_exception_handlers(app)


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "code": "RateLimitExceeded",
                "message": "too many requests",
                "status": 429,
                "request_id": getattr(request.state, "request_id", None),
            }
        },
    )


app.include_router(api_router, prefix="/api")
app.include_router(ws_router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "3.0.0"}


@app.get("/", tags=["meta"])
async def root() -> dict[str, str]:
    return {"name": "SPP API", "docs": "/docs"}


# Prometheus metrics - /metrics
try:  # pragma: no cover - optional dependency wiring
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
except Exception:  # noqa: BLE001
    logger.warning("prometheus_instrumentator.disabled")
