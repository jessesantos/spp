from celery import Celery
from app.infra.config import settings

celery_app = Celery(
    "spp",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.infra.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="America/Sao_Paulo",
    enable_utc=True,
    task_track_started=True,
    task_default_queue="training",
)

# Importing celery_beat attaches the beat schedule. We swallow ImportError
# so a minimal worker that does not need the scheduler still starts.
try:  # pragma: no cover - side-effect only
    from app.infra import celery_beat  # noqa: F401
except Exception:  # noqa: BLE001
    pass
