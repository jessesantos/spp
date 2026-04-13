"""slowapi-based rate limiter factory.

The limiter is created once and attached to ``app.state.limiter`` so
routes can apply per-endpoint decorators via ``@limiter.limit(...)``.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address


def build_limiter(default: str = "60/minute") -> Limiter:
    return Limiter(key_func=get_remote_address, default_limits=[default])
