"""Tests for the per-ticker Celery training lock.

Validam que chamadas concorrentes para o mesmo ticker sao serializadas
e que tickers diferentes nao bloqueiam entre si.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _point_models_dir_at_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Aponta ``settings.models_dir`` para o tmp_path antes do import."""
    from app.infra.config import settings

    monkeypatch.setattr(settings, "models_dir", str(tmp_path))


def test_same_ticker_calls_are_serialized(tmp_path: Path) -> None:
    from app.infra.tasks import _ticker_training_lock

    order: list[str] = []
    a_in, a_out = threading.Event(), threading.Event()
    b_in, b_out = threading.Event(), threading.Event()

    def worker_a() -> None:
        with _ticker_training_lock("PETR4"):
            order.append("a_start")
            a_in.set()
            # Segura o lock ate o teste liberar
            a_out.wait(timeout=3.0)
            order.append("a_end")

    def worker_b() -> None:
        # Espera A ja ter pegado o lock antes de B tentar
        a_in.wait(timeout=3.0)
        with _ticker_training_lock("PETR4"):
            order.append("b_start")
            b_in.set()
            b_out.set()
            order.append("b_end")

    thread_a = threading.Thread(target=worker_a)
    thread_b = threading.Thread(target=worker_b)
    thread_a.start()
    thread_b.start()

    # Antes de liberar A, B nao deve ter entrado
    a_in.wait(timeout=3.0)
    time.sleep(0.05)
    assert "b_start" not in order, "B entrou antes de A liberar o lock"

    # Libera A
    a_out.set()
    thread_a.join(timeout=3.0)
    thread_b.join(timeout=3.0)

    # A deve ter comecado e terminado antes de B comecar
    assert order.index("a_end") < order.index("b_start")


def test_different_tickers_do_not_block(tmp_path: Path) -> None:
    from app.infra.tasks import _ticker_training_lock

    b_entered = threading.Event()

    def hold_a() -> None:
        with _ticker_training_lock("PETR4"):
            # Segura por 2s
            time.sleep(2.0)

    def try_b() -> None:
        with _ticker_training_lock("VALE3"):
            b_entered.set()

    thread_a = threading.Thread(target=hold_a)
    thread_b = threading.Thread(target=try_b)
    thread_a.start()
    time.sleep(0.1)
    thread_b.start()

    # B deve entrar mesmo com A segurando PETR4 (lock e por ticker)
    assert b_entered.wait(timeout=1.5), "VALE3 foi bloqueado por lock de PETR4"

    thread_a.join(timeout=5.0)
    thread_b.join(timeout=5.0)


def test_lock_sanitizes_ticker_name(tmp_path: Path) -> None:
    from app.infra.tasks import _ticker_training_lock

    # Caracteres perigosos sao retirados antes de virar nome de arquivo.
    # Nao deve criar diretorios fora do models_dir.
    with _ticker_training_lock("../../etc/passwd"):
        pass
    files = list(tmp_path.glob("*.lock"))
    assert files, "lock file nao foi criado"
    for f in files:
        assert ".." not in f.name
        assert "/" not in f.name
