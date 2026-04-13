"""WebSocket endpoint streaming fresh predictions every 60 seconds."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.infra.dependencies import prediction_service

router = APIRouter()


@router.websocket("/ws/live/{ticker}")
async def live(websocket: WebSocket, ticker: str) -> None:
    await websocket.accept()
    ticker = ticker.upper().strip()
    service = prediction_service()
    try:
        while True:
            result = await service.predict(ticker, days=1)
            await websocket.send_json(result.model_dump(mode="json"))
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        return
    except Exception:  # noqa: BLE001
        await websocket.close(code=1011)
