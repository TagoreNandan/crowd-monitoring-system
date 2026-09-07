"""Tracks active WebSocket connections, grouped into named channels.

Two channel kinds exist in this phase:
  - "system"       -- WS /ws/system subscribers
  - "job:<job_id>"  -- WS /ws/jobs/{job_id} subscribers

Broadcasting is channel-scoped on purpose: a caller decides explicitly which
audience a message is for, rather than the manager guessing. Later phases
(pushing job.progress/risk.changed events as they happen) call
`broadcast()`; this phase only uses `send_to()` for the initial
ack/status/heartbeat traffic sent to a single freshly-connected socket.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import WebSocket

from ..schemas.ws import WSEnvelope

logger = logging.getLogger(__name__)

SYSTEM_CHANNEL = "system"


class ConnectionManager:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._connections: dict[str, set[WebSocket]] = {}

    @staticmethod
    def system_channel() -> str:
        return SYSTEM_CHANNEL

    @staticmethod
    def job_channel(job_id: str) -> str:
        return f"job:{job_id}"

    async def connect(self, websocket: WebSocket, channel: str) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.setdefault(channel, set()).add(websocket)
        logger.info("WebSocket connected: channel=%s", channel)

    async def disconnect(self, websocket: WebSocket, channel: str) -> None:
        async with self._lock:
            sockets = self._connections.get(channel)
            if sockets is not None:
                sockets.discard(websocket)
                if not sockets:
                    self._connections.pop(channel, None)
        logger.info("WebSocket disconnected: channel=%s", channel)

    async def send_to(self, websocket: WebSocket, envelope: WSEnvelope) -> None:
        await websocket.send_text(envelope.model_dump_json())

    async def broadcast(self, channel: str, envelope: WSEnvelope) -> None:
        async with self._lock:
            sockets = list(self._connections.get(channel, ()))
        payload = envelope.model_dump_json()
        stale: list[WebSocket] = []
        for ws in sockets:
            try:
                await ws.send_text(payload)
            except Exception:
                stale.append(ws)
        if stale:
            async with self._lock:
                for ws in stale:
                    self._connections.get(channel, set()).discard(ws)

    def connection_count(self, channel: str) -> int:
        return len(self._connections.get(channel, ()))
