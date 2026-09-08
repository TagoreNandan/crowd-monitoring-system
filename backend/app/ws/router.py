"""WebSocket endpoints: WS /ws/system and WS /ws/jobs/{job_id}.

Foundation only, per Phase 1 scope: connection ack, one status snapshot on
connect, a server-initiated heartbeat, minimal client ping/pong handling,
and clean disconnect cleanup. No per-frame metrics, detection arrays, or
binary video ever go over these sockets -- only small JSON envelopes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from crowd_monitoring.system_status import GLOBAL_STATUS

from ..schemas.ws import ConnectionAckData, JobStatusData, SystemStatusData, WSEnvelope, WSEventType
from .manager import ConnectionManager

logger = logging.getLogger(__name__)

router = APIRouter()

HEARTBEAT_INTERVAL_S = 20.0


def _system_status_envelope() -> WSEnvelope:
    snapshot = GLOBAL_STATUS.snapshot().to_dict()
    data = SystemStatusData(
        state=snapshot["state"],
        model_loaded=snapshot["model_loaded"],
        model_name=snapshot["model_name"],
        model_weights=snapshot["model_weights"],
        device=snapshot["device"],
        fps=snapshot["fps"],
        frames_processed=snapshot["frames_processed"],
        error_count=snapshot["error_count"],
    )
    return WSEnvelope(type=WSEventType.SYSTEM_STATUS, data=data.model_dump())


async def _heartbeat_loop(websocket: WebSocket, job_id: Optional[str]) -> None:
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_S)
        await websocket.send_text(WSEnvelope(type=WSEventType.PING, job_id=job_id).model_dump_json())


async def _receive_loop(websocket: WebSocket, job_id: Optional[str]) -> None:
    """Handles client -> server traffic. This phase has no client-issued
    commands other than replying to a client-sent ping -- anything else is
    ignored rather than closing the connection, so the contract can grow in
    later phases without breaking already-connected clients."""
    while True:
        message = await websocket.receive_text()
        if _is_ping(message):
            await websocket.send_text(WSEnvelope(type=WSEventType.PONG, job_id=job_id).model_dump_json())


def _is_ping(message: str) -> bool:
    stripped = message.strip()
    if stripped.lower() == "ping":
        return True
    try:
        parsed = json.loads(stripped)
    except (ValueError, TypeError):
        return False
    return isinstance(parsed, dict) and str(parsed.get("type", "")).lower() == "ping"


async def _serve(
    websocket: WebSocket,
    manager: ConnectionManager,
    channel: str,
    job_id: Optional[str],
    initial_envelopes: list[WSEnvelope],
) -> None:
    await manager.connect(websocket, channel)
    try:
        for envelope in initial_envelopes:
            await manager.send_to(websocket, envelope)

        receive_task = asyncio.create_task(_receive_loop(websocket, job_id))
        heartbeat_task = asyncio.create_task(_heartbeat_loop(websocket, job_id))
        done, pending = await asyncio.wait({receive_task, heartbeat_task}, return_when=asyncio.FIRST_EXCEPTION)
        for task in pending:
            task.cancel()
        for task in done:
            task.result()  # re-raise (typically WebSocketDisconnect)
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket, channel)


@router.websocket("/ws/system")
async def ws_system(websocket: WebSocket) -> None:
    manager: ConnectionManager = websocket.app.state.app_state.connection_manager
    channel = manager.system_channel()
    ack = WSEnvelope(type=WSEventType.CONNECTION_ACK, data=ConnectionAckData(channel=channel).model_dump())
    await _serve(websocket, manager, channel, job_id=None, initial_envelopes=[ack, _system_status_envelope()])


@router.websocket("/ws/jobs/{job_id}")
async def ws_job(websocket: WebSocket, job_id: str) -> None:
    app_state = websocket.app.state.app_state
    manager: ConnectionManager = app_state.connection_manager
    job_store = app_state.job_store

    record = job_store.get(job_id)
    if record is None:
        # No channel to join -- refuse the handshake with an application
        # close code rather than accepting a socket for a job that doesn't
        # exist.
        await websocket.close(code=4404, reason="job not found")
        return

    channel = manager.job_channel(job_id)
    ack = WSEnvelope(type=WSEventType.CONNECTION_ACK, job_id=job_id, data=ConnectionAckData(channel=channel).model_dump())
    status_envelope = WSEnvelope(
        type=WSEventType.JOB_STATUS,
        job_id=job_id,
        data=JobStatusData(status=record.status.value, progress=record.progress, message=record.message).model_dump(),
    )
    await _serve(websocket, manager, channel, job_id=job_id, initial_envelopes=[ack, status_envelope])


@router.websocket("/ws/feeds/{feed_id}")
async def ws_feed(websocket: WebSocket, feed_id: str) -> None:
    app_state = websocket.app.state.app_state
    manager: ConnectionManager = app_state.connection_manager

    channel = manager.feed_channel(feed_id)
    ack = WSEnvelope(type=WSEventType.CONNECTION_ACK, job_id=feed_id, data=ConnectionAckData(channel=channel).model_dump())
    status_envelope = WSEnvelope(
        type=WSEventType.FEED_STATUS,
        job_id=feed_id,
        data={"feed_id": feed_id, "status": "connected"},
    )
    await _serve(websocket, manager, channel, job_id=feed_id, initial_envelopes=[ack, status_envelope])
