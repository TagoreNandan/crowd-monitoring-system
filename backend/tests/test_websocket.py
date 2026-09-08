import pytest
from starlette.websockets import WebSocketDisconnect


def test_ws_system_connect_ack_and_status(client):
    with client.websocket_connect("/ws/system") as ws:
        ack = ws.receive_json()
        assert ack["type"] == "connection.ack"
        assert ack["job_id"] is None
        assert ack["data"]["channel"] == "system"

        status = ws.receive_json()
        assert status["type"] == "system.status"
        assert "state" in status["data"]
        assert "model_name" in status["data"]


def test_ws_system_ping_pong(client):
    with client.websocket_connect("/ws/system") as ws:
        ws.receive_json()  # ack
        ws.receive_json()  # status
        ws.send_text("ping")
        pong = ws.receive_json()
        assert pong["type"] == "pong"


def test_ws_job_connect_and_status(client, sample_jpeg_bytes):
    files = {"image": ("test.jpg", sample_jpeg_bytes, "image/jpeg")}
    job_id = client.post("/api/jobs/image", files=files).json()["job_id"]

    with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
        ack = ws.receive_json()
        assert ack["type"] == "connection.ack"
        assert ack["job_id"] == job_id

        status = ws.receive_json()
        assert status["type"] == "job.status"
        assert status["job_id"] == job_id
        assert status["data"]["status"] == "completed"
        assert status["data"]["progress"] == 1.0


def test_ws_job_not_found_disconnects(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/jobs/does-not-exist") as ws:
            ws.receive_json()


def test_ws_disconnect_is_clean_and_manager_still_works(client):
    with client.websocket_connect("/ws/system") as ws:
        ws.receive_json()
        ws.receive_json()
    # Socket closed by the `with` block above. A brand-new connection must
    # still work -- proves disconnect cleanup didn't corrupt shared state.
    with client.websocket_connect("/ws/system") as ws2:
        ack = ws2.receive_json()
        assert ack["type"] == "connection.ack"
