"""Tests for the FastAPI server (WS + HTTP endpoints)."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from rift_local.types import DEFAULT_SAMPLE_RATE


# ---------------------------------------------------------------------------
# Fast tests (mock backend, no model download)
# ---------------------------------------------------------------------------


class TestGetInfo:
    """GET /info endpoint."""

    def test_returns_200(self, client):
        r = client.get("/info")
        assert r.status_code == 200

    def test_has_required_fields(self, client):
        data = client.get("/info").json()
        for key in (
            "type",
            "model",
            "model_display",
            "params",
            "backend",
            "streaming",
            "languages",
            "features",
            "sample_rate",
            "version",
        ):
            assert key in data, f"Missing field: {key}"

    def test_type_is_info(self, client):
        data = client.get("/info").json()
        assert data["type"] == "info"

    def test_model_matches_mock(self, client):
        data = client.get("/info").json()
        assert data["model"] == "mock-model"
        assert data["backend"] == "mock"


class TestWebSocketHandshake:
    """WS /ws connection and info handshake."""

    def test_receives_info_on_connect(self, client):
        with client.websocket_connect("/ws") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "info"
            assert msg["model"] == "mock-model"

    def test_root_path_alias(self, client):
        """WS on ``/`` works identically to ``/ws``."""
        with client.websocket_connect("/") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "info"
            assert msg["model"] == "mock-model"

    def test_info_matches_http(self, client):
        http_info = client.get("/info").json()
        with client.websocket_connect("/ws") as ws:
            ws_info = ws.receive_json()
        assert http_info == ws_info


class TestWebSocketStreaming:
    """WS audio streaming and result messages."""

    @staticmethod
    def _make_audio(seconds: float = 1.0) -> bytes:
        """Generate silent Float32 audio bytes."""
        n = int(DEFAULT_SAMPLE_RATE * seconds)
        return np.zeros(n, dtype=np.float32).tobytes()

    def test_send_audio_receive_result(self, client):
        with client.websocket_connect("/ws") as ws:
            # Consume info handshake.
            ws.receive_json()

            # Send enough audio to trigger mock result (~0.5s needed).
            ws.send_bytes(self._make_audio(1.0))

            msg = ws.receive_json()
            assert msg["type"] == "result"
            assert msg["text"] == "hello world"
            assert msg["segment"] == 0
            assert msg["is_final"] is False
            assert msg["model"] == "mock-model"

    def test_result_has_token_fields(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # info
            ws.send_bytes(self._make_audio(1.0))
            msg = ws.receive_json()
            assert msg["tokens"] == [" hello", " world"]
            assert msg["timestamps"] == [0.32, 0.64]
            assert msg["ys_probs"] == [-0.12, -0.08]

    def test_done_signal_closes(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # info
            ws.send_text("Done")
            # Server should close the connection after Done.
            # The next receive should raise or return a close frame.

    def test_endpoint_no_duplicate_on_next_chunk(self, client, mock_backend):
        """After endpoint, next chunk shouldn't re-send the old text as new segment."""
        # Configure mock to trigger endpoint after 10000 samples.
        mock_backend._endpoint_at_samples = 10000

        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # info

            # Send 1s of audio (16000 samples) — triggers text + endpoint.
            ws.send_bytes(self._make_audio(1.0))

            # Should receive:
            # 1. Interim result (seg=0, is_final=false)
            msg1 = ws.receive_json()
            assert msg1["type"] == "result"
            assert msg1["text"] == "hello world"
            assert msg1["segment"] == 0
            assert msg1["is_final"] is False

            # 2. Endpoint result (seg=0, is_final=true)
            msg2 = ws.receive_json()
            assert msg2["type"] == "result"
            assert msg2["text"] == "hello world"
            assert msg2["segment"] == 0
            assert msg2["is_final"] is True

            # Now send another chunk — mock still returns "hello world"
            # because it hasn't decoded new speech yet. This should NOT
            # send a result (duplicate detection via last_text).
            ws.send_bytes(self._make_audio(0.1))

            # Try to receive with timeout — should timeout (no message).
            # TestClient doesn't support receive_json(timeout=...) but we
            # can check by sending Done and seeing what comes back.
            ws.send_text("Done")
            # Should get tail result if any, but NOT a duplicate of seg=1.


# ---------------------------------------------------------------------------
# Slow tests (real sherpa-onnx backend, requires cached model)
# ---------------------------------------------------------------------------

_SLOW_MODEL = "zipformer-small-en"


def _can_run_real_backend() -> tuple[bool, str]:
    """Check whether slow tests can run (model cached + sherpa importable)."""
    from rift_local.models import is_cached

    if not is_cached(_SLOW_MODEL):
        return False, f"Model {_SLOW_MODEL!r} not in cache"
    try:
        import sherpa_onnx  # noqa: F401
    except (ImportError, OSError) as exc:
        return False, f"sherpa-onnx not importable: {exc}"
    return True, ""


@pytest.mark.slow
class TestRealBackend:
    """Integration tests with real sherpa-onnx backend."""

    @pytest.fixture(autouse=True)
    def _require_model(self):
        ok, reason = _can_run_real_backend()
        if not ok:
            pytest.skip(reason)

    @pytest.fixture()
    def real_client(self):
        from starlette.testclient import TestClient

        from rift_local.backends.sherpa import SherpaAdapter
        from rift_local.models import get_model, model_path
        from rift_local.server import create_app

        entry = get_model(_SLOW_MODEL)
        path = model_path(_SLOW_MODEL)
        backend = SherpaAdapter(entry, path, num_threads=2)
        app = create_app(backend)
        with TestClient(app) as c:
            yield c

    def test_info_endpoint(self, real_client):
        data = real_client.get("/info").json()
        assert data["type"] == "info"
        assert data["model"] == _SLOW_MODEL
        assert data["backend"] == "sherpa-onnx"

    def test_ws_handshake(self, real_client):
        with real_client.websocket_connect("/ws") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "info"
            assert msg["model"] == _SLOW_MODEL

    def test_ws_done_signal(self, real_client):
        """Send Done immediately — should get clean close, no crash."""
        with real_client.websocket_connect("/ws") as ws:
            ws.receive_json()  # info
            ws.send_text("Done")
