"""Tests for the Moonshine backend adapter.

These tests mock the moonshine_voice library so they run without it
installed.  The integration tests at the bottom require the actual
library and a downloaded model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rift_local.backends import BackendAdapter


# ---------------------------------------------------------------------------
# Minimal moonshine_voice fakes (no import of the real library)
# ---------------------------------------------------------------------------


@dataclass
class FakeTranscriptLine:
    text: str = ""
    start_time: float = 0.0
    duration: float = 0.0
    line_id: int = 1
    is_complete: bool = False
    is_updated: bool = False
    is_new: bool = False
    has_text_changed: bool = False
    has_speaker_id: bool = False
    speaker_id: int = 0
    speaker_index: int = 0
    audio_data: Any = None
    last_transcription_latency_ms: int = 0


@dataclass
class FakeTranscript:
    lines: list


class FakeStream:
    """Simulates moonshine_voice.Stream with controllable transcript state."""

    def __init__(self) -> None:
        self._audio_fed: list[list[float]] = []
        self._transcript = FakeTranscript(lines=[])

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass

    def add_audio(self, audio_data: list[float], sample_rate: int = 16000) -> None:
        self._audio_fed.append(audio_data)

    def update_transcription(self, flags: int = 0) -> FakeTranscript:
        return self._transcript

    # -- Test helpers --

    def set_interim(
        self, text: str, *, line_id: int = 1, start_time: float = 0.0
    ) -> None:
        """Set the transcript to an in-progress line."""
        self._transcript = FakeTranscript(
            lines=[
                FakeTranscriptLine(
                    text=text,
                    line_id=line_id,
                    start_time=start_time,
                    is_complete=False,
                    is_new=False,
                    has_text_changed=True,
                    is_updated=True,
                )
            ]
        )

    def set_final(
        self, text: str, *, line_id: int = 1, start_time: float = 0.0
    ) -> None:
        """Set the transcript to a completed line."""
        self._transcript = FakeTranscript(
            lines=[
                FakeTranscriptLine(
                    text=text,
                    line_id=line_id,
                    start_time=start_time,
                    is_complete=True,
                    is_new=False,
                    has_text_changed=True,
                    is_updated=True,
                )
            ]
        )

    def set_empty(self) -> None:
        """Clear the transcript."""
        self._transcript = FakeTranscript(lines=[])


class FakeTranscriber:
    """Simulates moonshine_voice.Transcriber."""

    def __init__(self, **kwargs: Any) -> None:
        self._stream = FakeStream()

    def create_stream(self, **kwargs: Any) -> FakeStream:
        return self._stream

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def moonshine_adapter():
    """Create a MoonshineAdapter with faked moonshine_voice internals."""
    from rift_local.models.registry import ModelEntry

    entry = ModelEntry(
        name="moonshine-en-medium",
        backend="moonshine",
        source="en",
        display="Moonshine Medium Streaming EN (245M)",
        params="245M",
        languages=["en"],
        size_mb=190,
    )

    # Patch the moonshine_voice imports inside the adapter module.
    with (
        patch("rift_local.backends.moonshine.Transcriber", FakeTranscriber),
        patch("rift_local.backends.moonshine.ModelArch", MagicMock()),
    ):
        from rift_local.backends.moonshine import MoonshineAdapter

        adapter = MoonshineAdapter(
            entry,
            model_path="/fake/model/path",
            model_arch=5,  # MEDIUM_STREAMING
            sample_rate=16_000,
        )
        yield adapter


@pytest.fixture()
def fake_stream(moonshine_adapter) -> FakeStream:
    """Return the FakeStream from inside the adapter."""
    stream = moonshine_adapter.create_stream()
    return moonshine_adapter._stream


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestMoonshineAdapterProtocol:
    """Verify MoonshineAdapter satisfies BackendAdapter protocol."""

    def test_satisfies_protocol(self, moonshine_adapter):
        assert isinstance(moonshine_adapter, BackendAdapter)


class TestMoonshineAdapterInfo:
    """Info handshake correctness."""

    def test_info_type(self, moonshine_adapter):
        info = moonshine_adapter.get_info()
        assert info.type == "info"

    def test_info_backend(self, moonshine_adapter):
        info = moonshine_adapter.get_info()
        assert info.backend == "moonshine"

    def test_info_no_confidence(self, moonshine_adapter):
        info = moonshine_adapter.get_info()
        assert info.features.confidence is False

    def test_info_no_timestamps(self, moonshine_adapter):
        info = moonshine_adapter.get_info()
        assert info.features.timestamps is False

    def test_info_streaming(self, moonshine_adapter):
        info = moonshine_adapter.get_info()
        assert info.streaming is True


class TestMoonshineAdapterDecoding:
    """Pull-based decode/result/endpoint cycle."""

    def test_empty_transcript(self, moonshine_adapter, fake_stream):
        """No crash when transcript is empty."""
        moonshine_adapter.decode(fake_stream)
        result = moonshine_adapter.get_result(fake_stream)
        assert result.text == ""
        assert moonshine_adapter.is_endpoint(fake_stream) is False

    def test_interim_result(self, moonshine_adapter, fake_stream):
        """Interim text is returned, no endpoint."""
        # Feed some audio so there's something to decode.
        samples = np.zeros(1600, dtype=np.float32)
        moonshine_adapter.feed_audio(fake_stream, samples)

        # Simulate Moonshine producing an interim.
        moonshine_adapter._stream.set_interim("hello world", line_id=42)
        # Set active_line_id (normally done on first decode of non-complete line).
        moonshine_adapter._active_line_id = 42

        moonshine_adapter.decode(fake_stream)
        result = moonshine_adapter.get_result(fake_stream)

        assert result.text == "hello world"
        assert moonshine_adapter.is_endpoint(fake_stream) is False

    def test_endpoint_detection(self, moonshine_adapter, fake_stream):
        """Completed line triggers is_endpoint."""
        # First, decode an interim to set active_line_id.
        moonshine_adapter._stream.set_interim("hello", line_id=42)
        moonshine_adapter.decode(fake_stream)
        assert moonshine_adapter._active_line_id == 42

        # Now the line completes.
        moonshine_adapter._stream.set_final("hello world", line_id=42)
        moonshine_adapter.decode(fake_stream)

        assert moonshine_adapter.is_endpoint(fake_stream) is True
        assert moonshine_adapter.get_result(fake_stream).text == "hello world"

    def test_reset_clears_state(self, moonshine_adapter, fake_stream):
        """reset() clears endpoint and tracking state."""
        moonshine_adapter._stream.set_interim("hello", line_id=42)
        moonshine_adapter.decode(fake_stream)
        moonshine_adapter._stream.set_final("hello", line_id=42)
        moonshine_adapter.decode(fake_stream)

        assert moonshine_adapter.is_endpoint(fake_stream) is True

        moonshine_adapter.reset(fake_stream)
        assert moonshine_adapter.is_endpoint(fake_stream) is False
        assert moonshine_adapter.get_result(fake_stream).text == ""
        assert moonshine_adapter._active_line_id is None

    def test_feed_audio_forwards_to_stream(self, moonshine_adapter, fake_stream):
        """feed_audio converts numpy to list and passes to stream."""
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        moonshine_adapter.feed_audio(fake_stream, samples)

        assert len(moonshine_adapter._stream._audio_fed) == 1
        fed = moonshine_adapter._stream._audio_fed[0]
        assert len(fed) == 3
        assert abs(fed[0] - 0.1) < 1e-6

    def test_is_ready_after_feed(self, moonshine_adapter, fake_stream):
        """is_ready is True after feed_audio, False after decode."""
        assert moonshine_adapter.is_ready(fake_stream) is False

        samples = np.zeros(1600, dtype=np.float32)
        moonshine_adapter.feed_audio(fake_stream, samples)
        assert moonshine_adapter.is_ready(fake_stream) is True

        moonshine_adapter.decode(fake_stream)
        assert moonshine_adapter.is_ready(fake_stream) is False


class TestMoonshineServerIntegration:
    """Test MoonshineAdapter through the server WS loop using fakes."""

    @pytest.fixture()
    def moonshine_client(self, moonshine_adapter):
        """Starlette TestClient wired to the moonshine adapter."""
        from starlette.testclient import TestClient

        from rift_local.server import create_app

        app = create_app(moonshine_adapter)
        with TestClient(app) as c:
            yield c

    def test_ws_handshake(self, moonshine_client):
        with moonshine_client.websocket_connect("/ws") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "info"
            assert msg["backend"] == "moonshine"
            assert msg["features"]["confidence"] is False

    def test_http_info(self, moonshine_client):
        r = moonshine_client.get("/info")
        assert r.status_code == 200
        data = r.json()
        assert data["backend"] == "moonshine"

    def test_ws_done_signal(self, moonshine_client):
        """Send Done immediately — clean close, no crash."""
        with moonshine_client.websocket_connect("/ws") as ws:
            ws.receive_json()  # info
            ws.send_text("Done")


# ---------------------------------------------------------------------------
# Slow integration tests (require moonshine-voice installed + model cached)
# ---------------------------------------------------------------------------

_SLOW_MODEL = "moonshine-en-medium"


def _can_run_moonshine() -> tuple[bool, str]:
    """Check whether slow moonshine tests can run."""
    try:
        from moonshine_voice import get_model_for_language, ModelArch
    except (ImportError, OSError) as exc:
        return False, f"moonshine-voice not importable: {exc}"
    # Try to resolve the model (will download if not cached, but
    # for CI we expect it to be pre-cached).
    try:
        get_model_for_language(
            wanted_language="en", wanted_model_arch=ModelArch.MEDIUM_STREAMING.value
        )
    except Exception as exc:
        return False, f"Moonshine model not available: {exc}"
    return True, ""


@pytest.mark.slow
class TestRealMoonshineBackend:
    """Integration tests with real moonshine-voice backend."""

    @pytest.fixture(autouse=True)
    def _require_moonshine(self):
        ok, reason = _can_run_moonshine()
        if not ok:
            pytest.skip(reason)

    @pytest.fixture()
    def real_client(self):
        from starlette.testclient import TestClient

        from rift_local.backends.moonshine import (
            MoonshineAdapter,
            ensure_moonshine_model,
        )
        from rift_local.models import get_model
        from rift_local.server import create_app

        entry = get_model(_SLOW_MODEL)
        model_path, model_arch = ensure_moonshine_model(entry)
        backend = MoonshineAdapter(entry, model_path=model_path, model_arch=model_arch)
        app = create_app(backend)
        with TestClient(app) as c:
            yield c

    def test_info_endpoint(self, real_client):
        data = real_client.get("/info").json()
        assert data["type"] == "info"
        assert data["model"] == _SLOW_MODEL
        assert data["backend"] == "moonshine"

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
