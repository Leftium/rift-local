"""FastAPI server: WebSocket streaming ASR and HTTP endpoints."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from rift_local.backends import BackendAdapter, Result
from rift_local.types import DEFAULT_SAMPLE_RATE, ResultMessage

# Tail padding: ~0.4s of silence flushed after the client sends "Done".
# This ensures the recognizer processes any trailing audio in its buffer.
_TAIL_PADDING_SAMPLES = int(0.4 * DEFAULT_SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Sync helpers — run in a thread so the event loop stays responsive
# ---------------------------------------------------------------------------


@dataclass
class _DecodeOutput:
    """Results collected from a single decode pass (run in a thread)."""

    result: Result | None = None
    is_endpoint: bool = False
    endpoint_result: Result | None = None


def _process_audio(
    backend: BackendAdapter,
    stream: object,
    samples: np.ndarray,
) -> _DecodeOutput:
    """Feed audio, decode, and collect results.  Runs in a worker thread."""
    backend.feed_audio(stream, samples)

    while backend.is_ready(stream):
        backend.decode(stream)

    result = backend.get_result(stream)
    out = _DecodeOutput()

    if result.text.strip():
        out.result = result

    if backend.is_endpoint(stream):
        # Re-read result after endpoint (may have changed).
        out.is_endpoint = True
        ep_result = backend.get_result(stream)
        if ep_result.text.strip():
            out.endpoint_result = ep_result
        backend.reset(stream)

    return out


def _process_tail(
    backend: BackendAdapter,
    stream: object,
) -> Result | None:
    """Flush tail padding and return final result.  Runs in a worker thread."""
    padding = np.zeros(_TAIL_PADDING_SAMPLES, dtype=np.float32)
    backend.feed_audio(stream, padding)

    while backend.is_ready(stream):
        backend.decode(stream)

    result = backend.get_result(stream)
    return result if result.text.strip() else None


# ---------------------------------------------------------------------------
# Result → message helper
# ---------------------------------------------------------------------------


def _to_msg(
    result: Result,
    *,
    segment: int,
    is_final: bool,
    model: str,
) -> dict:
    return ResultMessage(
        text=result.text,
        tokens=result.tokens or None,
        timestamps=result.timestamps or None,
        ys_probs=result.ys_probs or None,
        lm_probs=result.lm_probs or None,
        context_scores=result.context_scores or None,
        start_time=result.start_time,
        segment=segment,
        is_final=is_final,
        model=model,
    ).model_dump()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(backend: BackendAdapter) -> FastAPI:
    """Build and return a FastAPI application wired to *backend*."""
    app = FastAPI(title="rift-local", docs_url=None, redoc_url=None)

    info = backend.get_info()
    info_dict = info.model_dump()

    # -- HTTP endpoints -----------------------------------------------------

    @app.get("/info")
    def get_info() -> dict:
        return info_dict

    # -- WebSocket endpoint -------------------------------------------------

    @app.websocket("/ws")
    @app.websocket("/")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()

        # Send info handshake immediately.
        await ws.send_json(info_dict)

        stream = backend.create_stream()
        segment = 0
        last_text = ""  # Deduplicate: only send when text changes.

        try:
            while True:
                message = await ws.receive()

                if message["type"] == "websocket.receive":
                    if "bytes" in message and message["bytes"]:
                        # Binary frame: raw Float32 audio samples.
                        samples = np.frombuffer(message["bytes"], dtype=np.float32)

                        # Run CPU-bound decode in a thread to keep the
                        # event loop (and WS keepalive pings) responsive.
                        out = await asyncio.to_thread(
                            _process_audio, backend, stream, samples
                        )

                        if out.result and out.result.text != last_text:
                            last_text = out.result.text
                            await ws.send_json(
                                _to_msg(
                                    out.result,
                                    segment=segment,
                                    is_final=False,
                                    model=info.model,
                                )
                            )

                        if out.is_endpoint and out.endpoint_result:
                            await ws.send_json(
                                _to_msg(
                                    out.endpoint_result,
                                    segment=segment,
                                    is_final=True,
                                    model=info.model,
                                )
                            )
                            segment += 1
                            # Preserve the endpoint text to prevent duplication
                            # when the next chunk arrives but no new speech decoded yet.
                            last_text = out.endpoint_result.text

                    elif "text" in message and message["text"]:
                        text = message["text"]
                        if text.strip() == "Done":
                            result = await asyncio.to_thread(
                                _process_tail, backend, stream
                            )
                            if result:
                                await ws.send_json(
                                    _to_msg(
                                        result,
                                        segment=segment,
                                        is_final=True,
                                        model=info.model,
                                    )
                                )

                            await ws.close()
                            return

                elif message["type"] == "websocket.disconnect":
                    return

        except WebSocketDisconnect:
            pass

    return app
