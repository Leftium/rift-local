"""FastAPI server: WebSocket streaming ASR and HTTP endpoints."""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from rift_local.backends import BackendAdapter
from rift_local.types import DEFAULT_SAMPLE_RATE, ResultMessage

# Tail padding: ~0.4s of silence flushed after the client sends "Done".
# This ensures the recognizer processes any trailing audio in its buffer.
_TAIL_PADDING_SAMPLES = int(0.4 * DEFAULT_SAMPLE_RATE)


def create_app(backend: BackendAdapter) -> FastAPI:
    """Build and return a FastAPI application wired to *backend*."""
    app = FastAPI(title="rift-local", docs_url=None, redoc_url=None)

    info = backend.get_info()

    # -- HTTP endpoints -----------------------------------------------------

    @app.get("/info")
    def get_info() -> dict:
        return info.model_dump()

    # -- WebSocket endpoint -------------------------------------------------

    @app.websocket("/ws")
    @app.websocket("/")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()

        # Send info handshake immediately.
        await ws.send_json(info.model_dump())

        stream = backend.create_stream()
        segment = 0

        try:
            while True:
                message = await ws.receive()

                if message["type"] == "websocket.receive":
                    if "bytes" in message and message["bytes"]:
                        # Binary frame: raw Float32 audio samples.
                        samples = np.frombuffer(message["bytes"], dtype=np.float32)
                        backend.feed_audio(stream, samples)

                        # Decode as many frames as available.
                        while backend.is_ready(stream):
                            backend.decode(stream)

                        result = backend.get_result(stream)

                        if result.text.strip():
                            msg = ResultMessage(
                                text=result.text,
                                tokens=result.tokens or None,
                                timestamps=result.timestamps or None,
                                ys_probs=result.ys_probs or None,
                                lm_probs=result.lm_probs or None,
                                context_scores=result.context_scores or None,
                                start_time=result.start_time,
                                segment=segment,
                                is_final=False,
                                model=info.model,
                            )
                            await ws.send_json(msg.model_dump())

                        # Check for endpoint (silence/pause detected).
                        if backend.is_endpoint(stream):
                            result = backend.get_result(stream)
                            if result.text.strip():
                                msg = ResultMessage(
                                    text=result.text,
                                    tokens=result.tokens or None,
                                    timestamps=result.timestamps or None,
                                    ys_probs=result.ys_probs or None,
                                    lm_probs=result.lm_probs or None,
                                    context_scores=result.context_scores or None,
                                    start_time=result.start_time,
                                    segment=segment,
                                    is_final=True,
                                    model=info.model,
                                )
                                await ws.send_json(msg.model_dump())
                                segment += 1

                            backend.reset(stream)

                    elif "text" in message and message["text"]:
                        text = message["text"]
                        if text.strip() == "Done":
                            # Flush tail padding and send final results.
                            padding = np.zeros(_TAIL_PADDING_SAMPLES, dtype=np.float32)
                            backend.feed_audio(stream, padding)

                            while backend.is_ready(stream):
                                backend.decode(stream)

                            result = backend.get_result(stream)
                            if result.text.strip():
                                msg = ResultMessage(
                                    text=result.text,
                                    tokens=result.tokens or None,
                                    timestamps=result.timestamps or None,
                                    ys_probs=result.ys_probs or None,
                                    lm_probs=result.lm_probs or None,
                                    context_scores=result.context_scores or None,
                                    start_time=result.start_time,
                                    segment=segment,
                                    is_final=True,
                                    model=info.model,
                                )
                                await ws.send_json(msg.model_dump())

                            await ws.close()
                            return

                elif message["type"] == "websocket.disconnect":
                    return

        except WebSocketDisconnect:
            pass

    return app
