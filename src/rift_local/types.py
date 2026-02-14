"""Shared data models for the rift-local protocol."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from rift_local import __version__

# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------

DEFAULT_PORT = 2177
DEFAULT_HOST = "127.0.0.1"
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_MODEL = "nemotron-streaming-en"

# ---------------------------------------------------------------------------
# Info message (WS handshake + GET /info)
# ---------------------------------------------------------------------------


class Features(BaseModel):
    """Capability flags reported in the info handshake."""

    timestamps: bool = True
    confidence: bool = True
    endpoint_detection: bool = True
    diarization: bool = False


class InfoMessage(BaseModel):
    """Server/model metadata sent on WS connect and via GET /info."""

    type: Literal["info"] = "info"
    model: str
    model_display: str
    params: str
    backend: str
    streaming: bool = True
    languages: list[str]
    features: Features = Field(default_factory=Features)
    sample_rate: int = DEFAULT_SAMPLE_RATE
    version: str = __version__


# ---------------------------------------------------------------------------
# Result message (WS transcription result)
# ---------------------------------------------------------------------------


class ResultMessage(BaseModel):
    """A single transcription result sent over the WebSocket."""

    type: Literal["result"] = "result"
    text: str
    tokens: list[str] | None = None
    timestamps: list[float] | None = None
    ys_probs: list[float] | None = None
    lm_probs: list[float] | None = None
    context_scores: list[float] | None = None
    start_time: float | None = None
    segment: int
    is_final: bool
    model: str
