"""Backend adapter interface for rift-local ASR engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from rift_local.types import InfoMessage


@dataclass
class Result:
    """Transcription result returned by a backend adapter."""

    text: str = ""
    tokens: list[str] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    ys_probs: list[float] = field(default_factory=list)
    lm_probs: list[float] = field(default_factory=list)
    context_scores: list[float] = field(default_factory=list)
    start_time: float = 0.0


# Stream is backend-specific; we use Any so each adapter can use its own type.
Stream = Any


@runtime_checkable
class BackendAdapter(Protocol):
    """Interface that all rift-local ASR backends implement."""

    def create_stream(self) -> Stream: ...

    def feed_audio(self, stream: Stream, samples: np.ndarray) -> None: ...

    def is_ready(self, stream: Stream) -> bool: ...

    def decode(self, stream: Stream) -> None: ...

    def get_result(self, stream: Stream) -> Result: ...

    def is_endpoint(self, stream: Stream) -> bool: ...

    def reset(self, stream: Stream) -> None: ...

    def get_info(self) -> InfoMessage: ...
