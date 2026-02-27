"""Moonshine streaming ASR backend adapter.

Wraps the moonshine-voice library's push-based Transcriber/Stream API
behind rift-local's pull-based BackendAdapter protocol.  The adapter
calls ``stream.add_audio()`` to feed samples and
``stream.update_transcription()`` to pull the current transcript state,
mapping Moonshine's ``TranscriptLine`` objects to rift-local ``Result``
messages.

See: https://github.com/moonshine-ai/moonshine
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rift_local.backends import Result
from rift_local.models.registry import ModelEntry
from rift_local.types import DEFAULT_SAMPLE_RATE, Features, InfoMessage

try:
    from moonshine_voice import (
        ModelArch,
        Transcriber,
        get_model_for_language,
    )
except ImportError:
    Transcriber = None  # type: ignore[assignment,misc]
    ModelArch = None  # type: ignore[assignment,misc]
    get_model_for_language = None  # type: ignore[assignment]


# Map rift-local model names to Moonshine ModelArch values.
_ARCH_MAP: dict[str, Any] = {}
if ModelArch is not None:
    _ARCH_MAP = {
        "moonshine-en-tiny": ModelArch.TINY_STREAMING,
        "moonshine-en-small": ModelArch.SMALL_STREAMING,
        "moonshine-en-medium": ModelArch.MEDIUM_STREAMING,
    }


def ensure_moonshine_model(entry: ModelEntry) -> tuple[str, Any]:
    """Download (if needed) and return ``(model_path, model_arch)``.

    Moonshine manages its own model cache via ``get_model_for_language``.
    We call it with the language from the registry entry and the desired
    architecture.

    Returns:
        Tuple of (model_path_str, ModelArch).
    """
    if get_model_for_language is None:
        msg = (
            "moonshine-voice is not installed.  "
            "Install it with:  pip install rift-local[moonshine]"
        )
        raise ImportError(msg)

    arch = _ARCH_MAP.get(entry.name)
    if arch is None:
        msg = f"No Moonshine architecture mapping for model {entry.name!r}"
        raise ValueError(msg)

    lang = entry.languages[0] if entry.languages else "en"
    model_path, model_arch = get_model_for_language(
        wanted_language=lang,
        wanted_model_arch=arch.value,
    )
    return model_path, model_arch


class MoonshineAdapter:
    """Wraps ``moonshine_voice.Transcriber`` behind the BackendAdapter protocol.

    Moonshine's native API is event/push-driven: feed audio via
    ``stream.add_audio()`` and receive ``LineStarted``/``LineTextChanged``/
    ``LineCompleted`` events.  This adapter converts that to pull semantics:

    - ``feed_audio()`` stores samples via ``stream.add_audio()`` but
      suppresses Moonshine's auto-update by using a very large
      ``update_interval``.
    - ``decode()`` explicitly calls ``stream.update_transcription()``
      to pull the current state.
    - ``get_result()`` reads the latest transcript line.
    - ``is_endpoint()`` detects when the active line has completed.
    """

    def __init__(
        self,
        entry: ModelEntry,
        model_path: str,
        model_arch: Any,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        if Transcriber is None:
            msg = (
                "moonshine-voice is not installed.  "
                "Install it with:  pip install rift-local[moonshine]"
            )
            raise ImportError(msg)

        self._entry = entry
        self._sample_rate = sample_rate

        # Use a very large update_interval so add_audio() never
        # auto-triggers transcription — we control timing via decode().
        self._transcriber = Transcriber(
            model_path=model_path,
            model_arch=model_arch,
            update_interval=999_999.0,
        )

        # Per-stream state (set in create_stream / reset).
        self._stream: Any = None
        self._last_text: str = ""
        self._is_endpoint: bool = False
        self._active_line_id: int | None = None
        self._result: Result = Result()
        self._needs_decode: bool = False

    # -- BackendAdapter protocol ----------------------------------------

    def create_stream(self) -> object:
        stream = self._transcriber.create_stream(
            update_interval=999_999.0,
        )
        stream.start()
        self._stream = stream
        self._last_text = ""
        self._is_endpoint = False
        self._active_line_id = None
        self._result = Result()
        self._needs_decode = False
        return stream

    def feed_audio(self, stream: object, samples: np.ndarray) -> None:
        # Moonshine expects a list of floats, not a numpy array.
        self._stream.add_audio(samples.tolist(), self._sample_rate)
        self._needs_decode = True

    def is_ready(self, stream: object) -> bool:
        # Return True exactly once per feed_audio() call so the
        # server's ``while is_ready: decode()`` loop runs one
        # decode pass then exits.
        return self._needs_decode

    def decode(self, stream: object) -> None:
        self._needs_decode = False
        transcript = self._stream.update_transcription()

        self._is_endpoint = False
        self._result = Result()

        if not transcript or not transcript.lines:
            return

        # Find the last (most recent) line.
        line = transcript.lines[-1]

        if line.text and line.text.strip():
            self._result = Result(
                text=line.text.strip(),
                start_time=line.start_time,
            )

        # Detect endpoint: active line became complete.
        if line.is_complete and line.line_id == self._active_line_id:
            self._is_endpoint = True

        # Track the active line for endpoint detection.
        if not line.is_complete:
            self._active_line_id = line.line_id

    def get_result(self, stream: object) -> Result:
        return self._result

    def is_endpoint(self, stream: object) -> bool:
        return self._is_endpoint

    def reset(self, stream: object) -> None:
        # Moonshine manages segments internally.  After an endpoint,
        # the next add_audio() will start a new line automatically.
        # We just clear our tracking state.
        self._last_text = ""
        self._is_endpoint = False
        self._active_line_id = None
        self._result = Result()

    def get_info(self) -> InfoMessage:
        return InfoMessage(
            model=self._entry.name,
            model_display=self._entry.display,
            params=self._entry.params,
            backend="moonshine",
            streaming=True,
            languages=list(self._entry.languages),
            features=Features(
                timestamps=False,
                confidence=False,
                endpoint_detection=True,
                diarization=False,
            ),
            sample_rate=self._sample_rate,
        )

    def close(self) -> None:
        """Release Moonshine resources."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
        if self._transcriber is not None:
            try:
                self._transcriber.close()
            except Exception:  # noqa: BLE001
                pass
            self._transcriber = None
