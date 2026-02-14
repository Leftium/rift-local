"""Model registry: known models, lookup helpers, and cache paths."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Cache root
# ---------------------------------------------------------------------------

_CACHE_DIR = Path.home() / ".cache" / "rift-local" / "models"


def _cache_dir() -> Path:
    """Return (and create) the rift-local model cache directory."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


# ---------------------------------------------------------------------------
# Model entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelEntry:
    """A single model in the registry."""

    name: str
    backend: str
    source: str  # Download URL (tarball) or HuggingFace repo ID
    display: str
    params: str
    languages: list[str]
    size_mb: int  # Extracted model size on disk
    download_mb: int | None = None  # Download size (tarball); None = same as size_mb
    streaming: bool = True
    platform: str | None = None  # None = all platforms, "darwin" = macOS only
    files: dict[str, str] = field(default_factory=dict)  # Logical role -> filename


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# fmt: off
_MODELS: dict[str, ModelEntry] = {
    "nemotron-streaming-en": ModelEntry(
        name="nemotron-streaming-en",
        backend="sherpa-onnx",
        source=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "asr-models/"
            "sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14.tar.bz2"
        ),
        files={
            "tokens": "tokens.txt",
            "encoder": "encoder.int8.onnx",
            "decoder": "decoder.int8.onnx",
            "joiner": "joiner.int8.onnx",
        },
        display="Nemotron Streaming EN 0.6B (int8)",
        params="0.6B",
        languages=["en"],
        size_mb=600,
        download_mb=447,
    ),
    "zipformer-kroko-en": ModelEntry(
        name="zipformer-kroko-en",
        backend="sherpa-onnx",
        source=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "asr-models/"
            "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06.tar.bz2"
        ),
        files={
            "tokens": "tokens.txt",
            "encoder": "encoder.onnx",
            "decoder": "decoder.onnx",
            "joiner": "joiner.onnx",
        },
        display="Zipformer Kroko EN (streaming)",
        params="~30M",
        languages=["en"],
        size_mb=68,
        download_mb=55,
    ),
    # -- Moonshine (moonshine-voice) ------------------------------------
    # Moonshine manages its own model cache via get_model_for_language().
    # source is the language code; files is empty (no manual download).
    "moonshine-tiny-en": ModelEntry(
        name="moonshine-tiny-en",
        backend="moonshine",
        source="en",
        display="Moonshine Tiny Streaming EN (34M)",
        params="34M",
        languages=["en"],
        size_mb=26,
    ),
    "moonshine-small-en": ModelEntry(
        name="moonshine-small-en",
        backend="moonshine",
        source="en",
        display="Moonshine Small Streaming EN (123M)",
        params="123M",
        languages=["en"],
        size_mb=95,
    ),
    "moonshine-medium-en": ModelEntry(
        name="moonshine-medium-en",
        backend="moonshine",
        source="en",
        display="Moonshine Medium Streaming EN (245M)",
        params="245M",
        languages=["en"],
        size_mb=190,
    ),
}
# fmt: on

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_model(name: str) -> ModelEntry:
    """Look up a model by name. Raises ``KeyError`` if not found."""
    try:
        return _MODELS[name]
    except KeyError:
        available = ", ".join(sorted(_MODELS))
        msg = f"Unknown model {name!r}. Available: {available}"
        raise KeyError(msg) from None


def list_models(
    *,
    backend: str | None = None,
    available_only: bool = True,
) -> list[ModelEntry]:
    """Return models, optionally filtered by backend.

    When *available_only* is ``True`` (default), models restricted to a
    different platform are excluded (e.g. macOS-only models on Linux).
    """
    current_platform = platform.system().lower()
    results: list[ModelEntry] = []
    for entry in _MODELS.values():
        if backend and entry.backend != backend:
            continue
        if available_only and entry.platform and entry.platform != current_platform:
            continue
        results.append(entry)
    return results


def model_path(name: str) -> Path:
    """Return the cache directory for a given model (may not exist yet)."""
    return _cache_dir() / name


def is_cached(name: str) -> bool:
    """Check whether all expected files for a model exist in the cache."""
    entry = get_model(name)
    base = model_path(name)
    if not base.exists():
        return False
    if not entry.files:
        # HuggingFace models — just check the directory exists and is non-empty.
        return any(base.iterdir())
    return all((base / fname).exists() for fname in entry.files.values())
