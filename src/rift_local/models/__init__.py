"""Model registry and download management for rift-local."""

from rift_local.models.download import ensure_model
from rift_local.models.registry import (
    ModelEntry,
    get_model,
    is_cached,
    list_models,
    model_path,
)

__all__ = [
    "ModelEntry",
    "ensure_model",
    "get_model",
    "is_cached",
    "list_models",
    "model_path",
]
