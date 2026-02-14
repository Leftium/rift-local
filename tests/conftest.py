"""Shared fixtures and pytest configuration."""

from __future__ import annotations

import numpy as np
import pytest

from rift_local.backends import BackendAdapter, Result
from rift_local.server import create_app
from rift_local.types import Features, InfoMessage


# ---------------------------------------------------------------------------
# --slow flag
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run slow tests (requires downloaded models).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--slow"):
        return
    skip_slow = pytest.mark.skip(reason="Use --slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------

_MOCK_INFO = InfoMessage(
    model="mock-model",
    model_display="Mock Model (test)",
    params="0",
    backend="mock",
    streaming=True,
    languages=["en"],
    features=Features(),
    sample_rate=16_000,
)


class MockBackend:
    """A fake backend that returns canned results for testing."""

    def __init__(self) -> None:
        self._text = ""
        self._samples_fed = 0
        self._decoded = False
        self._endpoint_at_samples = None  # Trigger endpoint after N samples.

    def create_stream(self) -> object:
        self._text = ""
        self._samples_fed = 0
        self._decoded = False
        return object()

    def feed_audio(self, stream: object, samples: np.ndarray) -> None:
        self._samples_fed += len(samples)
        self._decoded = False  # New audio: allow one more decode cycle.
        # Produce text after receiving enough audio (~0.5s at 16kHz).
        if self._samples_fed >= 8000:
            self._text = "hello world"

    def is_ready(self, stream: object) -> bool:
        return bool(self._text) and not self._decoded

    def decode(self, stream: object) -> None:
        self._decoded = True

    def get_result(self, stream: object) -> Result:
        return Result(
            text=self._text,
            tokens=[" hello", " world"] if self._text else [],
            timestamps=[0.32, 0.64] if self._text else [],
            ys_probs=[-0.12, -0.08] if self._text else [],
        )

    def is_endpoint(self, stream: object) -> bool:
        if self._endpoint_at_samples is None:
            return False
        return self._samples_fed >= self._endpoint_at_samples

    def reset(self, stream: object) -> None:
        self._text = ""
        self._samples_fed = 0
        self._endpoint_at_samples = None

    def get_info(self) -> InfoMessage:
        return _MOCK_INFO


# Verify MockBackend satisfies the protocol at import time.
assert isinstance(MockBackend(), BackendAdapter)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_backend() -> MockBackend:
    return MockBackend()


@pytest.fixture()
def client(mock_backend: MockBackend):
    """Starlette TestClient wired to the mock backend."""
    from starlette.testclient import TestClient

    app = create_app(mock_backend)
    with TestClient(app) as c:
        yield c
