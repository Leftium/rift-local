"""sherpa-onnx streaming ASR backend adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rift_local.backends import Result
from rift_local.models.registry import ModelEntry
from rift_local.types import DEFAULT_SAMPLE_RATE, Features, InfoMessage

try:
    import sherpa_onnx
except ImportError:
    sherpa_onnx = None  # type: ignore[assignment]


class SherpaAdapter:
    """Wraps ``sherpa_onnx.OnlineRecognizer`` behind the BackendAdapter protocol."""

    def __init__(
        self,
        entry: ModelEntry,
        model_dir: Path,
        *,
        num_threads: int = 2,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        if sherpa_onnx is None:
            msg = (
                "sherpa-onnx is not installed.  "
                "Install it with:  pip install rift-local[sherpa]"
            )
            raise ImportError(msg)

        self._entry = entry
        self._sample_rate = sample_rate

        tokens = str(model_dir / entry.files["tokens"])
        encoder = str(model_dir / entry.files["encoder"])
        decoder = str(model_dir / entry.files["decoder"])
        joiner = str(model_dir / entry.files["joiner"])

        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            num_threads=num_threads,
            sample_rate=sample_rate,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=20.0,
            decoding_method="greedy_search",
        )

    # -- BackendAdapter protocol ----------------------------------------

    def create_stream(self) -> sherpa_onnx.OnlineStream:
        return self._recognizer.create_stream()

    def feed_audio(self, stream: sherpa_onnx.OnlineStream, samples: np.ndarray) -> None:
        stream.accept_waveform(self._sample_rate, samples.tolist())

    def is_ready(self, stream: sherpa_onnx.OnlineStream) -> bool:
        return self._recognizer.is_ready(stream)

    def decode(self, stream: sherpa_onnx.OnlineStream) -> None:
        self._recognizer.decode_stream(stream)

    def get_result(self, stream: sherpa_onnx.OnlineStream) -> Result:
        text = self._recognizer.get_result(stream)
        tokens = self._recognizer.tokens(stream)
        timestamps = self._recognizer.timestamps(stream)
        ys_probs = self._recognizer.ys_probs(stream)
        lm_probs = self._recognizer.lm_probs(stream)
        context_scores = self._recognizer.context_scores(stream)
        start_time = self._recognizer.start_time(stream)
        return Result(
            text=text,
            tokens=list(tokens),
            timestamps=list(timestamps),
            ys_probs=list(ys_probs),
            lm_probs=list(lm_probs),
            context_scores=list(context_scores),
            start_time=start_time,
        )

    def is_endpoint(self, stream: sherpa_onnx.OnlineStream) -> bool:
        return self._recognizer.is_endpoint(stream)

    def reset(self, stream: sherpa_onnx.OnlineStream) -> None:
        self._recognizer.reset(stream)

    def get_info(self) -> InfoMessage:
        return InfoMessage(
            model=self._entry.name,
            model_display=self._entry.display,
            params=self._entry.params,
            backend="sherpa-onnx",
            streaming=True,
            languages=list(self._entry.languages),
            features=Features(
                timestamps=True,
                confidence=True,
                endpoint_detection=True,
                diarization=False,
            ),
            sample_rate=self._sample_rate,
        )
