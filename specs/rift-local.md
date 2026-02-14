# rift-local

Local inference server for [RIFT Transcription](https://github.com/Leftium/rift-transcription). Serves streaming and batch speech recognition over WebSocket, batch transcription and LLM transforms over HTTP, and works as a standalone CLI transcription tool -- all backed by local models with automatic download and model metadata.

```
pip install rift-local
rift-local serve --model nemotron-streaming-en
```

```
rift-local transcribe meeting.wav --transform "Fix punctuation and add paragraphs"
```

On Apple Silicon, install with MLX support for native GPU-accelerated batch transcription and LLM transforms:

```
pip install rift-local[mlx]
rift-local serve --model mlx-whisper-large-v3-turbo --llm mlx:llama-3.2-3b
```

---

## Why

RIFT Transcription connects to local ASR engines over WebSocket. Today that means connecting directly to `sherpa-onnx-online-websocket-server`, a C++ binary. This works but has three problems that compound as RIFT grows:

**1. No model metadata.** The sherpa server never reports which model is loaded. RIFT's UI cannot display "Nemotron 0.6B" vs "Zipformer Small" -- the client has no way to know. Users switching between models get no feedback about what's actually running.

**2. Locked to one engine.** Sherpa-onnx is the only local engine with a built-in WebSocket server. Moonshine v2 (the next planned RIFT local source) only has a Python API -- no WS server. Each new engine would need its own bespoke bridge. The client would need to know which engine is behind the socket and adapt accordingly.

**3. The Python server drops fields.** Sherpa-onnx has both a C++ and a Python WebSocket server. The C++ server sends rich per-token data (tokens, timestamps, log-probs, `is_final`). The Python server sends only `{text, segment}` -- losing all word-level detail. This is not a limitation of the Python bindings (they expose everything the C++ server does) but of the upstream Python server code, which simply never serializes the full result. RIFT's client already has two code paths to handle this discrepancy.

Beyond streaming, RIFT needs two more capabilities that no existing server provides:

**4. Batch re-transcription.** After a live streaming session, users may want to re-transcribe a segment (or the entire recording) with a higher-accuracy model. This is audio that already exists in the browser's memory -- not a file upload, but a buffer that should be sent through a better model and have the results replace specific segments.

**5. LLM transforms.** Raw transcripts need cleanup: punctuation, grammar, formatting, summarization. These are LLM tasks that should run locally alongside the ASR model, accessible from both the RIFT app and the command line.

**rift-local solves all five** with a single server process:

- It **reports model metadata** on connection (name, params, capabilities) because it controls the protocol.
- It **abstracts the engine** behind a unified interface. The client speaks one protocol regardless of whether sherpa-onnx, Moonshine, or a future engine is doing inference.
- It **uses the full Python bindings** for sherpa-onnx, accessing tokens, timestamps, and log-probs directly -- matching the C++ server's output without its deployment friction.
- It **handles batch transcription** over both WebSocket (for RIFT's in-memory audio) and HTTP (for files from scripts and CLI).
- It **runs LLM transforms** via HTTP endpoint and CLI command, using local models.

---

## Architecture

```
                           rift-local (single FastAPI process)
                          ┌─────────────────────────────────────────┐
Browser (RIFT)            │                                         │     ASR Engines
┌──────────────┐    WS    │  WS /ws                                 │    ┌──────────────┐
│  AudioWorklet├─────────>│    ├── streaming ASR (live mic)         │    │  sherpa-onnx  │
│  (live)      │<─────────│    ├── batch ASR (from buffer)    ──────┼───>│  OnlineRecog  │
│              │          │    └── re-transcribe (segment)          │    └──────────────┘
└──────────────┘          │                                         │    ┌──────────────┐
                          │  HTTP                                   │    │  Moonshine    │
External / CLI            │    ├── POST /transcribe (file)    ──────┼───>│  MicTranscr   │
┌──────────────┐   HTTP   │    ├── POST /transform (LLM)           │    └──────────────┘
│  curl / app  ├─────────>│    └── GET  /info (metadata)           │    ┌──────────────┐
│              │<─────────│                                         │    │  mlx-whisper  │
└──────────────┘          │  Core pipeline                          │    │  (macOS only) │
                          │    audio frames ──> engine ──> results  │    └──────────────┘
CLI (no server)           │                                         │    ┌──────────────┐
┌──────────────┐          │  Model Registry                         │    │  LLM backend  │
│ rift-local   │          │    (download + cache)                    │    │  (mlx-lm/    │
│  transcribe  ├──────────┤                                         │    │   ollama/     │
│  transform   │  direct  │  Backend Adapters                       │    │   llama.cpp)  │
└──────────────┘          │    ├── sherpa-onnx                      │    └──────────────┘
                          │    ├── moonshine                        │
                          │    ├── mlx-whisper (macOS)              │
                          │    └── (future)                         │
                          └─────────────────────────────────────────┘
```

**Two modes of operation:**

1. **Server mode** (`rift-local serve`): Starts a FastAPI server exposing WS and HTTP endpoints. This is the primary mode for use with the RIFT browser app.
2. **CLI mode** (`rift-local transcribe`, `rift-local transform`): Runs inference directly without starting a server. For command-line batch transcription and scripting.

Both share the same core pipeline, model registry, and backend adapters.

**Key design choices:**

- **One process, one model.** rift-local loads one ASR model at startup and serves it. To switch models, restart with a different `--model` flag. This keeps memory predictable and avoids GPU contention.
- **Single-user by default.** RIFT is a personal tool. The server processes one audio stream at a time with `max-batch-size=1` for lowest latency. Multi-client support is not a goal.
- **Localhost only.** The server binds to `127.0.0.1` by default. No auth, no TLS -- it's a local IPC mechanism. RIFT's Vite dev proxy (`/ws/sherpa` -> `ws://localhost:PORT`) handles remote/ngrok scenarios.
- **WS is the core pipeline.** The WebSocket handler contains the real audio processing loop. The HTTP `/transcribe` endpoint is a thin wrapper that decodes an uploaded audio file, feeds it through the same pipeline, and collects results. This means one code path for audio processing regardless of how audio arrives.
- **HTTP for request/response.** Batch file uploads and LLM text transforms are request/response operations. HTTP is the natural fit: `curl`-testable, no connection lifecycle, standard error codes. Audio file decoding (wav, mp3, m4a) happens server-side in the HTTP path.
- **Platform-conditional backends.** MLX backends (mlx-whisper for ASR, mlx-lm for LLM transforms) only run on macOS Apple Silicon. They are optional extras (`pip install rift-local[mlx]`) that provide native Apple Silicon performance. On other platforms, sherpa-onnx/Moonshine handle ASR and Ollama handles LLM transforms. The core pipeline and protocol are identical regardless of backend -- the platform difference is an installation choice, not a protocol difference.

---

## WebSocket Protocol

The WebSocket interface handles all real-time and buffer-based audio interactions.

### Connection

Client opens `ws://localhost:{port}/ws` (default port: 8765).

### Handshake (server -> client)

On connection, the server sends an `info` message before any transcription results:

```json
{
	"type": "info",
	"model": "nemotron-streaming-en",
	"model_display": "Nemotron Streaming EN 0.6B (int8)",
	"params": "0.6B",
	"backend": "sherpa-onnx",
	"streaming": true,
	"languages": ["en"],
	"features": {
		"timestamps": true,
		"confidence": true,
		"endpoint_detection": true,
		"diarization": false
	},
	"sample_rate": 16000,
	"version": "0.1.0"
}
```

The client can use this to display model info in the UI without any out-of-band configuration.

### Audio input (client -> server)

Binary WebSocket frames containing raw `Float32Array` samples at 16kHz mono. Same format RIFT already sends to the sherpa C++ server.

This covers three use cases with the same wire format:

| Use case              | Audio source                                | Timing                                               |
| --------------------- | ------------------------------------------- | ---------------------------------------------------- |
| **Live streaming**    | Microphone via AudioWorklet                 | Real-time (chunks arrive at mic speed)               |
| **Batch from buffer** | Previously recorded audio in browser memory | Fast (chunks arrive at read speed)                   |
| **Re-transcription**  | Selected segment audio from browser buffer  | Fast (single segment sent, results replace original) |

The server doesn't distinguish between these -- it receives Float32 frames and processes them. The difference is only on the client side (what produces the audio and what happens with the results).

### Stop signal (client -> server)

The string `"Done"` (text frame) signals end of audio. The server flushes any remaining audio through the recognizer with tail padding and sends final results.

### Transcription result (server -> client)

```json
{
	"type": "result",
	"text": "Hello world",
	"tokens": [" Hello", " world"],
	"timestamps": [0.32, 0.64],
	"ys_probs": [-0.12, -0.08],
	"lm_probs": [-0.03, -0.02],
	"context_scores": [0.0, 0.0],
	"start_time": 0.0,
	"segment": 0,
	"is_final": false,
	"model": "nemotron-streaming-en"
}
```

**Fields:**

| Field            | Type       | Required | Description                                        |
| ---------------- | ---------- | -------- | -------------------------------------------------- |
| `type`           | `string`   | Yes      | Always `"result"` for transcription messages       |
| `text`           | `string`   | Yes      | Full segment text                                  |
| `tokens`         | `string[]` | No       | BPE subword tokens (sherpa-onnx)                   |
| `timestamps`     | `number[]` | No       | Per-token timestamps in seconds from segment start |
| `ys_probs`       | `number[]` | No       | Per-token ASR model log-probs                      |
| `lm_probs`       | `number[]` | No       | Per-token language model log-probs                 |
| `context_scores` | `number[]` | No       | Per-token hotword/context boosting log-probs       |
| `start_time`     | `number`   | No       | Segment start time in seconds                      |
| `segment`        | `number`   | Yes      | Monotonically increasing segment ID                |
| `is_final`       | `boolean`  | Yes      | `true` when endpoint detected (silence/pause)      |
| `model`          | `string`   | Yes      | Model identifier (matches `info.model`)            |

**Compatibility note:** The `type` field is new (sherpa servers don't send it). RIFT's `sherpa.svelte.ts` will need a minor update to handle the `type` field, or rift-local can omit it for backward compatibility. The `model` field is always new. All other fields match the existing sherpa C++ server protocol -- RIFT's BPE coalescing, confidence calculation, and endpoint detection logic work unchanged.

### Protocol evolution

New fields may be added to `info` and `result` messages in future versions. Clients must ignore unknown fields. No existing field will change type or be removed without a major version bump.

---

## HTTP Endpoints

HTTP endpoints serve request/response operations: batch file transcription, LLM transforms, and server metadata. They run alongside the WebSocket endpoint from the same FastAPI process on the same port.

### `GET /info`

Returns server and model metadata. Same content as the WS handshake `info` message.

```
GET http://localhost:8765/info

200 OK
{
  "model": "nemotron-streaming-en",
  "model_display": "Nemotron Streaming EN 0.6B (int8)",
  "params": "0.6B",
  "backend": "sherpa-onnx",
  "streaming": true,
  "languages": ["en"],
  "features": { ... },
  "sample_rate": 16000,
  "version": "0.1.0"
}
```

Useful for health checks, UI pre-population, and debugging ("is the server running and what model is loaded?").

### `POST /transcribe`

Batch transcription of an audio file. Accepts common audio formats (wav, mp3, m4a, ogg, flac) -- the server decodes to PCM internally using `soundfile` or `ffmpeg`. No need for the client to pre-process audio into raw Float32.

```
POST http://localhost:8765/transcribe
Content-Type: multipart/form-data

audio: <file>

200 OK
{
  "text": "Full transcript text here...",
  "segments": [
    {
      "text": "Full transcript text here",
      "segment": 0,
      "start_time": 0.0,
      "is_final": true,
      "tokens": [...],
      "timestamps": [...]
    }
  ],
  "model": "nemotron-streaming-en",
  "duration": 12.5,
  "processing_time": 3.2
}
```

**Internally**, the HTTP handler decodes the audio file to Float32 PCM, feeds it through the same pipeline as the WebSocket handler, collects all results, and returns them as a single JSON response.

For large files there is no size limit -- the server processes audio in chunks through the streaming pipeline. Upload progress is handled natively by HTTP clients. Processing progress (how far through the file) is not available in the response; for progress feedback on large files, use the WebSocket interface instead.

**Why both WS and HTTP for batch?**

| Caller                    | Use                                    | Transport      | Reason                                                     |
| ------------------------- | -------------------------------------- | -------------- | ---------------------------------------------------------- |
| RIFT (re-transcribe)      | Audio buffer already in browser memory | WS             | Connection already open, no file encoding needed           |
| RIFT (full re-transcribe) | Entire session buffer                  | WS             | Same -- buffer, not file                                   |
| CLI script                | Audio file on disk                     | HTTP (via CLI) | `rift-local transcribe file.wav` calls pipeline directly   |
| External tool             | Audio file                             | HTTP           | `curl -F "audio=@file.wav"` -- simple, no WS client needed |
| CI pipeline               | Audio files                            | HTTP           | Standard HTTP, scriptable                                  |

### `POST /transform`

LLM text transformation. Sends text through a local LLM with a prompt.

```
POST http://localhost:8765/transform
Content-Type: application/json

{
  "text": "so um i was thinking that we should uh probably fix the thing",
  "prompt": "Fix punctuation, remove filler words, and capitalize properly"
}

200 OK
{
  "text": "I was thinking that we should probably fix the thing.",
  "model": "llama-3.2-3b",
  "processing_time": 1.1
}
```

The LLM backend is configured at server startup (see CLI options). LLM transforms are stateless -- each request is independent.

---

## CLI Interface

rift-local operates in two modes: **server mode** (long-running, serves RIFT and external clients) and **CLI mode** (one-shot, processes a file or text and exits).

### Server Commands

#### `rift-local serve`

Start the WebSocket + HTTP server.

```
rift-local serve [OPTIONS]

Options:
  --model NAME        ASR model to load (see 'rift-local list')   [default: nemotron-streaming-en]
  --port PORT         Server port (WS + HTTP)                     [default: 8765]
  --host HOST         Bind address                                [default: 127.0.0.1]
  --device DEVICE     Compute device: cpu, cuda, coreml, mlx      [default: auto]
  --threads N         Number of inference threads                  [default: 2]
  --llm MODEL         LLM for /transform endpoint (optional)      [default: none]
                      Prefix selects backend: mlx:, ollama:, openai:
                      Examples: mlx:llama-3.2-3b, ollama:llama3.2:3b
```

On first run with a given model, rift-local downloads the model files automatically with a progress bar, then starts serving.

### CLI Commands (no server needed)

#### `rift-local transcribe`

Transcribe an audio file directly. No server process required.

```
rift-local transcribe FILE [OPTIONS]

Options:
  --model NAME        ASR model to use                            [default: nemotron-streaming-en]
  --format FORMAT     Output format: text, json, srt              [default: text]
  --output FILE       Write output to file instead of stdout
  --transform PROMPT  Apply LLM transform to result (requires --llm)
  --llm MODEL         LLM for --transform (prefix: mlx:, ollama:, openai:)
  --device DEVICE     Compute device: cpu, cuda, coreml, mlx      [default: auto]
```

Examples:

```bash
# Simple transcription
rift-local transcribe meeting.wav

# JSON output with word-level detail
rift-local transcribe meeting.wav --format json

# Transcribe and clean up in one command
rift-local transcribe meeting.wav --llm mlx:llama-3.2-3b \
  --transform "Fix punctuation and add paragraphs"

# Pipeline with other tools
rift-local transcribe meeting.wav | grep "action item"

# Different model for better accuracy
rift-local transcribe podcast.mp3 --model whisper-large-v3

# MLX Whisper on Apple Silicon (batch, high accuracy)
rift-local transcribe podcast.mp3 --model mlx-whisper-large-v3-turbo

# Full MLX pipeline: transcribe + transform, no external services
rift-local transcribe meeting.wav --model mlx-whisper-large-v3-turbo \
  --llm mlx:llama-3.2-3b --transform "Fix punctuation and add paragraphs"
```

Accepts any audio format (wav, mp3, m4a, ogg, flac). Decoding happens internally.

#### `rift-local transform`

Apply an LLM transform to text. Reads from stdin or `--input`.

```
rift-local transform [OPTIONS]

Options:
  --prompt PROMPT     Transform instruction (required)
  --llm MODEL         LLM to use                                  [required]
  --input FILE        Read text from file instead of stdin
  --output FILE       Write result to file instead of stdout
```

Examples:

```bash
# Transform from stdin (mlx-lm on Apple Silicon)
echo "raw transcript text" | rift-local transform \
  --llm mlx:llama-3.2-3b \
  --prompt "Summarize this meeting"

# Transform using Ollama (cross-platform)
echo "raw transcript text" | rift-local transform \
  --llm ollama:llama3.2:3b \
  --prompt "Summarize this meeting"

# Transform a file
rift-local transform \
  --input meeting.txt \
  --output summary.txt \
  --llm mlx:llama-3.2-3b \
  --prompt "Extract action items as a bullet list"

# Pipeline: transcribe, then transform
rift-local transcribe meeting.wav | rift-local transform \
  --llm mlx:llama-3.2-3b \
  --prompt "Fix punctuation and grammar"
```

### Model Management Commands

#### `rift-local list`

Show available models.

```
rift-local list [OPTIONS]

Options:
  --installed         Show only downloaded models with cache sizes
```

Example output:

```
Available models:

  Backend: sherpa-onnx
  * nemotron-streaming-en     600MB   Streaming, EN, 0.6B params (int8)
    zipformer-small-en         55MB   Streaming, EN, fast/lightweight
    zipformer-bilingual-zh-en 370MB   Streaming, ZH+EN

  Backend: moonshine
    moonshine-tiny             34MB   Streaming, EN, 34M params
    moonshine-small           123MB   Streaming, EN, 123M params
    moonshine-medium          245MB   Streaming, EN, 245M params

  Backend: mlx-whisper (macOS Apple Silicon)
    mlx-whisper-large-v3-turbo 800MB  Batch, multilingual, 809M params (recommended)
    mlx-whisper-large-v3       1.5GB  Batch, multilingual, highest accuracy
    mlx-whisper-small.en       240MB  Batch, EN, fast
    mlx-whisper-medium-8bit    375MB  Batch, multilingual, 8-bit quantized

  * = installed
```

The `mlx-whisper` section only appears on macOS Apple Silicon systems. On other platforms it is hidden.

#### `rift-local info`

Show details about a specific model.

```
rift-local info nemotron-streaming-en

  Name:       nemotron-streaming-en
  Display:    Nemotron Streaming EN 0.6B (int8)
  Backend:    sherpa-onnx
  Params:     0.6B
  Size:       600MB
  Languages:  en
  Streaming:  yes
  Cached:     yes (~/.cache/rift-local/models/nemotron-streaming-en/)
```

#### `rift-local cache`

Manage downloaded models.

```
rift-local cache [OPTIONS]

Options:
  --clear NAME        Remove a specific model from cache

rift-local cache

  nemotron-streaming-en     598MB   ~/.cache/rift-local/models/nemotron-streaming-en/
  moonshine-medium          243MB   (huggingface hub)
  Total: 841MB

rift-local cache --clear nemotron-streaming-en
  Removed nemotron-streaming-en (598MB freed)
```

For HuggingFace models, `rift-local cache --clear` advises using `huggingface-cli cache` instead, since those models may be shared with other tools.

---

## Model Management

### Cache location

| Model source                              | Cache directory                            | Managed by                |
| ----------------------------------------- | ------------------------------------------ | ------------------------- |
| Sherpa-onnx models (GitHub releases)      | `~/.cache/rift-local/models/{model-name}/` | rift-local                |
| HuggingFace models (Moonshine, Qwen3-ASR) | `~/.cache/huggingface/hub/`                | `huggingface_hub` library |
| MLX models (mlx-whisper, mlx-lm)          | `~/.cache/huggingface/hub/`                | `huggingface_hub` library |

Sherpa-onnx models are distributed as tarballs on GitHub releases, not on HuggingFace. rift-local downloads and extracts these to its own cache directory. HuggingFace models (including all MLX models from the mlx-community org) use the standard `huggingface_hub` library, which handles caching, versioning, and deduplication automatically. This follows the same pattern as WhisperLiveKit, speaches, and other Python ASR servers.

### Model registry

rift-local maintains an internal registry mapping friendly model names to download sources and file layouts:

```python
MODELS = {
    "nemotron-streaming-en": {
        "backend": "sherpa-onnx",
        "source": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14.tar.bz2",
        "files": {
            "tokens": "tokens.txt",
            "encoder": "encoder.int8.onnx",
            "decoder": "decoder.int8.onnx",
            "joiner": "joiner.int8.onnx",
        },
        "display": "Nemotron Streaming EN 0.6B (int8)",
        "params": "0.6B",
        "languages": ["en"],
        "size_mb": 600,
    },
    "moonshine-medium": {
        "backend": "moonshine",
        "source": "usefulsensors/moonshine-medium",  # HuggingFace repo ID
        "display": "Moonshine Medium (245M)",
        "params": "245M",
        "languages": ["en"],
        "size_mb": 245,
    },
    "mlx-whisper-large-v3-turbo": {
        "backend": "mlx-whisper",
        "source": "mlx-community/whisper-large-v3-turbo",  # HuggingFace repo ID
        "display": "Whisper Large V3 Turbo (MLX)",
        "params": "809M",
        "languages": ["multilingual"],
        "streaming": False,
        "size_mb": 800,
        "platform": "darwin",  # macOS only
    },
}
```

The registry ships with the package and is updated with new releases. Users do not edit it.

### Auto-download

When any command references a model that is not cached (`serve`, `transcribe`, `info`):

1. Print model name, size, and download source
2. Download with a progress bar (using `httpx` or `huggingface_hub` depending on source)
3. For tarballs: extract to `~/.cache/rift-local/models/{model-name}/`
4. For HF models: `huggingface_hub.snapshot_download()` handles everything
5. Verify expected files exist
6. Proceed with the command

Subsequent runs skip download entirely.

---

## Backends

Each backend is a Python class implementing a common adapter interface:

```python
class BackendAdapter:
    """Interface that all rift-local backends implement."""

    def create_stream(self) -> Stream: ...
    def feed_audio(self, stream: Stream, samples: np.ndarray) -> None: ...
    def is_ready(self, stream: Stream) -> bool: ...
    def decode(self, stream: Stream) -> None: ...
    def get_result(self, stream: Stream) -> Result: ...
    def is_endpoint(self, stream: Stream) -> bool: ...
    def reset(self, stream: Stream) -> None: ...
    def get_info(self) -> dict: ...
```

### sherpa-onnx

Uses `sherpa_onnx.OnlineRecognizer` via pip-installable `sherpa-onnx` Python bindings.

The adapter calls the recognizer directly and serializes the full `OnlineRecognizerResult` -- `text`, `tokens`, `timestamps`, `ys_probs`, `lm_probs`, `context_scores`, `start_time`, `segment`, `is_final` -- all fields that the upstream Python WS server discards.

```python
result = recognizer.get_result(stream)
is_endpoint = recognizer.is_endpoint(stream)

message = {
    "type": "result",
    "text": result.text,
    "tokens": list(result.tokens),
    "timestamps": list(result.timestamps),
    "ys_probs": list(result.ys_probs),
    "lm_probs": list(result.lm_probs),
    "context_scores": list(result.context_scores),
    "start_time": result.start_time,
    "segment": segment,
    "is_final": is_endpoint,
    "model": model_name,
}
```

**Install:** `pip install rift-local[sherpa]` (sherpa-onnx is an optional dependency since it pulls in ONNX Runtime).

### Moonshine

Uses `moonshine` Python package. Moonshine's `MicTranscriber` emits `LineStarted`, `LineTextChanged`, and `LineCompleted` events. The adapter maps these to the rift-local protocol:

| Moonshine event   | rift-local mapping                                                        |
| ----------------- | ------------------------------------------------------------------------- |
| `LineStarted`     | New `segment` ID                                                          |
| `LineTextChanged` | `is_final: false` (interim -- note: Moonshine interims are non-monotonic) |
| `LineCompleted`   | `is_final: true` (endpoint)                                               |

Moonshine does not provide per-token timestamps or log-probs. Those fields will be absent from results. The `info` handshake will report `features.timestamps: false` and `features.confidence: false` so RIFT can adapt its UI (e.g. disable per-word confidence coloring).

**Install:** `pip install rift-local[moonshine]`

### mlx-whisper (macOS only)

Uses Apple's [`mlx-whisper`](https://pypi.org/project/mlx-whisper/) package, which runs OpenAI Whisper models natively on Apple Silicon via the [MLX](https://github.com/ml-explore/mlx) framework. MLX uses Apple Silicon's unified memory directly -- no Metal translation layer, no ONNX Runtime overhead. On an M4 Mac Mini, mlx-whisper transcribes significantly faster than CPU-based ONNX or whisper.cpp for batch workloads.

**Batch only.** Whisper is an encoder-decoder model that processes complete audio segments, not a streaming frame-by-frame architecture. The mlx-whisper adapter does not serve the WS streaming path. It serves:

- `POST /transcribe` (HTTP file upload)
- `rift-local transcribe` (CLI)
- WS batch-from-buffer (buffers all frames until `"Done"`, then runs `mlx_whisper.transcribe()` on the complete audio)

The `info` handshake reports `streaming: false` so RIFT knows not to use this backend for live mic input.

```python
import mlx_whisper

result = mlx_whisper.transcribe(
    audio_path_or_array,
    path_or_hf_repo=model_repo,
    word_timestamps=True,
)

# result["segments"][0]:
# {
#   "text": "Hello world",
#   "start": 0.0, "end": 1.5,
#   "words": [
#     {"word": " Hello", "start": 0.32, "end": 0.64, "probability": 0.98},
#     {"word": " world", "start": 0.64, "end": 1.1, "probability": 0.95},
#   ]
# }
```

The adapter maps Whisper's segment/word structure to rift-local's result format:

| Whisper field       | rift-local field | Notes                                           |
| ------------------- | ---------------- | ----------------------------------------------- |
| `segment.text`      | `text`           | Direct mapping                                  |
| `word.word`         | `tokens[]`       | Whisper words are whitespace-delimited, not BPE |
| `word.start`        | `timestamps[]`   | Word-level timestamps                           |
| `word.probability`  | `ys_probs[]`     | Converted to log-prob: `log(probability)`       |
| `segment.start`     | `start_time`     | Segment start                                   |
| sequential index    | `segment`        | Monotonic segment ID                            |
| always true (batch) | `is_final`       | Batch results are always final                  |

**Available models** (all from [mlx-community](https://huggingface.co/collections/mlx-community/whisper-663256f9964fbb1177db93dc) on HuggingFace):

| Registry name                | HuggingFace repo                        | Size   | Notes                                    |
| ---------------------------- | --------------------------------------- | ------ | ---------------------------------------- |
| `mlx-whisper-large-v3-turbo` | `mlx-community/whisper-large-v3-turbo`  | ~800MB | Best speed/quality tradeoff, recommended |
| `mlx-whisper-large-v3`       | `mlx-community/whisper-large-v3-mlx`    | ~1.5GB | Highest accuracy, multilingual           |
| `mlx-whisper-small.en`       | `mlx-community/whisper-small.en-mlx`    | ~240MB | Fast, English only                       |
| `mlx-whisper-medium`         | `mlx-community/whisper-medium-mlx-8bit` | ~375MB | 8-bit quantized, good balance            |

Models auto-download from HuggingFace on first use via `mlx_whisper`'s built-in model resolution (which uses `huggingface_hub` internally).

**Platform guard:** On non-macOS systems (or Macs without Apple Silicon), `pip install rift-local[mlx]` will fail at the `mlx` dependency. The mlx-whisper backend simply won't be available. rift-local detects this at startup and excludes mlx models from `rift-local list`. If a user explicitly requests an mlx model on an unsupported platform, rift-local prints a clear error: `"mlx-whisper requires macOS on Apple Silicon. Install sherpa-onnx or moonshine instead."`

**Install:** `pip install rift-local[mlx]` (pulls in `mlx-whisper`, `mlx-lm`, and their shared `mlx` core dependency).

### Future ASR backends

| Backend          | Engine                          | Notes                                                                                   |
| ---------------- | ------------------------------- | --------------------------------------------------------------------------------------- |
| `qwen3-asr`      | vLLM with OpenAI-compatible API | Adapter translates between vLLM's streaming HTTP/SSE and rift-local's WS protocol       |
| `qwen3-asr-mlx`  | mlx-lm with Qwen3-ASR weights   | MLX-converted Qwen3-ASR models (0.6B, 1.7B) available on mlx-community HF org           |
| `parakeet-mlx`   | Custom MLX inference            | NVIDIA Parakeet CTC/TDT models converted to MLX (high accuracy, potentially streamable) |
| `nemotron-cpp`   | nemotron-asr.cpp subprocess     | Pure C++/ggml, no Python deps; adapter communicates via stdin/stdout pipe               |
| `faster-whisper` | faster-whisper library          | Offline (not streaming), but useful for high-accuracy batch re-transcription            |

New backends are added as optional dependency groups: `pip install rift-local[qwen]`, etc.

### LLM backend

LLM transforms use a separate backend from ASR. Options:

| Backend                      | Access                       | Notes                                                          |
| ---------------------------- | ---------------------------- | -------------------------------------------------------------- |
| mlx-lm (macOS)               | Python bindings (in-process) | Native Apple Silicon, shares `mlx` core with mlx-whisper       |
| Ollama                       | HTTP API (`localhost:11434`) | User runs Ollama separately; rift-local calls its API          |
| llama.cpp (llama-cpp-python) | Python bindings              | Bundled inference, no separate process, cross-platform         |
| OpenAI-compatible API        | HTTP                         | Any local server with OpenAI-compatible `/v1/chat/completions` |

The LLM backend is configured via `--llm` flag with a prefix that selects the backend:

```
--llm mlx:llama-3.2-3b          # mlx-lm, downloads from mlx-community HF org
--llm ollama:llama3.2:3b         # Ollama HTTP API (Ollama must be running)
--llm openai:http://localhost:8080  # Any OpenAI-compatible server
```

If no prefix is given, rift-local auto-selects: `mlx-lm` on Apple Silicon (if installed), Ollama if running, otherwise error with install instructions.

#### mlx-lm

[`mlx-lm`](https://pypi.org/project/mlx-lm/) is Apple's official LLM inference package for MLX. Actively maintained (v0.30.7, Feb 2026), supports streaming generation, prompt caching, and thousands of quantized models from the [mlx-community](https://huggingface.co/mlx-community) HuggingFace org.

```python
from mlx_lm import load, stream_generate

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
messages = [{"role": "user", "content": prompt}]
formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

for response in stream_generate(model, tokenizer, formatted, max_tokens=2048):
    yield response.text
```

**Why mlx-lm over the alternatives on Apple Silicon:**

- **In-process, no daemon.** Unlike Ollama, no separate process. rift-local loads the LLM directly. One process, one set of memory.
- **Native MLX.** llama-cpp-python uses Metal as a backend; mlx-lm _is_ the native Apple Silicon framework. No translation layers.
- **Shared runtime.** If mlx-whisper handles ASR and mlx-lm handles transforms, the MLX framework is loaded once. Both packages depend on `mlx` core.
- **Prompt caching.** For repeated transforms with the same system prompt (e.g., "Fix punctuation" applied to many segments), mlx-lm caches the prompt KV state. Subsequent calls skip recomputing the system prompt.

**Recommended models for transforms** (on M4 Mac Mini with 16GB):

| Model                      | Size (4-bit) | Good for                    |
| -------------------------- | ------------ | --------------------------- |
| Llama-3.2-3B-Instruct-4bit | ~1.7 GB      | General transforms, default |
| Qwen3-4B-4bit              | ~2.2 GB      | Multilingual transforms     |
| Gemma-3-4B-it-4bit         | ~2.2 GB      | Instruction following       |
| Llama-3.1-8B-Instruct-4bit | ~4.3 GB      | Higher quality, still fast  |

All models auto-download from HuggingFace on first use.

**Platform guard:** Same as mlx-whisper -- macOS Apple Silicon only. On other platforms, use Ollama or llama-cpp-python instead. Installed via `pip install rift-local[mlx]` (shared extra with mlx-whisper).

---

## Batch Transcription

Batch transcription uses the same core pipeline as streaming. The difference is how audio arrives and how results are delivered.

### From RIFT (WebSocket)

When RIFT re-transcribes a segment or full recording, the audio is already in the browser's memory as Float32 PCM. RIFT sends it over the existing WebSocket connection at read speed (faster than real-time). The server processes frames identically to live streaming -- same code path, same result format. The only difference is timing: frames arrive faster.

This is the preferred path for RIFT because:

- The WS connection is already open from the live session
- Audio is already in Float32 PCM (no encoding/decoding needed)
- Results stream back incrementally (RIFT can show progress)
- No file I/O or multipart encoding overhead

### From external tools (HTTP)

External callers (scripts, CLI, CI pipelines, `curl`) use `POST /transcribe` with an audio file. The HTTP handler:

1. Receives the uploaded file
2. Decodes it to Float32 PCM (using `soundfile` or `ffmpeg` -- supports wav, mp3, m4a, ogg, flac)
3. Feeds it through the same pipeline as the WS handler
4. Collects all results
5. Returns a single JSON response

This path adds audio decoding (which the WS path doesn't need) but provides the simpler request/response interface that external tools expect.

### From CLI (direct)

`rift-local transcribe file.wav` skips the server entirely. It loads the model, decodes the file, runs the pipeline, and prints results. Same pipeline, no network hop.

---

## LLM Transforms

Text transforms powered by a local LLM. Available via HTTP endpoint (`POST /transform`) and CLI command (`rift-local transform`).

### What transforms are for

Raw ASR output often needs cleanup:

| Problem             | Transform prompt                           |
| ------------------- | ------------------------------------------ |
| Missing punctuation | "Add punctuation and capitalize sentences" |
| Filler words        | "Remove um, uh, like, you know"            |
| Formatting          | "Add paragraph breaks at topic changes"    |
| Summarization       | "Summarize this meeting in bullet points"  |
| Action items        | "Extract action items as a checklist"      |

These are LLM tasks -- too complex for regex but straightforward for a 3B-parameter model running locally.

### What transforms are NOT for

Deterministic text operations (regex find/replace, case conversion, custom word list substitution) are **script-based transforms** that run in RIFT's browser UI. These are fast (milliseconds), predictable, and don't need a model. They stay in the browser -- rift-local does not implement them.

The boundary is clear: if it needs a model, it's an LLM transform (rift-local). If it's a deterministic string operation, it's a script transform (RIFT browser).

### Pipeline integration

Transforms compose naturally with transcription:

```bash
# CLI: transcribe then transform in one command
rift-local transcribe meeting.wav --llm mlx:llama-3.2-3b \
  --transform "Fix punctuation and add paragraphs"

# CLI: pipe through transform separately
rift-local transcribe meeting.wav | rift-local transform \
  --llm mlx:llama-3.2-3b --prompt "Extract action items"

# RIFT app: user selects text, clicks "Transform", types prompt
# RIFT sends POST /transform to running rift-local server
```

---

## Phased Rollout

### Phase 1: Streaming ASR bridge (validates protocol)

- FastAPI server with WS endpoint
- sherpa-onnx backend adapter with full field serialization
- `info` handshake with model metadata
- Model registry with 2-3 sherpa models (Nemotron, Zipformer small, Zipformer bilingual)
- Auto-download for sherpa GitHub release tarballs
- `serve`, `list` CLI commands
- **Validates against:** existing RIFT `sherpa.svelte.ts` client (minor update to handle `type` field and display `model`)

### Phase 2: Batch + Moonshine + MLX Whisper + model metadata in RIFT UI

- HTTP `POST /transcribe` endpoint (file upload with audio decoding)
- HTTP `GET /info` endpoint
- `rift-local transcribe` CLI command
- Moonshine backend adapter
- mlx-whisper backend adapter (macOS Apple Silicon, batch only)
- HuggingFace Hub integration for model downloads (shared by Moonshine and MLX models)
- Platform detection: show/hide MLX models based on system capabilities
- RIFT client displays model name from `info` handshake
- `info`, `cache` CLI commands
- `pip install rift-local[mlx]` optional extra

### Phase 3: LLM transforms

- HTTP `POST /transform` endpoint
- `rift-local transform` CLI command
- `--transform` flag on `rift-local transcribe`
- mlx-lm backend for Apple Silicon (in-process, shares `mlx` runtime with mlx-whisper)
- Ollama backend (HTTP API, cross-platform fallback)
- `--llm` flag with backend prefix (`mlx:`, `ollama:`, `openai:`)
- Auto-detection: prefer mlx-lm on Apple Silicon, Ollama elsewhere

### Phase 4: Polish

- `--device` flag (CUDA, CoreML, MLX auto-detection)
- SRT/VTT output format for `transcribe`
- Error messages for missing optional dependencies ("Install MLX support: `pip install rift-local[mlx]`")
- Graceful handling of model download interruptions (resume partial downloads)
- Additional ASR backends (faster-whisper for high-accuracy batch, Qwen3-ASR, Parakeet MLX)

---

## What rift-local Does NOT Do

- **Cloud provider auth/CORS.** Cloud providers (Deepgram, Soniox, ElevenLabs, etc.) are handled by RIFT's SvelteKit API routes or browser-direct connections. rift-local is for local models only.
- **Audio processing.** No resampling, VAD, noise reduction, or transcoding for the WS path. RIFT's AudioWorklet sends ready-to-use 16kHz Float32 PCM. (The HTTP path does decode audio files, but does not apply VAD or noise reduction.)
- **Script-based transforms.** Deterministic text operations (regex, case conversion, word lists) run in RIFT's browser UI as JS/TS functions. rift-local handles LLM transforms only.
- **Multi-user serving.** Single audio stream, single model, single user. Not a production ASR API server.
- **GPU management.** Does not auto-select GPU, manage VRAM, or swap models on demand. One model loaded at startup, use `--device` to pick compute target. (MLX backends use Apple Silicon unified memory automatically -- no device selection needed.)

---

## Prior Art

| Project                                                                           | Stars | Approach                                           | Relevance to rift-local                         |
| --------------------------------------------------------------------------------- | ----- | -------------------------------------------------- | ----------------------------------------------- |
| [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit)                   | 9.7k  | FastAPI + WebSocket, auto-downloads via HF Hub     | UX model for `pip install` + `serve` pattern    |
| [speaches](https://github.com/speaches-ai/speaches)                               | 2.9k  | OpenAI-compatible API, dynamic model loading       | Model registry and `model_aliases.json` pattern |
| [vosk-server](https://github.com/alphacep/vosk-server)                            | 1.2k  | WS/gRPC/WebRTC server for Vosk/Kaldi               | Multi-protocol local ASR server precedent       |
| [whisper_streaming](https://github.com/ufal/whisper_streaming)                    | 3.5k  | TCP streaming with LocalAgreement policy           | Streaming buffering strategies                  |
| [mlx-whisper](https://pypi.org/project/mlx-whisper/)                              | --    | Apple's Whisper on MLX, HF Hub models              | ASR backend for Apple Silicon batch path        |
| [mlx-lm](https://pypi.org/project/mlx-lm/)                                        | --    | Apple's LLM inference on MLX, streaming generation | LLM backend for Apple Silicon transforms        |
| [lightning-whisper-mlx](https://github.com/mustafaaljadery/lightning-whisper-mlx) | 870   | Batched MLX Whisper with distilled model support   | Performance benchmark for MLX Whisper approach  |
| sherpa-onnx Python server                                                         | --    | Built-in, 500 lines, drops most fields             | What rift-local replaces                        |
