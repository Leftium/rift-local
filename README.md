# rift-local

Local inference server for [RIFT Transcription](https://github.com/Leftium/rift-transcription). Serves streaming speech recognition over WebSocket, backed by local models with automatic download.

## Install

```
pip install rift-local
```

### Backend extras

rift-local supports multiple ASR backends, each installed as an optional extra:

```bash
pip install rift-local[sherpa]      # sherpa-onnx (Nemotron, Zipformer)
pip install rift-local[moonshine]   # Moonshine Gen 2 (via moonshine-voice)
pip install rift-local[sherpa,moonshine]  # both
```

On Apple Silicon, add MLX support for future GPU-accelerated batch transcription:

```bash
pip install rift-local[mlx]
```

For development (includes pytest):

```bash
pip install rift-local[dev]
```

## Models

List all available models and see which are installed:

```
rift-local list
rift-local list --installed
```

### sherpa-onnx models

| Model | Params | Languages | Download | Notes |
|-------|--------|-----------|----------|-------|
| `nemotron-streaming-en` | 0.6B | EN | 447 MB | Best accuracy. |
| `zipformer-small-en` | ~30M | EN | 296 MB | Lightweight, fast. |
| `zipformer-bilingual-zh-en` | ~70M | ZH, EN | 487 MB | Bilingual Chinese + English. |

Requires: `pip install rift-local[sherpa]`

### Moonshine models

| Model | Params | Languages | Size | Notes |
|-------|--------|-----------|------|-------|
| `moonshine-tiny-en` | 34M | EN | 26 MB | Fastest. Good for low-resource. |
| `moonshine-small-en` | 123M | EN | 95 MB | Balanced speed/accuracy. |
| `moonshine-medium-en` | 245M | EN | 190 MB | **Default.** Best Moonshine accuracy. |

Requires: `pip install rift-local[moonshine]`

Moonshine models are downloaded automatically by the `moonshine-voice` library on first use.

## Usage

### Server mode (for RIFT app)

Start the WebSocket server with any model:

```bash
# Moonshine (default)
rift-local serve

# sherpa-onnx
rift-local serve --model nemotron-streaming-en

# Custom host/port
rift-local serve --model moonshine-tiny-en --host 0.0.0.0 --port 8080
```

The server auto-downloads the model on first run, then listens on:
- **WebSocket**: `ws://127.0.0.1:2177/ws` (streaming ASR)
- **HTTP**: `http://127.0.0.1:2177/info` (model metadata)

### Server options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `moonshine-medium-en` | Model name from registry |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `2177` | Server port |
| `--threads` | `2` | Inference threads |

## WebSocket protocol

1. Client connects to `/ws`
2. Server sends `info` JSON (model name, features, sample rate)
3. Client sends binary frames of Float32 PCM audio at 16 kHz
4. Server sends `result` JSON messages with partial/final transcriptions
5. Client sends text `"Done"` to end the session

## Running tests

```bash
# Install dev + backend dependencies
pip install -e ".[dev,sherpa,moonshine]"

# Run fast tests (mocked backends, no model download)
pytest

# Run all tests including slow integration tests (downloads models)
pytest --slow
```

Tests are in the `tests/` directory:
- `test_server.py` — WebSocket server tests using a mock backend
- `test_moonshine.py` — Moonshine adapter unit tests (mocked) + integration tests (slow)
- `conftest.py` — Shared `MockBackend` fixture and `--slow` flag

## Spec

See [specs/rift-local.md](specs/rift-local.md) for the full design document.

## License

[MIT](LICENSE)
