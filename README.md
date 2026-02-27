# rift-local

Local inference server for [RIFT Transcription](https://github.com/Leftium/rift-transcription). Serves streaming speech recognition over WebSocket, backed by local models with automatic download.

## Install

Requires Python 3.10+. Install with [uv](https://docs.astral.sh/uv/):

```bash
brew install uv              # macOS (or: curl -LsSf https://astral.sh/uv/install.sh | sh)
uv tool install rift-local
```

Or with pip in a virtual environment:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install rift-local
```

### Backend extras

rift-local supports multiple ASR backends, each installed as an optional extra:

```bash
uv tool install "rift-local[sherpa]"             # sherpa-onnx (Nemotron, Kroko)
uv tool install "rift-local[moonshine]"          # Moonshine Gen 2
uv tool install "rift-local[sherpa,moonshine]"   # both
```

Or with pip (inside a venv):

```bash
pip install rift-local[sherpa]             # sherpa-onnx (Nemotron, Kroko)
pip install rift-local[moonshine]          # Moonshine Gen 2 (via moonshine-voice)
pip install rift-local[sherpa,moonshine]   # both
```

For development (includes pytest):

```bash
pip install -e ".[dev]"
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
| `nemotron-en` | 0.6B | EN | 447 MB | Best accuracy. |
| `zipformer-en-kroko` | ~30M | EN | 55 MB | Lightweight, fast. Only ~68 MB on disk. |

Requires: `pip install rift-local[sherpa]`

### Moonshine models

| Model | Params | Languages | Size | Notes |
|-------|--------|-----------|------|-------|
| `moonshine-en-tiny` | 34M | EN | 26 MB | Fastest. Good for low-resource. |
| `moonshine-en-small` | 123M | EN | 95 MB | Balanced speed/accuracy. |
| `moonshine-en-medium` | 245M | EN | 190 MB | **Default.** Best Moonshine accuracy. |

Requires: `pip install rift-local[moonshine]`

Moonshine models are downloaded automatically by the `moonshine-voice` library on first use.

## Usage

### Server mode (for RIFT app)

Start the WebSocket server with any model:

```bash
# Start server and open RIFT Transcription in your browser
rift-local serve --open

# Moonshine (default model)
rift-local serve

# sherpa-onnx
rift-local serve --model nemotron-en

# Custom host/port
rift-local serve --model moonshine-en-tiny --host 0.0.0.0 --port 8080
```

The `--open` flag launches [RIFT Transcription](https://rift-transcription.vercel.app) in your browser, pre-configured to connect to the local server. The voice source is set to "Local" automatically — just click to start the mic.

For local development of the RIFT Transcription client:

```bash
rift-local serve --open dev          # opens http://localhost:5173
rift-local serve --open dev:3000     # custom port
```

The server auto-downloads the model on first run, then listens on:
- **WebSocket**: `ws://127.0.0.1:2177/ws` (streaming ASR)
- **HTTP**: `http://127.0.0.1:2177/info` (model metadata)

### Server options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `moonshine-en-medium` | Model name from registry |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `2177` | Server port |
| `--threads` | `2` | Inference threads |
| `--open` | off | Open browser to RIFT Transcription client |

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
