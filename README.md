# rift-local

Local inference server for [RIFT Transcription](https://github.com/Leftium/rift-transcription). Serves streaming and batch speech recognition over WebSocket, batch transcription and LLM transforms over HTTP, and works as a standalone CLI transcription tool — all backed by local models with automatic download.

## Install

```
pip install rift-local
```

On Apple Silicon, install with MLX support for native GPU-accelerated batch transcription and LLM transforms:

```
pip install rift-local[mlx]
```

## Usage

### Server mode (for RIFT app)

```
rift-local serve --model nemotron-streaming-en
```

### CLI transcription

```
rift-local transcribe meeting.wav
```

### LLM text transforms

```
rift-local transcribe meeting.wav --llm mlx:llama-3.2-3b \
  --transform "Fix punctuation and add paragraphs"
```

## Spec

See [specs/rift-local.md](specs/rift-local.md) for the full design document.

## License

[MIT](LICENSE)
