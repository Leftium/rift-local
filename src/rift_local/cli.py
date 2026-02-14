"""rift-local command-line interface."""

from __future__ import annotations

import argparse
import sys

from rift_local import __version__
from rift_local.types import DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_PORT


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rift-local",
        description="Local inference server for RIFT Transcription.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"rift-local {__version__}",
    )

    sub = parser.add_subparsers(dest="command")

    # -- serve --------------------------------------------------------------
    serve_parser = sub.add_parser("serve", help="Start the WebSocket + HTTP server.")
    serve_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"ASR model to load (see 'rift-local list')  [default: {DEFAULT_MODEL}]",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Server port (WS + HTTP)  [default: {DEFAULT_PORT}]",
    )
    serve_parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Bind address  [default: {DEFAULT_HOST}]",
    )
    serve_parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Number of inference threads  [default: 2]",
    )

    # -- list ---------------------------------------------------------------
    list_parser = sub.add_parser("list", help="Show available models.")
    list_parser.add_argument(
        "--installed",
        action="store_true",
        help="Show only downloaded models with cache sizes.",
    )

    # -- dispatch -----------------------------------------------------------
    args = parser.parse_args()

    if args.command == "serve":
        _cmd_serve(args)
    elif args.command == "list":
        _cmd_list(args)
    else:
        parser.print_help()
        sys.exit(0)


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def _cmd_serve(args: argparse.Namespace) -> None:
    """Start the rift-local server."""
    import uvicorn

    from rift_local.models import get_model
    from rift_local.server import create_app

    entry = get_model(args.model)
    backend = _create_backend(entry, threads=args.threads)

    app = create_app(backend)
    info = backend.get_info()

    print(f"rift-local v{__version__}")
    print(f"Model:   {info.model_display}")
    print(f"Backend: {info.backend}")
    print(f"Server:  http://{args.host}:{args.port}")
    print(f"WS:      ws://{args.host}:{args.port}/ws")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def _create_backend(entry, *, threads: int = 2):
    """Instantiate the appropriate backend adapter for *entry*."""
    if entry.backend == "sherpa-onnx":
        from rift_local.backends.sherpa import SherpaAdapter
        from rift_local.models import ensure_model

        model_dir = ensure_model(entry.name)
        return SherpaAdapter(
            entry,
            model_dir,
            num_threads=threads,
            sample_rate=16_000,
        )

    if entry.backend == "moonshine":
        from rift_local.backends.moonshine import (
            MoonshineAdapter,
            ensure_moonshine_model,
        )

        model_path, model_arch = ensure_moonshine_model(entry)
        return MoonshineAdapter(
            entry,
            model_path=model_path,
            model_arch=model_arch,
            sample_rate=16_000,
        )

    msg = f"Unknown backend {entry.backend!r} for model {entry.name!r}"
    raise ValueError(msg)


def _cmd_list(args: argparse.Namespace) -> None:
    """Print available models."""
    from rift_local.models import is_cached, list_models

    models = list_models()

    if args.installed:
        models = [m for m in models if is_cached(m.name)]
        if not models:
            print("No models installed. Run 'rift-local serve' to download one.")
            return

    # Group by backend.
    by_backend: dict[str, list] = {}
    for m in models:
        by_backend.setdefault(m.backend, []).append(m)

    print("Available models:\n")
    for backend, entries in by_backend.items():
        print(f"  Backend: {backend}")
        for m in entries:
            marker = "*" if is_cached(m.name) else " "
            langs = ", ".join(m.languages).upper()
            dl = (
                f" ({m.download_mb}MB dl)"
                if m.download_mb and m.download_mb != m.size_mb
                else ""
            )
            print(
                f"  {marker} {m.name:<30s} {m.size_mb:>5d}MB{dl}   {m.display}  [{langs}]"
            )
        print()
    print("  * = installed")


if __name__ == "__main__":
    main()
