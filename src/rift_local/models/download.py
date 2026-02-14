"""Download and extract model files from remote sources."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import httpx

from rift_local.models.registry import ModelEntry, get_model, is_cached, model_path


def ensure_model(name: str) -> Path:
    """Return the local cache path for *name*, downloading if necessary.

    Raises ``RuntimeError`` if the download fails or expected files are
    missing after extraction.
    """
    entry = get_model(name)
    dest = model_path(name)

    if is_cached(name):
        return dest

    _download_tarball(entry, dest)

    if not is_cached(name):
        missing = [
            fname for fname in entry.files.values() if not (dest / fname).exists()
        ]
        msg = (
            f"Model {name!r} downloaded but expected files are missing: "
            f"{missing}. Contents: {sorted(p.name for p in dest.iterdir())}"
        )
        raise RuntimeError(msg)

    return dest


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _download_tarball(entry: ModelEntry, dest: Path) -> None:
    """Download a ``.tar.bz2`` tarball and extract to *dest*."""
    url = entry.source
    dest.mkdir(parents=True, exist_ok=True)

    dl_mb = entry.download_mb or entry.size_mb
    _log(f"Downloading {entry.display} ({dl_mb} MB download) ...")
    _log(f"  {url}")

    # Download to a temp file, close it fully, then extract.
    fd, tmp_path_str = tempfile.mkstemp(suffix=".tar.bz2")
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "wb") as f:
            _stream_download(url, f)

        # Verify file size matches content-length.
        actual = tmp_path.stat().st_size
        _log(f"Download complete: {actual / 1_048_576:.1f} MB on disk")

        _log("Extracting ...")
        _extract_tarball(tmp_path, dest)
    finally:
        tmp_path.unlink(missing_ok=True)

    _log(f"Model ready: {dest}")


def _stream_download(url: str, dest_file: object) -> None:
    """Stream-download *url* to an open file object with progress output."""
    with httpx.stream(
        "GET", url, follow_redirects=True, timeout=httpx.Timeout(30, read=300)
    ) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        last_pct = -1

        for chunk in response.iter_bytes(chunk_size=131_072):
            dest_file.write(chunk)  # type: ignore[union-attr]
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded * 100 // total
                if pct != last_pct:
                    last_pct = pct
                    _progress(
                        f"\r  {downloaded / 1_048_576:.1f} / {total / 1_048_576:.1f} MB ({pct}%)"
                    )

        if total > 0:
            _progress("\n")

        # Flush to OS before we close.
        dest_file.flush()  # type: ignore[union-attr]
        os.fsync(dest_file.fileno())  # type: ignore[union-attr]

        if total > 0 and downloaded != total:
            msg = f"Incomplete download: got {downloaded} bytes, expected {total}"
            raise RuntimeError(msg)


def _extract_tarball(archive: Path, dest: Path) -> None:
    """Extract *archive* to *dest*, stripping the single top-level directory.

    Sherpa-onnx tarballs contain one top-level directory (e.g.
    ``sherpa-onnx-streaming-zipformer-en-2023-06-26/``).  We strip that
    prefix so files land directly in *dest*.

    Uses the ``tar`` system command if available (handles large bz2 files
    more reliably than Python's tarfile on some platforms), falling back
    to Python's ``tarfile`` module.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        if _has_tar_command():
            subprocess.run(
                ["tar", "xjf", str(archive), "-C", str(tmp)],
                check=True,
            )
        else:
            with tarfile.open(str(archive), "r:bz2") as tf:
                tf.extractall(tmp, filter="data")

        # Sherpa tarballs have a single top-level directory.  Detect it
        # and move its contents into *dest* directly.
        children = list(tmp.iterdir())
        src = children[0] if len(children) == 1 and children[0].is_dir() else tmp

        for item in src.iterdir():
            target = dest / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))


def _has_tar_command() -> bool:
    """Return True if the system ``tar`` command is available."""
    return shutil.which("tar") is not None


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _progress(msg: str) -> None:
    print(msg, end="", file=sys.stderr, flush=True)
