"""rift-local command-line interface."""

import argparse
import sys

from rift_local import __version__


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

    parser.parse_args()

    # No subcommand given — print help
    parser.print_help()
    sys.exit(0)


if __name__ == "__main__":
    main()
