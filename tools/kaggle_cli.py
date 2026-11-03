"""Run the Kaggle CLI without mixing its import-time output into command data."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import redirect_stdout
import importlib
import sys
from typing import cast


def main() -> None:
    """Import the official entry point safely, then run it with its arguments."""
    # Its anonymous-access check recognises the `kaggle` executable name;
    # preserve that identity while leaving command arguments intact.
    original_program = sys.argv[0] if sys.argv else None
    if original_program is not None:
        sys.argv[0] = "kaggle"
    try:
        with redirect_stdout(sys.stderr):
            cli = importlib.import_module("kaggle.cli")
        entry_point = cast(Callable[[], None], getattr(cli, "main"))
        entry_point()
    finally:
        if original_program is not None:
            sys.argv[0] = original_program


if __name__ == "__main__":
    main()
