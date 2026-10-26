"""Command-line interface: argument parsing, REPL, and hook installation."""

from __future__ import annotations

import sys

__all__ = ["configure_utf8_streams"]


def configure_utf8_streams() -> None:
    """Avoid UnicodeEncodeError on legacy Windows code pages.

    Rich renders box-drawing characters and emoji that cp1252 cannot encode.
    Reconfiguring the streams keeps the CLI (and direct REPL use) from
    crashing; terminals that support UTF-8 render them normally.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):  # pragma: no cover - non-file streams
            pass


configure_utf8_streams()
