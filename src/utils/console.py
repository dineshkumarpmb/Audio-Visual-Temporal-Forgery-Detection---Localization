"""Console encoding, so a report character cannot fail a passing run.

Windows consoles default to cp1252 here. Em-dashes survive that, but the status
glyphs used throughout `reports/` (U+26D4 no-entry, U+2705 check) do not, and a
`print()` of one raises `UnicodeEncodeError` *after* all the work is done — turning
a passing gate into a non-zero exit, which is the worst possible failure mode for a
CI check.

`init_console()` switches stdout/stderr to UTF-8 and degrades unmappable characters
to a replacement glyph rather than raising. Files are always written with an explicit
`encoding="utf-8"` and are unaffected.
"""

from __future__ import annotations

import contextlib
import sys


def init_console(encoding: str = "utf-8", errors: str = "replace") -> None:
    """Make stdout/stderr tolerant of non-ASCII. Safe to call more than once."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue  # redirected to something that is not a TextIOWrapper
        # Already detached, or not a reconfigurable wrapper; not worth failing over.
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding=encoding, errors=errors)
