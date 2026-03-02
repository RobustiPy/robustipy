"""robustipy package initialization.

This module intentionally avoids global warning-hook side effects at import
time. If compact robustipy-only warning formatting is desired, call
`enable_compact_warnings()` explicitly.
"""

from __future__ import annotations

import sys
import warnings

__all__ = ["enable_compact_warnings", "disable_compact_warnings"]

_ORIGINAL_SHOWWARNING = warnings.showwarning


def _robustipy_showwarning(message, category, filename, lineno, file=None, line=None):
    """Format robustipy warnings without file paths."""
    path = str(filename).replace("\\", "/")
    if "robustipy" in path:
        if file is None:
            file = sys.stderr
        try:
            file.write(f"{category.__name__}: {message}\n")
            return
        except Exception:
            # Fall back to the original handler if needed.
            pass
    _ORIGINAL_SHOWWARNING(message, category, filename, lineno, file=file, line=line)


def enable_compact_warnings() -> None:
    """Enable compact formatting for warnings emitted from robustipy modules."""
    if warnings.showwarning is not _robustipy_showwarning:
        warnings.showwarning = _robustipy_showwarning


def disable_compact_warnings() -> None:
    """Restore the original warning formatting handler."""
    if warnings.showwarning is _robustipy_showwarning:
        warnings.showwarning = _ORIGINAL_SHOWWARNING

