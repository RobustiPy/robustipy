"""Robustipy package initialization.

Keep warning output clean by stripping internal file paths for robustipy
warnings while leaving external warnings unchanged.
"""

from __future__ import annotations

import sys
import warnings

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


def _install_warning_formatter() -> None:
    if warnings.showwarning is not _robustipy_showwarning:
        warnings.showwarning = _robustipy_showwarning


_install_warning_formatter()
