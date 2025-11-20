"""XLA-safe logging utilities.

This module provides logging functions that avoid XLA graph breaks
by deferring actual printing until after graph execution.
"""

import sys
from typing import Any


# Global flag to enable/disable XLA-safe logging
_XLA_MODE = False
_LOG_BUFFER: list[str] = []


def set_xla_mode(enabled: bool) -> None:
    """Enable or disable XLA-safe logging mode.

    Args:
        enabled: If True, print statements are buffered to avoid graph breaks.
    """
    global _XLA_MODE
    _XLA_MODE = enabled


def is_xla_mode() -> bool:
    """Check if XLA-safe mode is enabled.

    Returns:
        True if XLA mode is enabled, False otherwise.
    """
    return _XLA_MODE


def xla_print(*args: Any, **kwargs: Any) -> None:
    """Print function that respects XLA mode.

    In XLA mode, messages are buffered instead of printed immediately.
    Otherwise, behaves like standard print().

    Args:
        *args: Arguments to print.
        **kwargs: Keyword arguments passed to print().
    """
    message = " ".join(str(arg) for arg in args)

    if _XLA_MODE:
        # Buffer the message instead of printing
        _LOG_BUFFER.append(message)
    else:
        # Print normally
        print(message, **kwargs)


def flush_xla_logs(file: Any = None) -> None:
    """Flush buffered XLA logs.

    Args:
        file: File to write to (defaults to sys.stdout).
    """
    global _LOG_BUFFER

    if not _LOG_BUFFER:
        return

    if file is None:
        file = sys.stdout

    for message in _LOG_BUFFER:
        print(message, file=file)

    _LOG_BUFFER.clear()


def clear_xla_logs() -> None:
    """Clear buffered XLA logs without printing."""
    global _LOG_BUFFER
    _LOG_BUFFER.clear()


def get_buffered_logs() -> list[str]:
    """Get the current buffered logs without clearing them.

    Returns:
        List of buffered log messages.
    """
    return _LOG_BUFFER.copy()
