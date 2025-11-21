"""Threading compatibility utilities for Colab/TPU environments.

This module provides thread-safe wrappers that can be disabled for
environments where threading is problematic (e.g., Google Colab TPU).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any


class NoOpLock:
    """A no-op lock that doesn't actually lock anything.

    Useful for environments where threading is disabled.
    """

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        return True

    def release(self) -> None:
        pass

    def locked(self) -> bool:
        return False

    def __enter__(self) -> "NoOpLock":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class NoOpRLock(NoOpLock):
    """A no-op reentrant lock."""
    pass


class NoOpCondition:
    """A no-op condition variable."""

    def __init__(self, lock: Any = None) -> None:
        self.lock = lock if lock is not None else NoOpLock()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        return True

    def release(self) -> None:
        pass

    def wait(self, timeout: float | None = None) -> bool:
        return True

    def notify(self, n: int = 1) -> None:
        pass

    def notify_all(self) -> None:
        pass

    def __enter__(self) -> "NoOpCondition":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


# Global flag to enable/disable threading
_THREADING_ENABLED = True


def set_threading_enabled(enabled: bool) -> None:
    """Enable or disable threading support globally.

    Args:
        enabled: If False, all locks will be no-ops.
    """
    global _THREADING_ENABLED
    _THREADING_ENABLED = enabled


def is_threading_enabled() -> bool:
    """Check if threading is currently enabled.

    Returns:
        True if threading is enabled, False otherwise.
    """
    return _THREADING_ENABLED


def get_lock() -> threading.Lock | NoOpLock:
    """Get a lock (real or no-op depending on threading mode).

    Returns:
        A real Lock if threading is enabled, otherwise a NoOpLock.
    """
    if _THREADING_ENABLED:
        return threading.Lock()
    else:
        return NoOpLock()


def get_rlock() -> threading.RLock | NoOpRLock:
    """Get a reentrant lock (real or no-op depending on threading mode).

    Returns:
        A real RLock if threading is enabled, otherwise a NoOpRLock.
    """
    if _THREADING_ENABLED:
        return threading.RLock()
    else:
        return NoOpRLock()


def get_condition(lock: Any = None) -> threading.Condition | NoOpCondition:
    """Get a condition variable (real or no-op depending on threading mode).

    Args:
        lock: Optional lock to use with the condition.

    Returns:
        A real Condition if threading is enabled, otherwise a NoOpCondition.
    """
    if _THREADING_ENABLED:
        return threading.Condition(lock)
    else:
        return NoOpCondition(lock)


@contextmanager
def optional_lock(lock: Any):
    """Context manager that only acquires the lock if threading is enabled.

    Args:
        lock: The lock to conditionally acquire.

    Yields:
        None
    """
    if _THREADING_ENABLED and lock is not None:
        with lock:
            yield
    else:
        yield
