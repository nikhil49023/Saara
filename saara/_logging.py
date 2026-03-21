"""
SAARA Logging Utilities

Helpers for consistent logging across the package.
"""

import logging
from typing import Optional, Callable


def setup_logging(level: int = logging.INFO, name: Optional[str] = None) -> logging.Logger:
    """
    Set up logging for SAARA.

    Args:
        level: Logging level (default: INFO)
        name: Logger name (default: 'saara')

    Returns:
        Configured logger instance
    """
    logger_name = name or "saara"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Only add handler if logger doesn't have one
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class ProgressCallback:
    """Base class for progress callbacks."""

    def __call__(self, message: str) -> None:
        """Called with progress message."""
        raise NotImplementedError


class LoggingProgressCallback(ProgressCallback):
    """Callback that logs progress messages."""

    def __init__(self, logger: logging.Logger):
        """Initialize with a logger instance."""
        self.logger = logger

    def __call__(self, message: str) -> None:
        """Log the message at INFO level."""
        self.logger.info(message)


class CallableProgressCallback(ProgressCallback):
    """Wrapper for callable progress callbacks."""

    def __init__(self, callback: Callable[[str], None]):
        """Initialize with a callable."""
        self.callback = callback

    def __call__(self, message: str) -> None:
        """Call the wrapped function."""
        self.callback(message)


def merge_callbacks(*callbacks) -> Optional[ProgressCallback]:
    """
    Merge multiple callbacks into one.

    Args:
        *callbacks: Callback instances or callables

    Returns:
        Merged callback that calls all provided callbacks
    """
    if not callbacks:
        return None

    # Filter out None values
    valid_callbacks = [c for c in callbacks if c is not None]
    if not valid_callbacks:
        return None

    if len(valid_callbacks) == 1:
        return valid_callbacks[0]

    class MergedCallback(ProgressCallback):
        def __init__(self, cbs):
            self.callbacks = cbs

        def __call__(self, message: str) -> None:
            for cb in self.callbacks:
                if isinstance(cb, ProgressCallback):
                    cb(message)
                elif callable(cb):
                    cb(message)

    return MergedCallback(valid_callbacks)
