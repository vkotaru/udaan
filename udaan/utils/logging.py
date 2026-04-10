"""Structured logging for Udaan library.

This module provides a consistent logging interface for the library.
Library users can configure logging through standard Python logging mechanisms.

Usage:
    # In library code
    from udaan.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Controller initialized")

    # For users to enable logging output
    from udaan.utils.logging import setup_logging
    setup_logging()  # Enable INFO level output
    setup_logging(level=logging.DEBUG)  # Enable debug output
"""

from __future__ import annotations

import logging
import sys
from typing import Any

# Create library root logger
_root_logger = logging.getLogger("udaan")
_root_logger.addHandler(logging.NullHandler())  # Don't force handler on library users


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name.

    Args:
        name: Module name, typically __name__.

    Returns:
        Logger instance under the udaan namespace.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Message")
    """
    # Ensure logger is under udaan namespace
    if name.startswith("udaan."):
        return logging.getLogger(name)
    return logging.getLogger(f"udaan.{name}")


class _ColoredFormatter(logging.Formatter):
    """Formatter that adds ANSI colors to log level names."""

    COLORS = {
        logging.DEBUG: "\033[90m",  # gray
        logging.INFO: "\033[92m",  # green
        logging.WARNING: "\033[93m",  # yellow
        logging.ERROR: "\033[91m",  # red
        logging.CRITICAL: "\033[1;91m",  # bold red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def _supports_color(stream: Any) -> bool:
    """Check if the output stream supports ANSI colors."""
    return hasattr(stream, "isatty") and stream.isatty()


def setup_logging(
    level: int = logging.INFO,
    format_string: str | None = None,
    stream: Any = None,
    color: bool | None = None,
) -> None:
    """Configure logging for interactive use.

    Call this function to enable logging output from Udaan.
    By default, the library produces no output (NullHandler).

    Args:
        level: Logging level (default: INFO).
        format_string: Custom format string. If None, uses default format.
        stream: Output stream (default: sys.stdout).
        color: Enable colored output. None = auto-detect from terminal.

    Example:
        >>> import logging
        >>> from udaan.utils.logging import setup_logging
        >>> setup_logging()  # Enable INFO level with colors
        >>> setup_logging(level=logging.DEBUG)  # Enable debug
    """
    if format_string is None:
        format_string = "[%(levelname)s] %(name)s: %(message)s"

    if stream is None:
        stream = sys.stdout

    if color is None:
        color = _supports_color(stream)

    handler = logging.StreamHandler(stream)
    if color:
        handler.setFormatter(_ColoredFormatter(format_string))
    else:
        handler.setFormatter(logging.Formatter(format_string))

    _root_logger.addHandler(handler)
    _root_logger.setLevel(level)


class LoggerMixin:
    """Mixin class providing logging capabilities to classes.

    Inherit from this mixin to get a _logger property that returns
    a logger named after the class.

    Example:
        >>> class MyController(LoggerMixin):
        ...     def compute(self):
        ...         self._logger.debug("Computing...")
    """

    @property
    def _logger(self) -> logging.Logger:
        """Get logger for this class instance."""
        return logging.getLogger(f"udaan.{self.__class__.__module__}.{self.__class__.__name__}")

    def _log_debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug message."""
        self._logger.debug(msg, *args, **kwargs)

    def _log_info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an info message."""
        self._logger.info(msg, *args, **kwargs)

    def _log_warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning message."""
        self._logger.warning(msg, *args, **kwargs)

    def _log_error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message."""
        self._logger.error(msg, *args, **kwargs)
