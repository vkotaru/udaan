"""Colored console output utilities.

This module provides colored print functions for backward compatibility.
New code should prefer using the logging module directly.

Example:
    >>> from udaan.utils import printc_warn, printc_fail, printc_ok
    >>> printc_warn("This is a warning")
    >>> printc_fail("This is an error")
    >>> printc_ok("Success!")
"""

from __future__ import annotations

import sys

from .logging import get_logger

_logger = get_logger("printout")


class bcolors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def _supports_color() -> bool:
    """Check if the terminal supports color output."""
    if not hasattr(sys.stdout, "isatty"):
        return False
    return sys.stdout.isatty()


def printc(text: str, color: str) -> None:
    """Print text with ANSI color.

    Args:
        text: Text to print.
        color: ANSI color code from bcolors.
    """
    if _supports_color():
        print(color + text + bcolors.ENDC)
    else:
        print(text)


def printc_warn(text: str) -> None:
    """Print warning message in yellow.

    Also logs the message at WARNING level.

    Args:
        text: Warning message to print.
    """
    _logger.warning(text)
    printc(text, bcolors.WARNING)


def printc_fail(text: str) -> None:
    """Print error message in red.

    Also logs the message at ERROR level.

    Args:
        text: Error message to print.
    """
    _logger.error(text)
    printc(text, bcolors.FAIL)


def printc_ok(text: str) -> None:
    """Print success message in green.

    Also logs the message at INFO level.

    Args:
        text: Success message to print.
    """
    _logger.info(text)
    printc(text, bcolors.OKGREEN)
