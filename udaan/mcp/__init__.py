"""MCP server for udaan: drive every simulated system from an MCP client such as Claude.

- :mod:`udaan.mcp.catalog` — the systems, their parameters and bounds, and
  argument validation.
- :mod:`udaan.mcp.runner` — headless runs in a child process: ``simulate``,
  ``render_video``, ``generate_xml``, ``check_reference``.
- :mod:`udaan.mcp.server` — the MCP tools over stdio (needs the ``mcp`` extra:
  ``pip install "udaan[mcp]"``).

``catalog`` and ``runner`` work without the ``mcp`` SDK; only ``server`` needs
it, and this package does not import it, so a run's child process does not load
the SDK either. :data:`MCP_AVAILABLE` says whether the SDK (2.x) is installed.
The ``udaan-mcp`` console script calls :func:`main`.
"""

from __future__ import annotations

import importlib.util
import sys

from . import catalog, runner


def _sdk_available() -> bool:
    try:
        return importlib.util.find_spec("mcp.server.mcpserver") is not None
    except ModuleNotFoundError:
        return False


#: Whether the MCP Python SDK 2.x, which :mod:`udaan.mcp.server` needs, is installed.
MCP_AVAILABLE = _sdk_available()


def main() -> None:
    """Serve over stdio, or exit 1 with the install line when the ``mcp`` SDK is missing."""
    if not MCP_AVAILABLE:
        sys.stderr.write('udaan-mcp needs the MCP Python SDK (mcp 2.x): pip install "udaan[mcp]"\n')
        raise SystemExit(1)
    from . import server

    server.main()


__all__ = ["MCP_AVAILABLE", "catalog", "main", "runner"]
