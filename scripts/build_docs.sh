#!/usr/bin/env bash
#
# Build (and optionally serve) the Sphinx documentation.
#
# Usage:
#   scripts/build_docs.sh              # install docs deps + build HTML
#   scripts/build_docs.sh --serve      # build, then serve at http://localhost:8000
#   scripts/build_docs.sh --serve 9000 # serve on a custom port
#   scripts/build_docs.sh --live       # live-reload server (rebuilds on save)
#   scripts/build_docs.sh --clean      # remove previous build first
#
set -euo pipefail

# Resolve repo root from this script's location, so it works from anywhere.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SERVE=0
LIVE=0
CLEAN=0
PORT=8000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --serve) SERVE=1; shift; [[ "${1:-}" =~ ^[0-9]+$ ]] && { PORT="$1"; shift; } ;;
        --live)  LIVE=1; shift; [[ "${1:-}" =~ ^[0-9]+$ ]] && { PORT="$1"; shift; } ;;
        --clean) CLEAN=1; shift ;;
        -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

echo ">> Installing docs dependencies (uv sync --extra docs)..."
uv sync --extra docs

BUILD_DIR="docs/_build/html"
if [[ "$CLEAN" -eq 1 ]]; then
    echo ">> Cleaning $BUILD_DIR..."
    rm -rf "$BUILD_DIR"
fi

if [[ "$LIVE" -eq 1 ]]; then
    echo ">> Starting live-reload server at http://localhost:$PORT (Ctrl-C to stop)..."
    exec uv run --with sphinx-autobuild sphinx-autobuild --port "$PORT" docs "$BUILD_DIR"
fi

echo ">> Building HTML docs..."
uv run sphinx-build -b html docs "$BUILD_DIR"
echo ">> Built: $REPO_ROOT/$BUILD_DIR/index.html"

if [[ "$SERVE" -eq 1 ]]; then
    echo ">> Serving at http://localhost:$PORT (Ctrl-C to stop)..."
    exec python -m http.server -d "$BUILD_DIR" "$PORT"
fi
