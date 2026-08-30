#!/usr/bin/env bash
set -euo pipefail

AUTONOMY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${AUTONOMY_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "autonomy/setup_mac.sh supports macOS only." >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv from https://docs.astral.sh/uv/ and rerun this script." >&2
  exit 2
fi

uv sync --extra autonomy --extra dev
mkdir -p runtime
echo "MAC MUJOCO SETUP COMPLETE"
echo "Next: ./autonomy/run_rescue.sh (or carry, scan, controller)."
