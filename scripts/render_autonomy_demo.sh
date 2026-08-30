#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source autonomy/_common.sh

if ! has_cli_flag --offline-plan "$@"; then
  require_gemini_key
fi

exec uv run --extra autonomy python -m everest_g1.demo_video "$@"
