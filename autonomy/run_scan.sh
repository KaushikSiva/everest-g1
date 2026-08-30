#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_gemini_key

exec uv run --extra autonomy mjpython -m summit_sentinel \
  --mode viewer --seconds 180 --autonomy scan "$@"
