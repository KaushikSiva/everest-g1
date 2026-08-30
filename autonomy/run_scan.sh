#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
if ! has_cli_flag --offline-plan "$@"; then
  require_gemini_key
fi
if [[ "${spatial_audio_enabled}" == "1" ]]; then
  set -- --spatial-audio --acoustic-localization "$@"
fi

exec uv run --extra autonomy mjpython -m summit_sentinel \
  --mode viewer --seconds 180 --autonomy scan \
  "$@"
