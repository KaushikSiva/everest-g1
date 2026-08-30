#!/usr/bin/env bash
set -euo pipefail

AUTONOMY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${AUTONOMY_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This launcher is the macOS MuJoCo lane. See docs for Linux/Isaac." >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ then rerun setup_mac.sh." >&2
  exit 2
fi

require_gemini_key() {
  if [[ -z "${GEMINI_API_KEY:-}" ]]; then
    read -r -s -p "Gemini API key (used for this process only): " GEMINI_API_KEY
    echo
    export GEMINI_API_KEY
  fi
  if [[ -z "${GEMINI_API_KEY}" ]]; then
    echo "GEMINI_API_KEY cannot be empty." >&2
    exit 2
  fi
}

live_call_args=()
if [[ "${EVEREST_ARM_LIVE_CALL:-}" == "ARM-LIVE-CALL" ]]; then
  live_call_args+=(--arm-live-call)
fi

audio_args=(--spatial-audio --acoustic-localization)
if [[ "${EVEREST_DISABLE_SPATIAL_AUDIO:-}" == "1" ]]; then
  audio_args=()
fi
