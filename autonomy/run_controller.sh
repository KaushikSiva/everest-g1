#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

CALIBRATION="runtime/dualsense.json"
if [[ ! -f "${CALIBRATION}" ]]; then
  echo "Missing ${CALIBRATION}. Calibrate first:" >&2
  echo "uv run summit-sentinel --calibrate-joystick ${CALIBRATION} --joystick-index 0" >&2
  exit 2
fi

if [[ "${spatial_audio_enabled}" == "1" ]]; then
  set -- --spatial-audio --acoustic-localization "$@"
fi
if [[ "${live_call_enabled}" == "1" ]]; then
  set -- --arm-live-call "$@"
fi

exec uv run mjpython -m summit_sentinel \
  --mode viewer --seconds 600 --joystick --joystick-index 0 \
  --joystick-calibration "${CALIBRATION}" \
  --bridge-db runtime/summit.db --telemetry-hz 15 \
  "$@"
