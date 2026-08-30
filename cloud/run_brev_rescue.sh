#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
arena_dir="/home/ubuntu/workspace/everest-g1-cloud/IsaacLab-Arena"

if [[ ! -x "${arena_dir}/.venv/bin/python" ]]; then
  echo "RUN FAILED: run cloud/brev_setup.sh first." >&2
  exit 1
fi

arm_args=()
if [[ "${1:-}" == "--arm-live-call" ]]; then
  if [[ -z "${BEACON_API_TOKEN:-}" || -z "${BEACON_API_URL:-}" ]]; then
    echo "RUN FAILED: BEACON_API_URL and BEACON_API_TOKEN must already be exported." >&2
    exit 1
  fi
  read -r -p "Type ARM-LIVE-CALL to permit one real outbound call: " confirmation
  if [[ "${confirmation}" != "ARM-LIVE-CALL" ]]; then
    echo "RUN CANCELLED: live call was not armed." >&2
    exit 1
  fi
  export EVEREST_ARM_LIVE_CALL="ARM-LIVE-CALL"
  arm_args+=(--arm_live_call)
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--arm-live-call]" >&2
  exit 2
fi

export PYTHONPATH="${repo_root}/src:${arena_dir}:${PYTHONPATH:-}"
export ACCEPT_EULA=Y

cd "${arena_dir}"
exec .venv/bin/python -m everest_g1.isaac.run \
  --viz kit \
  --enable_cameras \
  --policy_type everest_approach \
  --num_envs 1 \
  --num_steps 3000 \
  "${arm_args[@]}" \
  everest_g1_rescue
