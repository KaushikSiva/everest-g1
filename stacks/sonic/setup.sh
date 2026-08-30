#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${repo_root}/cloud/pins.env"

mode="${1:---mujoco}"
if [[ "${mode}" != "--mujoco" && "${mode}" != "--checkout-only" ]]; then
  echo "usage: $0 [--mujoco|--checkout-only]" >&2
  exit 2
fi

stack_root="${EVEREST_STACK_ROOT:-/home/ubuntu/workspace/everest-g1-cloud}"
sonic_dir="${stack_root}/GR00T-WholeBodyControl"

for command in git git-lfs; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "SONIC SETUP FAILED: ${command} is required." >&2
    exit 1
  fi
done

mkdir -p "${stack_root}"
if [[ ! -d "${sonic_dir}/.git" ]]; then
  git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git "${sonic_dir}"
fi
git -C "${sonic_dir}" fetch origin "${GROOT_WBC_SHA}"
git -C "${sonic_dir}" checkout --detach "${GROOT_WBC_SHA}"
git -C "${sonic_dir}" lfs pull

if [[ "${mode}" == "--mujoco" ]]; then
  (
    cd "${sonic_dir}"
    bash install_scripts/install_mujoco_sim.sh
  )
fi

echo "SONIC SETUP COMPLETE: ${sonic_dir} (${mode})"
echo "Isaac training remains in its own 2.3.2 environment; see docs/ISAAC_LAB_GROOT_SONIC.md."
