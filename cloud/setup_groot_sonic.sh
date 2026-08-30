#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${repo_root}/cloud/pins.env"

workspace_root="/home/ubuntu/workspace/everest-g1-cloud"
wbc_dir="${workspace_root}/GR00T-WholeBodyControl"

if [[ ! -d "${wbc_dir}/.git" ]]; then
  git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git "${wbc_dir}"
fi
git -C "${wbc_dir}" fetch origin "${GROOT_WBC_SHA}"
git -C "${wbc_dir}" checkout --detach "${GROOT_WBC_SHA}"

echo "Pinned GR00T Whole-Body Control at ${GROOT_WBC_SHA}."
echo "SONIC is a separate Isaac Lab 2.3.x environment from Arena's Isaac Lab 3.0 beta environment."
echo "Follow docs/GROOT_SONIC.md; do not install gear_sonic into the Arena virtualenv."
