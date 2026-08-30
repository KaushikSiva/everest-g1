#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${repo_root}/cloud/pins.env"

stack_root="${EVEREST_STACK_ROOT:-/home/ubuntu/workspace/everest-g1-cloud}"
groot_dir="${stack_root}/Isaac-GR00T"

for command in git git-lfs uv; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "GR00T SETUP FAILED: ${command} is required." >&2
    exit 1
  fi
done

mkdir -p "${stack_root}"
if [[ ! -d "${groot_dir}/.git" ]]; then
  git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git "${groot_dir}"
fi
git -C "${groot_dir}" fetch origin "${ISAAC_GROOT_SHA}"
git -C "${groot_dir}" checkout --detach "${ISAAC_GROOT_SHA}"
git -C "${groot_dir}" submodule update --init --recursive
git -C "${groot_dir}" lfs pull

(
  cd "${groot_dir}"
  uv sync --python 3.12
  uv run python -c "import gr00t; print('GR00T N1.7 import OK')"
)

echo "GR00T SETUP COMPLETE: ${groot_dir}"
