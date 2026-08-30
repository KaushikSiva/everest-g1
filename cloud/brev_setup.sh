#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${repo_root}/cloud/pins.env"

workspace_root="/home/ubuntu/workspace/everest-g1-cloud"
arena_dir="${workspace_root}/IsaacLab-Arena"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "BREV SETUP FAILED: Isaac Lab-Arena requires Linux x86_64." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "BREV SETUP FAILED: nvidia-smi is unavailable; select an NVIDIA GPU instance." >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "BREV SETUP FAILED: install uv using Brev's user-level setup, then retry." >&2
  exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
mkdir -p "${workspace_root}" "${workspace_root}/datasets" "${workspace_root}/models"

if [[ ! -d "${arena_dir}/.git" ]]; then
  git clone --recurse-submodules https://github.com/isaac-sim/IsaacLab-Arena.git "${arena_dir}"
fi

git -C "${arena_dir}" fetch origin "${ISAACLAB_ARENA_SHA}"
git -C "${arena_dir}" checkout --detach "${ISAACLAB_ARENA_SHA}"
git -C "${arena_dir}" submodule update --init --recursive

resolved_isaaclab="$(git -C "${arena_dir}/submodules/IsaacLab" rev-parse HEAD)"
resolved_groot="$(git -C "${arena_dir}/submodules/Isaac-GR00T" rev-parse HEAD)"
if [[ "${resolved_isaaclab}" != "${ISAACLAB_SHA}" ]]; then
  echo "BREV SETUP FAILED: Arena resolved Isaac Lab ${resolved_isaaclab}, expected ${ISAACLAB_SHA}." >&2
  exit 1
fi
if [[ "${resolved_groot}" != "${ISAAC_GROOT_SHA}" ]]; then
  echo "BREV SETUP FAILED: Arena resolved Isaac-GR00T ${resolved_groot}, expected ${ISAAC_GROOT_SHA}." >&2
  exit 1
fi

(
  cd "${arena_dir}"
  uv sync --no-default-groups --group isaaclab-from-wheel --group gr00t-client
  uv pip install --python .venv/bin/python --no-deps --editable "${repo_root}"
  .venv/bin/python -c "import isaaclab, isaaclab_arena, everest_g1; print('Isaac/Arena/Everest imports OK')"
)

echo "BREV SETUP COMPLETE: ${arena_dir}"
echo "Next: ${repo_root}/cloud/run_brev_rescue.sh"
