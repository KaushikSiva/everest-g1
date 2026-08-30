"""Modal probes for headless Arena and bounded GR00T N1.7/SONIC fine-tuning.

This intentionally keeps simulation and SONIC training in separate images:
Arena currently pins Isaac Lab 3.0 beta, while the released SONIC stack pins
Isaac Lab 2.3.2. No credential values are embedded in this file.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path, PurePosixPath

import modal

ARENA_SHA = "435745729199fd03273ab5038d82480e02268ec6"
ISAAC_GROOT_SHA = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
GROOT_WBC_SHA = "a0732b642c0333077e127a2f56ab0014c196bca4"
REPO_ROOT = Path(__file__).resolve().parents[1]

app = modal.App("everest-g1")
datasets = modal.Volume.from_name("everest-g1-datasets", create_if_missing=True)
models = modal.Volume.from_name("everest-g1-models", create_if_missing=True)

arena_image = (
    modal.Image.from_registry("nvcr.io/nvidia/isaac-lab:3.0.0-beta2", add_python="3.12")
    .entrypoint([])
    .env({"ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y"})
    .apt_install("git", "git-lfs")
    .run_commands(
        "git clone https://github.com/isaac-sim/IsaacLab-Arena.git /opt/arena",
        f"git -C /opt/arena checkout --detach {ARENA_SHA}",
        "git -C /opt/arena submodule update --init --recursive",
        "/workspace/IsaacLab/isaaclab.sh -p -m pip install -e '/opt/arena[dev]'",
    )
    .add_local_dir(REPO_ROOT / "src", "/opt/everest-g1/src", copy=True)
    .env({"PYTHONPATH": "/opt/everest-g1/src:/opt/arena"})
)

sonic_image = (
    modal.Image.from_registry("nvcr.io/nvidia/isaac-lab:2.3.2", add_python="3.11")
    .entrypoint([])
    .env({"ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y"})
    .apt_install("git", "git-lfs")
    .run_commands(
        "git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git /opt/groot-wbc",
        f"git -C /opt/groot-wbc checkout --detach {GROOT_WBC_SHA}",
        "/workspace/IsaacLab/isaaclab.sh -p -m pip install -e "
        "'/opt/groot-wbc/gear_sonic[training]'",
    )
)

groot_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .entrypoint([])
    .apt_install("git", "git-lfs", "ffmpeg")
    .uv_pip_install("uv")
    .run_commands(
        "git clone https://github.com/NVIDIA/Isaac-GR00T.git /opt/Isaac-GR00T",
        f"git -C /opt/Isaac-GR00T checkout --detach {ISAAC_GROOT_SHA}",
        "cd /opt/Isaac-GR00T && uv sync --all-extras",
    )
)


def _run(command: list[str], timeout_s: int) -> dict[str, object]:
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout[-8_000:],
        "stderr": completed.stderr[-8_000:],
    }


@app.function(image=arena_image, gpu="L40S", timeout=20 * 60)
def probe_arena() -> dict[str, object]:
    gpu = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], 30)
    imports = _run(
        [
            "/workspace/IsaacLab/isaaclab.sh",
            "-p",
            "-c",
            "import isaaclab, isaaclab_arena, everest_g1; print('imports-ok')",
        ],
        180,
    )
    return {"gpu": gpu, "imports": imports, "ready": imports["returncode"] == 0}


@app.function(
    image=arena_image,
    gpu="L40S",
    timeout=45 * 60,
    volumes={"/datasets": datasets, "/models": models},
)
def evaluate_headless(steps: int = 500) -> dict[str, object]:
    if not 20 <= steps <= 10_000:
        raise ValueError("steps must be between 20 and 10000")
    return _run(
        [
            "/workspace/IsaacLab/isaaclab.sh",
            "-p",
            "-m",
            "everest_g1.isaac.run",
            "--headless",
            "--policy_type",
            "everest_approach",
            "--num_envs",
            "1",
            "--num_steps",
            str(steps),
            "everest_g1_rescue",
        ],
        40 * 60,
    )


@app.function(
    image=groot_image,
    gpu="H100",
    timeout=24 * 60 * 60,
    volumes={"/datasets": datasets, "/models": models},
    secrets=[modal.Secret.from_name("everest-huggingface", required_keys=["HF_TOKEN"])],
)
def finetune_groot_n17_sonic(
    dataset_name: str,
    output_name: str,
    max_steps: int = 2_000,
) -> dict[str, object]:
    dataset = _volume_child("/datasets", dataset_name)
    output = _volume_child("/models", output_name)
    if not 1 <= max_steps <= 20_000:
        raise ValueError("max_steps must be between 1 and 20000")
    return _run(
        [
            "/opt/Isaac-GR00T/.venv/bin/python",
            "/opt/Isaac-GR00T/gr00t/experiment/launch_finetune.py",
            "--base-model-path",
            "nvidia/GR00T-N1.7-3B",
            "--dataset-path",
            dataset,
            "--embodiment-tag",
            "UNITREE_G1_SONIC",
            "--modality-config-path",
            "/opt/Isaac-GR00T/gr00t/configs/data/embodiment_configs.py",
            "--num-gpus",
            "1",
            "--output-dir",
            output,
            "--max-steps",
            str(max_steps),
            "--global-batch-size",
            "32",
            "--dataloader-num-workers",
            "4",
        ],
        23 * 60 * 60,
    )


@app.function(
    image=sonic_image,
    gpu="H100",
    timeout=24 * 60 * 60,
    volumes={"/datasets": datasets, "/models": models},
    secrets=[modal.Secret.from_name("everest-huggingface", required_keys=["HF_TOKEN"])],
)
def train_sonic_controller(
    robot_motion_name: str,
    smpl_motion_name: str,
    output_name: str,
    iterations: int = 100,
) -> dict[str, object]:
    robot_motion = _volume_child("/datasets", robot_motion_name)
    smpl_motion = _volume_child("/datasets", smpl_motion_name)
    output = _volume_child("/models", output_name)
    if not 1 <= iterations <= 20_000:
        raise ValueError("iterations must be between 1 and 20000")
    return _run(
        [
            "/workspace/IsaacLab/isaaclab.sh",
            "-p",
            "/opt/groot-wbc/gear_sonic/train_agent_trl.py",
            "+exp=manager/universal_token/all_modes/sonic_release",
            "+checkpoint=/models/sonic_release/last.pt",
            "num_envs=4096",
            "headless=True",
            f"++algo.config.num_learning_iterations={iterations}",
            f"++manager_env.commands.motion.motion_lib_cfg.motion_file={robot_motion}",
            f"++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file={smpl_motion}",
            f"++hydra.run.dir={output}",
        ],
        23 * 60 * 60,
    )


def _volume_child(root: str, name: str) -> str:
    child = PurePosixPath(name)
    if child.is_absolute() or not name or ".." in child.parts:
        raise ValueError("volume paths must be non-empty relative paths without '..'")
    return str(PurePosixPath(root) / child)


@app.local_entrypoint()
def main(steps: int = 500) -> None:
    probe = probe_arena.remote()
    print(json.dumps(probe, indent=2, sort_keys=True))
    if not probe["ready"]:
        raise SystemExit(
            "Modal GPU/import probe failed; use the Brev lane for interactive debugging."
        )
    result = evaluate_headless.remote(steps)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["returncode"] != 0:
        raise SystemExit("Headless Arena evaluation failed.")
