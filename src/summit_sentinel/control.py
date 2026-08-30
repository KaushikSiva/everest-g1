"""Unitree-compatible observation, policy, and PD control primitives."""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from summit_sentinel.config import ControllerConfig

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def projected_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Match Unitree's deployment projection of gravity into the base frame."""

    qw, qx, qy, qz = quaternion_wxyz
    return np.asarray(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
    torque_limits: np.ndarray,
) -> np.ndarray:
    """Convert joint-position targets to bounded motor torques."""

    torque = (target_q - q) * kp - dq * kd
    return np.clip(torque, -torque_limits, torque_limits)


class Policy(Protocol):
    mode: str

    def reset(self) -> None: ...

    def infer(self, observation: np.ndarray) -> np.ndarray: ...


@dataclass
class HoldPolicy:
    """Safe fallback that requests the official default standing angles."""

    num_actions: int
    mode: str = "hold-braced"

    def reset(self) -> None:
        """The stateless safe-hold policy has no episode memory to clear."""

        return None

    def infer(self, observation: np.ndarray) -> np.ndarray:
        del observation
        return np.zeros(self.num_actions, dtype=np.float32)


class TorchScriptPolicy:
    """Thin adapter around the official bundled Unitree TorchScript actor."""

    mode = "torchscript"

    def __init__(self, path: Path, num_actions: int) -> None:
        import torch

        self._torch = torch
        self._model = torch.jit.load(str(path), map_location="cpu")
        self._model.eval()
        if not callable(getattr(self._model, "reset_memory", None)):
            raise RuntimeError("recurrent TorchScript policy does not export reset_memory()")
        self._num_actions = num_actions

    def reset(self) -> None:
        """Clear the exported LSTM hidden and cell state for a new episode."""

        with self._torch.inference_mode():
            self._model.reset_memory()

    def infer(self, observation: np.ndarray) -> np.ndarray:
        tensor = self._torch.from_numpy(observation.astype(np.float32, copy=False)).unsqueeze(0)
        with self._torch.inference_mode():
            output = self._model(tensor)
        action = output.detach().cpu().numpy().reshape(-1).astype(np.float32)
        if action.shape != (self._num_actions,) or not np.all(np.isfinite(action)):
            raise RuntimeError(f"policy returned unsafe action shape/value: {action.shape}")
        return action


def policy_sha256(path: Path) -> str:
    """Hash policy bytes without asking PyTorch to deserialize them."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as policy_file:
        for chunk in iter(lambda: policy_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy(
    path: Path,
    num_actions: int,
    enabled: bool = True,
    *,
    expected_sha256: str | None,
) -> tuple[Policy, str | None]:
    """Verify a trusted digest before TorchScript deserialization.

    TorchScript is executable serialized code. A missing or mismatched digest
    falls back without reaching ``torch.jit.load``.
    """

    if not enabled:
        return HoldPolicy(num_actions), "policy disabled by operator"
    if expected_sha256 is None:
        return HoldPolicy(num_actions), "TorchScript refused: no trusted policy SHA-256 supplied"
    normalized_digest = expected_sha256.lower()
    if not _SHA256_PATTERN.fullmatch(normalized_digest):
        return HoldPolicy(num_actions), "TorchScript refused: invalid trusted policy SHA-256"
    try:
        observed_digest = policy_sha256(path)
        if not hmac.compare_digest(observed_digest, normalized_digest):
            return (
                HoldPolicy(num_actions),
                "TorchScript refused: policy SHA-256 mismatch before deserialization",
            )
        return TorchScriptPolicy(path, num_actions), None
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        return HoldPolicy(num_actions), f"TorchScript unavailable: {error}"


@dataclass
class ObservationBuilder:
    """Build the exact 47-value vector used by Unitree's 12-DOF G1 policy.

    Order: angular velocity (3), projected gravity (3), scaled command (3),
    relative joint positions (12), scaled joint velocities (12), previous
    action (12), and gait phase sine/cosine (2).
    """

    config: ControllerConfig

    def build(
        self,
        *,
        angular_velocity: np.ndarray,
        quaternion_wxyz: np.ndarray,
        command: np.ndarray,
        joint_position: np.ndarray,
        joint_velocity: np.ndarray,
        previous_action: np.ndarray,
        sim_time: float,
    ) -> np.ndarray:
        phase = (sim_time % self.config.gait_period) / self.config.gait_period
        obs = np.concatenate(
            (
                angular_velocity * self.config.ang_vel_scale,
                projected_gravity(quaternion_wxyz),
                command * self.config.command_scale,
                (joint_position - self.config.default_angles) * self.config.dof_pos_scale,
                joint_velocity * self.config.dof_vel_scale,
                previous_action,
                np.asarray(
                    [np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)],
                    dtype=np.float32,
                ),
            )
        ).astype(np.float32, copy=False)
        if obs.shape != (self.config.num_observations,) or not np.all(np.isfinite(obs)):
            raise RuntimeError(f"observation contract violated: {obs.shape} or non-finite value")
        return obs
