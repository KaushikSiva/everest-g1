"""Validated configuration for the official Unitree G1 deployment contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config" / "g1.yaml"
SCENE_PATH = PACKAGE_ROOT / "assets" / "unitree_g1" / "summit_scene.xml"
POLICY_PATH = PACKAGE_ROOT / "assets" / "policies" / "g1_motion.pt"
POLICY_SHA256 = "cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d"


def _vector(section: dict[str, Any], key: str, length: int) -> np.ndarray:
    value = np.asarray(section[key], dtype=np.float32)
    if value.shape != (length,):
        raise ValueError(f"{key} must contain exactly {length} values; got {value.shape}")
    return value


@dataclass(frozen=True)
class ControllerConfig:
    joint_names: tuple[str, ...]
    kps: np.ndarray
    kds: np.ndarray
    default_angles: np.ndarray
    torque_limits: np.ndarray
    ang_vel_scale: float
    dof_pos_scale: float
    dof_vel_scale: float
    action_scale: float
    command_scale: np.ndarray
    num_actions: int
    num_observations: int
    gait_period: float


@dataclass(frozen=True)
class SimulationConfig:
    timestep: float
    control_decimation: int
    fall_base_height: float
    fall_gravity_z: float

    @property
    def policy_dt(self) -> float:
        return self.timestep * self.control_decimation


@dataclass(frozen=True)
class JoystickConfig:
    deadzone: float
    axis_lateral: int
    axis_forward: int
    axis_yaw: int
    max_forward: float
    max_lateral: float
    max_yaw: float
    reset_button: int
    emergency_stop_button: int
    resume_button: int
    quit_button: int


@dataclass(frozen=True)
class AppConfig:
    simulation: SimulationConfig
    controller: ControllerConfig
    joystick: JoystickConfig


def load_config(path: Path | None = None) -> AppConfig:
    """Load and validate the simulation contract from YAML."""

    config_path = path or DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    sim = raw["simulation"]
    ctrl = raw["controller"]
    joystick = raw["joystick"]
    num_actions = int(ctrl["num_actions"])
    num_observations = int(ctrl["num_observations"])
    joint_names = tuple(ctrl["joint_names"])
    if num_actions != 12 or len(joint_names) != num_actions:
        raise ValueError("the vendored G1 policy requires exactly 12 ordered leg actions")
    if num_observations != 47:
        raise ValueError("the vendored G1 policy requires exactly 47 observations")

    simulation = SimulationConfig(
        timestep=float(sim["timestep"]),
        control_decimation=int(sim["control_decimation"]),
        fall_base_height=float(sim["fall_base_height"]),
        fall_gravity_z=float(sim["fall_gravity_z"]),
    )
    if not np.isclose(simulation.timestep, 0.002) or not np.isclose(simulation.policy_dt, 0.02):
        raise ValueError("the policy contract requires 0.002 s physics and 0.02 s policy steps")

    controller = ControllerConfig(
        joint_names=joint_names,
        kps=_vector(ctrl, "kps", num_actions),
        kds=_vector(ctrl, "kds", num_actions),
        default_angles=_vector(ctrl, "default_angles", num_actions),
        torque_limits=_vector(ctrl, "torque_limits", num_actions),
        ang_vel_scale=float(ctrl["ang_vel_scale"]),
        dof_pos_scale=float(ctrl["dof_pos_scale"]),
        dof_vel_scale=float(ctrl["dof_vel_scale"]),
        action_scale=float(ctrl["action_scale"]),
        command_scale=_vector(ctrl, "command_scale", 3),
        num_actions=num_actions,
        num_observations=num_observations,
        gait_period=float(ctrl["gait_period"]),
    )
    joystick_config = JoystickConfig(**joystick)
    axis_indices = (
        joystick_config.axis_lateral,
        joystick_config.axis_forward,
        joystick_config.axis_yaw,
    )
    button_indices = (
        joystick_config.reset_button,
        joystick_config.emergency_stop_button,
        joystick_config.resume_button,
        joystick_config.quit_button,
    )
    if not 0.0 <= joystick_config.deadzone < 1.0:
        raise ValueError("joystick deadzone must be in [0, 1)")
    if min(*axis_indices, *button_indices) < 0:
        raise ValueError("joystick axis and button indices must be non-negative")
    if (
        min(
            joystick_config.max_forward,
            joystick_config.max_lateral,
            joystick_config.max_yaw,
        )
        <= 0.0
    ):
        raise ValueError("joystick command limits must be positive")
    return AppConfig(simulation=simulation, controller=controller, joystick=joystick_config)
