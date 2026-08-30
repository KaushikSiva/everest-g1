"""Bounded Isaac Lab-Arena policy for the first rescue commissioning gate."""

from __future__ import annotations

import atexit
import math
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import torch
from gymnasium.spaces.dict import Dict as GymSpacesDict
from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

from everest_g1.beacon import (
    BeaconCallWorker,
    BeaconSettings,
    JsonlAuditLog,
    new_simulation_id,
)
from everest_g1.isaac.camera import isaac_front_camera_jpeg
from everest_g1.models import RescueObservation
from everest_g1.rescue import ApproachLimits, ProximityLatch, approach_person

_WBC_COMMAND_DIM = 7  # navigation xyz, base height, torso rpy


@dataclass
class EverestApproachPolicyCfg(PolicyCfg):
    """Configuration exposed through Arena's generated policy CLI."""

    person_x_m: float = 2.6
    person_y_m: float = 0.75
    touch_distance_m: float = 0.15
    dwell_s: float = 0.25
    max_forward_mps: float = 0.20
    max_lateral_mps: float = 0.12
    max_yaw_rps: float = 0.30
    control_dt_s: float = 0.02
    base_height_m: float = 0.75
    simulation_id: str = ""
    audit_log: str = "runtime/everest-g1-events.jsonl"
    arm_live_call: bool = False


@register_policy
class EverestApproachPolicy(PolicyBase[EverestApproachPolicyCfg]):
    """Approach the proxy, stop, dwell, and enqueue at most one BeaconCall."""

    name = "everest_approach"

    def __init__(self, config: EverestApproachPolicyCfg):
        super().__init__(config)
        self.simulation_id = config.simulation_id or new_simulation_id()
        self.limits = ApproachLimits(
            touch_distance_m=config.touch_distance_m,
            max_forward_mps=config.max_forward_mps,
            max_lateral_mps=config.max_lateral_mps,
            max_yaw_rps=config.max_yaw_rps,
        )
        self.latch = ProximityLatch(config.touch_distance_m, config.dwell_s)
        self.audit_log = JsonlAuditLog(Path(config.audit_log))
        self.call_worker: BeaconCallWorker | None = None
        self._disarmed_event_written = False
        self.front_camera_status = "not_requested"
        self.front_camera_bytes = 0
        if config.arm_live_call:
            settings = BeaconSettings.from_env(arm_requested=True)
            settings.validate()
            self.call_worker = BeaconCallWorker(settings, self.audit_log)
            atexit.register(self.close)
        self.audit_log.write(
            "simulation_started",
            simulator="isaac_lab",
            simulation_id=self.simulation_id,
            live_call_armed=self.call_worker is not None,
        )

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        if env.action_space.shape[0] != 1:
            raise RuntimeError("everest_approach currently requires --num_envs 1")

        robot = env.unwrapped.scene["robot"]
        root_xy = robot.data.root_pos_w[0, :2]
        root_quat = robot.data.root_quat_w[0]  # Isaac Lab uses w, x, y, z.
        yaw = _yaw_from_wxyz(root_quat)
        command = approach_person(
            robot_xy=(float(root_xy[0].item()), float(root_xy[1].item())),
            robot_yaw_rad=yaw,
            person_xy=(self.config.person_x_m, self.config.person_y_m),
            limits=self.limits,
            reached=self.latch.latched,
        )
        reached = self.latch.update(command.surface_distance_m, self.config.control_dt_s)

        action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        joint_count = action.shape[-1] - _WBC_COMMAND_DIM
        joint_positions = robot.data.joint_pos
        if joint_positions.shape[-1] < joint_count:
            raise RuntimeError(
                f"G1 action expects {joint_count} joints but state has {joint_positions.shape[-1]}"
            )
        action[:, :joint_count] = joint_positions[:, :joint_count]
        action[:, -7:-4] = torch.tensor(
            [[command.forward_mps, command.lateral_mps, command.yaw_rps]],
            device=action.device,
        )
        action[:, -4] = self.config.base_height_m
        # The final three torso RPY commands remain zero.

        if reached:
            action[:, -7:-4] = 0.0
            image_jpeg = None
            if self.call_worker is not None and not self.call_worker.submitted:
                try:
                    image_jpeg = isaac_front_camera_jpeg(observation)
                    self.front_camera_status = "captured"
                    self.front_camera_bytes = len(image_jpeg)
                    self.audit_log.write(
                        "front_camera_captured",
                        simulator="isaac_lab",
                        simulation_id=self.simulation_id,
                        jpeg_bytes=self.front_camera_bytes,
                    )
                except Exception as error:
                    self.front_camera_status = "capture_failed"
                    self.audit_log.write(
                        "front_camera_capture_failed",
                        simulator="isaac_lab",
                        simulation_id=self.simulation_id,
                        error_type=type(error).__name__,
                    )
            observation_facts = RescueObservation(
                simulation_id=self.simulation_id,
                distance_m=command.surface_distance_m,
                image_jpeg=image_jpeg,
            )
            if self.call_worker is not None:
                self.call_worker.submit_once(observation_facts)
            elif not self._disarmed_event_written:
                self.audit_log.write(
                    "proximity_reached_call_disarmed",
                    simulation_id=self.simulation_id,
                    distance_m=round(command.surface_distance_m, 3),
                )
                self._disarmed_event_written = True
        return action

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids
        # Reset proximity across episodes, but never re-arm the one-call-per-process latch.
        self.latch.reset()

    def close(self) -> None:
        if self.call_worker is not None:
            self.call_worker.close(timeout_s=50.0)


def _yaw_from_wxyz(quaternion: torch.Tensor) -> float:
    w, x, y, z = (float(value.item()) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
