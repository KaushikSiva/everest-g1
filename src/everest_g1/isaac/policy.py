"""Bounded Isaac Lab-Arena policy for the first rescue commissioning gate."""

from __future__ import annotations

import atexit
import math
from dataclasses import dataclass, replace
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
from everest_g1.models import RescueObservation
from everest_g1.rescue import (
    ApproachLimits,
    ProximityLatch,
    approach_person,
    body_bearing,
    target_from_bearing,
)
from everest_g1.spatial_audio import RescueAudio, SpatialAudioSettings

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
    acoustic_localization: bool = False
    spatial_audio: bool = False
    spatial_audio_out: str = "runtime/everest-g1-rescue.wav"


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
        self._latch_cued = False
        audio_settings = SpatialAudioSettings(
            acoustic_localization=config.acoustic_localization,
            render_cue=config.spatial_audio,
            output_path=Path(config.spatial_audio_out),
        )
        self.audio = RescueAudio(audio_settings) if audio_settings.enabled else None
        if config.arm_live_call:
            settings = BeaconSettings.from_env(arm_requested=True)
            settings.validate()
            self.call_worker = BeaconCallWorker(settings, self.audit_log)
        if self.call_worker is not None or self.audio is not None:
            atexit.register(self.close)
        self.audit_log.write(
            "simulation_started",
            simulation_id=self.simulation_id,
            live_call_armed=self.call_worker is not None,
        )
        if self.audio is not None:
            self.audit_log.write(
                "spatial_audio_started",
                simulation_id=self.simulation_id,
                acoustic_localization=self.audio.sensor is not None,
                cue_rendered=self.audio.renderer is not None,
            )

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        del observation
        if env.action_space.shape[0] != 1:
            raise RuntimeError("everest_approach currently requires --num_envs 1")

        robot = env.unwrapped.scene["robot"]
        root_xy = robot.data.root_pos_w[0, :2]
        root_quat = robot.data.root_quat_w[0]  # Isaac Lab uses w, x, y, z.
        yaw = _yaw_from_wxyz(root_quat)
        robot_xy = (float(root_xy[0].item()), float(root_xy[1].item()))
        person_xy = (self.config.person_x_m, self.config.person_y_m)

        # Audio may steer the approach. The range gate that stops the robot and
        # releases a BeaconCall always uses the observed geometry.
        gate = approach_person(
            robot_xy=robot_xy,
            robot_yaw_rad=yaw,
            person_xy=person_xy,
            limits=self.limits,
            reached=self.latch.latched,
        )
        command = gate
        target_xy = person_xy
        if self.audio is not None and not gate.reached:
            estimate = self.audio.observe(robot_xy=robot_xy, robot_yaw_rad=yaw, source_xy=person_xy)
            if estimate is not None:
                target_xy = target_from_bearing(
                    robot_xy=robot_xy,
                    robot_yaw_rad=yaw,
                    bearing_rad=estimate.bearing_rad,
                    surface_distance_m=gate.surface_distance_m,
                    limits=self.limits,
                )
                steer = approach_person(
                    robot_xy=robot_xy,
                    robot_yaw_rad=yaw,
                    person_xy=target_xy,
                    limits=self.limits,
                )
                command = replace(
                    steer,
                    surface_distance_m=gate.surface_distance_m,
                    reached=gate.reached,
                )
        reached = self.latch.update(command.surface_distance_m, self.config.control_dt_s)
        if self.audio is not None:
            self.audio.cue(
                dt_s=self.config.control_dt_s,
                bearing_rad=body_bearing(robot_xy=robot_xy, robot_yaw_rad=yaw, target_xy=target_xy),
                distance_m=command.surface_distance_m,
            )

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
            if not self._latch_cued:
                self._latch_cued = True
                if self.audio is not None:
                    self.audio.mark_proximity_latched()
            observation_facts = RescueObservation(
                simulation_id=self.simulation_id,
                distance_m=command.surface_distance_m,
            )
            if self.call_worker is not None:
                if self.call_worker.submit_once(observation_facts) and self.audio is not None:
                    self.audio.mark_call_submitted()
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
        if self.audio is not None:
            self.audio.reset()

    def close(self) -> None:
        if self.call_worker is not None:
            self.call_worker.close(timeout_s=2.0)
        if self.audio is not None:
            summary = self.audio.close(self.simulation_id)
            self.audio = None
            if summary is not None:
                self.audit_log.write(
                    "spatial_audio_written",
                    simulation_id=self.simulation_id,
                    **summary,
                )


def _yaw_from_wxyz(quaternion: torch.Tensor) -> float:
    w, x, y, z = (float(value.item()) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
