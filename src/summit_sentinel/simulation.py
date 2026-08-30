"""MuJoCo environment and deterministic G1 control loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np

from summit_sentinel.config import POLICY_PATH, POLICY_SHA256, SCENE_PATH, AppConfig, load_config
from summit_sentinel.control import (
    HoldPolicy,
    ObservationBuilder,
    load_policy,
    pd_control,
    projected_gravity,
)
from summit_sentinel.terrain import TERRAIN_SEED, ensure_terrain, plateau_world_height

ROBOT_NOMINAL_BASE_HEIGHT = 0.793
StopAuthority = Literal["local", "supervisory", "fault"]


class SafetyFault(RuntimeError):
    """A numeric/control invariant failed before a physics update."""


@dataclass(frozen=True)
class StepResult:
    time: float
    command: np.ndarray
    observation: np.ndarray
    action: np.ndarray
    torque: np.ndarray
    fell: bool
    reset: bool
    emergency_stop_latched: bool
    physics_advanced: bool
    stop_authority: StopAuthority | None
    locomotion_inhibited: bool


class SummitSentinelEnv:
    """A single G1 in a compressed Everest-inspired MuJoCo scene."""

    def __init__(
        self,
        *,
        config: AppConfig | None = None,
        scene_path: Path = SCENE_PATH,
        policy_path: Path = POLICY_PATH,
        trusted_policy_sha256: str | None = None,
        seed: int = TERRAIN_SEED,
        use_policy: bool = True,
        auto_reset: bool = True,
    ) -> None:
        self.config = config or load_config()
        if seed != TERRAIN_SEED:
            raise ValueError(
                f"the checked Everest heightfield uses fixed seed {TERRAIN_SEED}; got {seed}"
            )
        self.seed = seed
        self.auto_reset = auto_reset
        ensure_terrain(seed=seed)

        # Deliberately use the supported path-based model loader: relative
        # include, mesh, and heightfield asset paths resolve from this scene.
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.config.simulation.timestep
        self._validate_model()
        terrain_friction = float(self.model.geom("everest_terrain").friction[0])
        self.control_mode: Literal["hold", "supervisory"] = "supervisory"
        self.scenario_conditions: dict[str, float] = {
            "friction": terrain_friction,
            "wind_mps": 0.0,
            "visibility_m": 10_000.0,
            "snow_depth_m": 0.0,
            "effective_friction": terrain_friction,
        }

        self._fallback_policy = HoldPolicy(self.config.controller.num_actions)
        expected_policy_sha256 = (
            POLICY_SHA256
            if policy_path.resolve() == POLICY_PATH.resolve()
            else trusted_policy_sha256
        )
        self.policy, self.policy_warning = load_policy(
            policy_path,
            self.config.controller.num_actions,
            enabled=use_policy,
            expected_sha256=expected_policy_sha256,
        )
        self.observation_builder = ObservationBuilder(self.config.controller)
        self.previous_action = np.zeros(self.config.controller.num_actions, dtype=np.float32)
        self.target_joint_position = self.config.controller.default_angles.copy()
        self.physics_steps = 0
        self.reset_count = 0
        self.emergency_stop_latched = False
        self.emergency_stop_reason: str | None = None
        self.stop_authority: StopAuthority | None = None
        self._reset_since_emergency_stop = False
        self._reset_authority: str | None = None
        self.locomotion_inhibited = False
        self.last_observation = np.zeros(self.config.controller.num_observations, dtype=np.float32)
        self.last_torque = np.zeros(self.config.controller.num_actions, dtype=np.float32)
        self.reset(authority="system")

    @property
    def policy_mode(self) -> str:
        return self.policy.mode

    @property
    def policy_hz(self) -> float:
        return 1.0 / self.config.simulation.policy_dt

    def _validate_model(self) -> None:
        expected = self.config.controller.num_actions
        if (self.model.nq, self.model.nv, self.model.nu) != (19, 18, expected):
            raise ValueError(
                "unexpected G1 dimensions: "
                f"nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}"
            )
        # Named access is intentional: it guards against silent changes in the
        # vendored XML joint order.
        self.model.joint("floating_base_joint")
        for index, joint_name in enumerate(self.config.controller.joint_names):
            joint = self.model.joint(joint_name)
            actuator = self.model.actuator(joint_name)
            if int(joint.id) < 0 or int(actuator.id) != index:
                raise ValueError(f"joint/actuator contract mismatch for {joint_name}")
        self.model.geom("everest_terrain")
        self.model.body("downed_person")
        self.model.site("downed_person_target")

    @property
    def rescue_target_xy(self) -> tuple[float, float]:
        position = self.data.site("downed_person_target").xpos
        return float(position[0]), float(position[1])

    def _joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(
            [self.data.joint(name).qpos[0] for name in self.config.controller.joint_names],
            dtype=np.float32,
        )
        dq = np.asarray(
            [self.data.joint(name).qvel[0] for name in self.config.controller.joint_names],
            dtype=np.float32,
        )
        return q, dq

    def _activate_fallback(self, warning: str) -> None:
        """Switch to the preconstructed hold policy without loading any file."""

        self.policy = self._fallback_policy
        self.policy.reset()
        self.policy_warning = f"{warning}; using preconstructed hold-braced fallback"

    def reset(
        self,
        *,
        authority: Literal["local", "remote", "system"] = "local",
        require_local_ack: bool = False,
    ) -> None:
        """Reset robot state without silently clearing a latched emergency stop."""

        was_stopped = self.emergency_stop_latched
        previous_inhibition = self.locomotion_inhibited
        if was_stopped and self.stop_authority in {"local", "fault"} and authority == "remote":
            raise RuntimeError("remote reset cannot acknowledge a local or fault stop")
        try:
            self.policy.reset()
        except RuntimeError as error:
            self._activate_fallback(f"policy memory reset failure: {error}")
        mujoco.mj_resetData(self.model, self.data)
        root = self.data.joint("floating_base_joint")
        root.qpos[:] = np.asarray(
            [0.0, 0.0, ROBOT_NOMINAL_BASE_HEIGHT + plateau_world_height(), 1.0, 0.0, 0.0, 0.0]
        )
        root.qvel[:] = 0.0
        for joint_name, angle in zip(
            self.config.controller.joint_names,
            self.config.controller.default_angles,
            strict=True,
        ):
            self.data.joint(joint_name).qpos[0] = angle
            self.data.joint(joint_name).qvel[0] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.previous_action.fill(0.0)
        self.target_joint_position[:] = self.config.controller.default_angles
        self.last_observation.fill(0.0)
        self.last_torque.fill(0.0)
        self.physics_steps = 0
        self.reset_count += 1
        self._reset_since_emergency_stop = was_stopped
        self._reset_authority = authority if was_stopped else None
        if require_local_ack:
            self.locomotion_inhibited = True
        elif authority == "local":
            self.locomotion_inhibited = False
        else:
            self.locomotion_inhibited = previous_inhibition

    def emergency_stop(
        self,
        reason: str = "operator",
        *,
        authority: StopAuthority = "local",
    ) -> bool:
        """Latch a local stop and zero controls before any further physics step.

        The simulation is frozen while latched: :meth:`step` does not run the
        policy, PD controller, MuJoCo, SQLite, MCP, or any network code. A reset
        followed by an explicit resume is required to continue.
        """

        if self.stop_authority in {"local", "fault"} and authority == "supervisory":
            return False
        newly_latched = not self.emergency_stop_latched or self.stop_authority != authority
        self.emergency_stop_latched = True
        self.emergency_stop_reason = reason[:160]
        self.stop_authority = authority
        self._reset_since_emergency_stop = False
        self._reset_authority = None
        self.data.ctrl.fill(0.0)
        self.last_torque.fill(0.0)
        return newly_latched

    def request_supervisory_stop(self, reason: str = "remote supervisory request") -> bool:
        """Latch a best-effort remote pause without impersonating local e-stop."""

        return self.emergency_stop(reason, authority="supervisory")

    def apply_scenario_conditions(self, conditions: dict[str, float]) -> None:
        """Apply validated environmental conditions without touching controls."""

        bounds = {
            "friction": (0.2, 1.5),
            "wind_mps": (0.0, 30.0),
            "visibility_m": (10.0, 10_000.0),
            "snow_depth_m": (0.0, 0.5),
        }
        if set(conditions) != set(bounds):
            raise ValueError("scenario conditions have unexpected fields")
        normalized: dict[str, float] = {}
        for name, (minimum, maximum) in bounds.items():
            value = float(conditions[name])
            if not np.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"scenario condition {name} is outside its safe bound")
            normalized[name] = value
        snow_fraction = normalized["snow_depth_m"] / bounds["snow_depth_m"][1]
        effective_friction = max(0.1, normalized["friction"] * (1.0 - 0.6 * snow_fraction))
        self.model.geom("everest_terrain").friction[0] = effective_friction
        normalized["effective_friction"] = effective_friction
        self.scenario_conditions = normalized

    def set_control_mode(self, mode: str) -> None:
        """Select hold/supervisory command handling without changing stop state."""

        if mode not in {"hold", "supervisory"}:
            raise ValueError("control mode must be hold or supervisory")
        self.control_mode = mode

    def resume(self, *, authority: Literal["local", "remote"] = "local") -> bool:
        """Clear a stop only after the operator or agent explicitly reset state."""

        if not self.emergency_stop_latched:
            return False
        if self.stop_authority in {"local", "fault"} and authority != "local":
            raise RuntimeError("remote resume cannot clear a local or fault stop")
        if not self._reset_since_emergency_stop:
            raise RuntimeError("emergency stop is latched; reset before resume")
        if self.stop_authority in {"local", "fault"} and self._reset_authority != "local":
            raise RuntimeError("local reset acknowledgement is required before resume")
        self.emergency_stop_latched = False
        self.emergency_stop_reason = None
        self.stop_authority = None
        self._reset_since_emergency_stop = False
        self._reset_authority = None
        return True

    @staticmethod
    def _require_finite(name: str, value: np.ndarray) -> None:
        if not np.all(np.isfinite(value)):
            raise SafetyFault(f"non-finite {name}")

    def _stopped_result(self) -> StepResult:
        self.data.ctrl.fill(0.0)
        self.last_torque.fill(0.0)
        return StepResult(
            time=float(self.data.time),
            command=np.zeros(3, dtype=np.float32),
            observation=self.last_observation.copy(),
            action=self.previous_action.copy(),
            torque=np.zeros(self.config.controller.num_actions, dtype=np.float32),
            fell=False,
            reset=False,
            emergency_stop_latched=True,
            physics_advanced=False,
            stop_authority=self.stop_authority,
            locomotion_inhibited=self.locomotion_inhibited,
        )

    def _fallen(self) -> bool:
        root_qpos = self.data.joint("floating_base_joint").qpos
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            return True
        relative_height = float(root_qpos[2] - plateau_world_height())
        gravity_z = float(projected_gravity(root_qpos[3:7])[2])
        return (
            relative_height < self.config.simulation.fall_base_height
            or gravity_z > self.config.simulation.fall_gravity_z
        )

    def _apply_fallback_stand_brace(self) -> None:
        """Keep the unlearned fallback upright with an explicit simulator brace.

        A position-only open-loop pose cannot actively balance a free humanoid.
        When TorchScript is absent, the safest useful visual fallback is a
        stationary, kinematically braced pelvis while leg joints retain bounded
        PD control. Joystick commands are ignored in this mode.
        """

        if self.policy.mode != "hold-braced":
            return
        root = self.data.joint("floating_base_joint")
        root.qpos[:] = np.asarray(
            [
                0.0,
                0.0,
                ROBOT_NOMINAL_BASE_HEIGHT + plateau_world_height(),
                1.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        root.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _policy_update(self, command: np.ndarray) -> None:
        q, dq = self._joint_state()
        root = self.data.joint("floating_base_joint")
        observation = self.observation_builder.build(
            angular_velocity=np.asarray(root.qvel[3:6], dtype=np.float32),
            quaternion_wxyz=np.asarray(root.qpos[3:7], dtype=np.float32),
            command=command,
            joint_position=q,
            joint_velocity=dq,
            previous_action=self.previous_action,
            sim_time=self.physics_steps * self.config.simulation.timestep,
        )
        self._require_finite("observation", observation)
        try:
            action = self.policy.infer(observation)
        except RuntimeError as error:
            # Inference failure degrades to a stationary default pose instead of
            # retaining a stale locomotion target.
            self._activate_fallback(f"runtime policy failure: {error}")
            action = self.policy.infer(observation)
        if action.shape != (self.config.controller.num_actions,):
            raise SafetyFault(f"unsafe action shape: {action.shape}")
        self._require_finite("action", action)
        self.previous_action[:] = action
        desired = (
            action * self.config.controller.action_scale + self.config.controller.default_angles
        )
        joint_ranges = np.asarray(
            [self.model.joint(name).range for name in self.config.controller.joint_names]
        )
        self._require_finite("desired joint target", desired)
        self.target_joint_position[:] = np.clip(desired, joint_ranges[:, 0], joint_ranges[:, 1])
        self._require_finite("clipped joint target", self.target_joint_position)
        self.last_observation[:] = observation

    def step(self, command: np.ndarray | None = None) -> StepResult:
        # Keep this as the literal first branch. A latched stop accepts even a
        # malformed command without conversion, policy work, torque, or physics.
        if self.emergency_stop_latched:
            return self._stopped_result()

        velocity_command = np.zeros(3, dtype=np.float32)
        if command is not None:
            candidate = np.asarray(command, dtype=np.float32)
            if candidate.shape != (3,) or not np.all(np.isfinite(candidate)):
                raise ValueError("velocity command must contain three finite values")
            velocity_command[:] = candidate
        velocity_command[:] = np.clip(velocity_command, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))

        if self.locomotion_inhibited or self.control_mode == "hold":
            velocity_command.fill(0.0)

        try:
            self._require_finite("qpos", self.data.qpos)
            self._require_finite("qvel", self.data.qvel)

            if self.physics_steps % self.config.simulation.control_decimation == 0:
                self._policy_update(velocity_command)

            q, dq = self._joint_state()
            self._require_finite("joint position", q)
            self._require_finite("joint velocity", dq)
            self._require_finite("joint target", self.target_joint_position)
            torque = pd_control(
                self.target_joint_position,
                q,
                self.config.controller.kps,
                dq,
                self.config.controller.kds,
                self.config.controller.torque_limits,
            )
            self._require_finite("torque", torque)
        except RuntimeError as error:
            self.emergency_stop(str(error), authority="fault")
            return self._stopped_result()
        self.data.ctrl[:] = torque
        mujoco.mj_step(self.model, self.data)
        self.physics_steps += 1
        self.last_torque[:] = torque
        self._apply_fallback_stand_brace()

        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            self.emergency_stop("non-finite state after physics step", authority="fault")

        fell = self._fallen()
        did_reset = False
        result_time = float(self.data.time)
        result_observation = self.last_observation.copy()
        result_action = self.previous_action.copy()
        if fell and self.auto_reset and not self.emergency_stop_latched:
            self.reset(authority="system", require_local_ack=True)
            did_reset = True

        return StepResult(
            time=result_time,
            command=velocity_command.copy(),
            observation=result_observation,
            action=result_action,
            torque=torque.copy(),
            fell=fell,
            reset=did_reset,
            emergency_stop_latched=self.emergency_stop_latched,
            physics_advanced=True,
            stop_authority=self.stop_authority,
            locomotion_inhibited=self.locomotion_inhibited,
        )
