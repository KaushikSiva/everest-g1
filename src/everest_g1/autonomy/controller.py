"""Deterministic execution layer for Gemini-selected MuJoCo route IDs."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from everest_g1.autonomy.planning import AutonomyMode, GeminiRoutePlanner, RouteOption
from everest_g1.beacon import JsonlAuditLog
from everest_g1.mujoco import MujocoRescueController, yaw_from_wxyz
from everest_g1.rescue import ApproachLimits
from summit_sentinel.simulation import SummitSentinelEnv


class AutonomousMujocoController:
    """Execute a validated route while keeping all low-level control local."""

    def __init__(
        self,
        *,
        env: SummitSentinelEnv,
        mode: AutonomyMode,
        route: RouteOption,
        planner: GeminiRoutePlanner,
        arm_live_call: bool = False,
        simulation_id: str = "",
        audit_log: Path = Path("runtime/everest-g1-events.jsonl"),
        planning_frame_bytes: int = 0,
    ) -> None:
        if not route.hard_safe:
            raise ValueError("cannot execute a route outside the local safety envelope")
        self.mode = mode
        self.route = route
        self.planner = planner
        self.control_dt_s = env.config.simulation.timestep
        self.audit_log = JsonlAuditLog(audit_log)
        self.stage = "scan" if mode == "scan" else "approach-route"
        self.waypoint_index = 0
        self.scan_cycles = 0
        self.completed = False
        self.carry_proxy_attached = False
        self.planning_frame_bytes = int(planning_frame_bytes)
        if self.planning_frame_bytes < 0:
            raise ValueError("planning_frame_bytes must be non-negative")
        self.last_command = np.zeros(3, dtype=np.float32)
        self._scan_forward = True
        self._waypoint_limits = ApproachLimits(
            touch_distance_m=0.13,
            robot_radius_m=0.0,
            person_radius_m=0.0,
            max_forward_mps=0.18,
            max_lateral_mps=0.10,
            max_yaw_rps=0.25,
            position_gain=0.48,
            yaw_gain=0.70,
        )
        self.rescue = (
            MujocoRescueController(
                person_xy=env.rescue_target_xy,
                control_dt_s=self.control_dt_s,
                arm_live_call=arm_live_call,
                simulation_id=simulation_id,
                audit_log=audit_log,
            )
            if mode in {"rescue", "carry"}
            else None
        )
        self.audit_log.write(
            "autonomy_plan_selected",
            simulator="mujoco",
            autonomy_mode=mode,
            planner=planner.provider,
            model=planner.model,
            route_id=route.route_id,
            aggregate_risk=route.aggregate_risk,
            factors=[
                "slope_deg",
                "temperature_c",
                "wind_mps",
                "visibility_m",
                "snow_depth_m",
                "effective_friction",
                "distance_from_start_m",
            ],
            reason=planner.last_reason,
            observations=planner.last_observations,
            front_camera_bytes=self.planning_frame_bytes,
            selected_route=route.prompt_payload(),
        )

    @property
    def live_call_armed(self) -> bool:
        return self.rescue is not None and self.rescue.live_call_armed

    @property
    def call_submitted(self) -> bool:
        return self.rescue is not None and self.rescue.call_submitted

    def update(self, env: SummitSentinelEnv) -> np.ndarray:
        if env.emergency_stop_latched or self.completed:
            self.last_command.fill(0.0)
            return self.last_command.copy()
        root = env.data.joint("floating_base_joint").qpos.copy()

        if self.stage == "approach-route":
            command = self._follow_waypoints(root, self.route.approach_waypoints)
            if command is not None:
                return command
            self.stage = "approach-person"
            self.waypoint_index = 0

        if self.stage == "approach-person":
            assert self.rescue is not None
            command = self.rescue.update(
                root,
                image_supplier=env.front_camera_jpeg if self.rescue.live_call_armed else None,
            )
            self.last_command[:] = command
            if not self.rescue.latch.latched:
                return self.last_command.copy()
            if self.mode == "rescue":
                self.stage = "complete"
                self.completed = True
                self.audit_log.write(
                    "autonomy_mission_complete",
                    autonomy_mode=self.mode,
                    route_id=self.route.route_id,
                )
                self.last_command.fill(0.0)
                return self.last_command.copy()
            env.set_casualty_carrying(True)
            self.carry_proxy_attached = True
            self.stage = "carry"
            self.waypoint_index = 0
            self.audit_log.write(
                "simulation_carry_proxy_attached",
                route_id=self.route.route_id,
                physical_grasp=False,
            )
            self.last_command.fill(0.0)
            return self.last_command.copy()

        if self.stage == "carry":
            command = self._follow_waypoints(root, self.route.mission_waypoints)
            if command is not None:
                return command
            self.stage = "complete"
            self.completed = True
            self.audit_log.write(
                "autonomy_mission_complete",
                autonomy_mode=self.mode,
                route_id=self.route.route_id,
                carry_proxy_attached=True,
            )
            self.last_command.fill(0.0)
            return self.last_command.copy()

        if self.stage == "scan":
            points = (
                self.route.mission_waypoints
                if self._scan_forward
                else tuple(reversed(self.route.mission_waypoints))
            )
            command = self._follow_waypoints(root, points)
            if command is not None:
                return command
            self.scan_cycles += 1
            self._scan_forward = not self._scan_forward
            self.waypoint_index = 0
            self.audit_log.write(
                "autonomy_scan_cycle_complete",
                route_id=self.route.route_id,
                scan_cycles=self.scan_cycles,
            )
            self.last_command.fill(0.0)
            return self.last_command.copy()

        self.last_command.fill(0.0)
        return self.last_command.copy()

    def _follow_waypoints(
        self,
        root_qpos: np.ndarray,
        waypoints: tuple[tuple[float, float], ...],
    ) -> np.ndarray | None:
        while self.waypoint_index < len(waypoints):
            target = waypoints[self.waypoint_index]
            dx = target[0] - float(root_qpos[0])
            dy = target[1] - float(root_qpos[1])
            distance = math.hypot(dx, dy)
            if distance <= self._waypoint_limits.touch_distance_m:
                self.audit_log.write(
                    "autonomy_waypoint_reached",
                    autonomy_mode=self.mode,
                    route_id=self.route.route_id,
                    waypoint_index=self.waypoint_index,
                    x_m=target[0],
                    y_m=target[1],
                )
                self.waypoint_index += 1
                continue
            yaw = yaw_from_wxyz(root_qpos[3:7])
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            body_x = cos_yaw * dx + sin_yaw * dy
            body_y = -sin_yaw * dx + cos_yaw * dy
            target_yaw = math.atan2(dy, dx)
            yaw_error = math.atan2(
                math.sin(target_yaw - yaw),
                math.cos(target_yaw - yaw),
            )
            self.last_command[:] = np.asarray(
                [
                    np.clip(
                        self._waypoint_limits.position_gain * body_x,
                        -self._waypoint_limits.max_forward_mps,
                        self._waypoint_limits.max_forward_mps,
                    ),
                    np.clip(
                        self._waypoint_limits.position_gain * body_y,
                        -self._waypoint_limits.max_lateral_mps,
                        self._waypoint_limits.max_lateral_mps,
                    ),
                    np.clip(
                        self._waypoint_limits.yaw_gain * yaw_error,
                        -self._waypoint_limits.max_yaw_rps,
                        self._waypoint_limits.max_yaw_rps,
                    ),
                ],
                dtype=np.float32,
            )
            return self.last_command.copy()
        return None

    def close(self) -> None:
        self.last_command.fill(0.0)
        if self.rescue is not None:
            self.rescue.close()
