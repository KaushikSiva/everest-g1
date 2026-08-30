"""MuJoCo adapter for the shared Everest rescue and BeaconCall flow."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from everest_g1.beacon import (
    BeaconCallWorker,
    BeaconSettings,
    JsonlAuditLog,
    new_simulation_id,
)
from everest_g1.models import NavigationCommand, RescueObservation
from everest_g1.rescue import ApproachLimits, ProximityLatch, approach_person


class MujocoRescueController:
    """Approach, stop, dwell, and enqueue one BeaconCall from MuJoCo."""

    def __init__(
        self,
        *,
        person_xy: tuple[float, float],
        control_dt_s: float,
        arm_live_call: bool = False,
        simulation_id: str = "",
        audit_log: Path = Path("runtime/everest-g1-events.jsonl"),
        limits: ApproachLimits | None = None,
        dwell_s: float = 0.25,
    ) -> None:
        if control_dt_s <= 0:
            raise ValueError("control_dt_s must be positive")
        self.person_xy = person_xy
        self.control_dt_s = control_dt_s
        self.simulation_id = simulation_id or new_simulation_id()
        self.limits = limits or ApproachLimits()
        self.latch = ProximityLatch(self.limits.touch_distance_m, dwell_s)
        self.audit_log = JsonlAuditLog(audit_log)
        self.call_worker: BeaconCallWorker | None = None
        self._disarmed_event_written = False
        self.last_command = NavigationCommand(0.0, 0.0, 0.0, float("inf"), False)

        if arm_live_call:
            settings = BeaconSettings.from_env(arm_requested=True)
            settings.validate()
            self.call_worker = BeaconCallWorker(settings, self.audit_log)
        self.audit_log.write(
            "simulation_started",
            simulator="mujoco",
            simulation_id=self.simulation_id,
            live_call_armed=self.call_worker is not None,
        )

    @property
    def live_call_armed(self) -> bool:
        return self.call_worker is not None

    @property
    def call_submitted(self) -> bool:
        return self.call_worker is not None and self.call_worker.submitted

    def update(self, robot_qpos: np.ndarray) -> np.ndarray:
        """Return the next body-frame [forward, lateral, yaw] command."""

        qpos = np.asarray(robot_qpos, dtype=np.float64)
        if qpos.shape != (7,) or not np.all(np.isfinite(qpos)):
            raise ValueError("MuJoCo rescue root pose must contain seven finite values")
        command = approach_person(
            robot_xy=(float(qpos[0]), float(qpos[1])),
            robot_yaw_rad=yaw_from_wxyz(qpos[3:7]),
            person_xy=self.person_xy,
            limits=self.limits,
            reached=self.latch.latched,
        )
        reached = self.latch.update(command.surface_distance_m, self.control_dt_s)
        self.last_command = command
        if reached:
            facts = RescueObservation(self.simulation_id, command.surface_distance_m)
            if self.call_worker is not None:
                self.call_worker.submit_once(facts)
            elif not self._disarmed_event_written:
                self.audit_log.write(
                    "proximity_reached_call_disarmed",
                    simulator="mujoco",
                    simulation_id=self.simulation_id,
                    distance_m=round(command.surface_distance_m, 3),
                )
                self._disarmed_event_written = True
            return np.zeros(3, dtype=np.float32)
        return np.asarray(
            [command.forward_mps, command.lateral_mps, command.yaw_rps],
            dtype=np.float32,
        )

    def close(self) -> None:
        if self.call_worker is not None:
            self.call_worker.close(timeout_s=2.0)


def yaw_from_wxyz(quaternion: np.ndarray) -> float:
    """Return planar yaw from a MuJoCo w, x, y, z quaternion."""

    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
