"""MuJoCo adapter for the shared Everest rescue and BeaconCall flow."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from everest_g1.beacon import (
    BeaconCallWorker,
    BeaconSettings,
    JsonlAuditLog,
    new_simulation_id,
)
from everest_g1.models import NavigationCommand, RescueObservation
from everest_g1.rescue import (
    ApproachLimits,
    ProximityLatch,
    approach_person,
    body_bearing,
    target_from_bearing,
)
from everest_g1.spatial_audio import RescueAudio, SpatialAudioSettings


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
        spatial_audio: SpatialAudioSettings | None = None,
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
        self.front_camera_status = "not_requested"
        self.front_camera_bytes = 0
        self._latch_cued = False
        self._audio_sensor_elapsed_s = float("inf")
        self.last_command = NavigationCommand(0.0, 0.0, 0.0, float("inf"), False)
        audio_settings = spatial_audio or SpatialAudioSettings()
        self.audio = RescueAudio(audio_settings) if audio_settings.enabled else None

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
        if self.audio is not None:
            self.audit_log.write(
                "spatial_audio_started",
                simulator="mujoco",
                simulation_id=self.simulation_id,
                acoustic_localization=self.audio.sensor is not None,
                cue_rendered=self.audio.renderer is not None,
            )

    @property
    def live_call_armed(self) -> bool:
        return self.call_worker is not None

    @property
    def call_submitted(self) -> bool:
        return self.call_worker is not None and self.call_worker.submitted

    def update(
        self,
        robot_qpos: np.ndarray,
        *,
        image_supplier: Callable[[], bytes] | None = None,
    ) -> np.ndarray:
        """Return the next body-frame [forward, lateral, yaw] command."""

        qpos = np.asarray(robot_qpos, dtype=np.float64)
        if qpos.shape != (7,) or not np.all(np.isfinite(qpos)):
            raise ValueError("MuJoCo rescue root pose must contain seven finite values")
        robot_xy = (float(qpos[0]), float(qpos[1]))
        robot_yaw = yaw_from_wxyz(qpos[3:7])

        # The onboard range gate always uses the observed geometry. Audio may
        # steer, but it may never decide that the robot is close enough to stop
        # or to hand the incident to BeaconCall.
        gate = approach_person(
            robot_xy=robot_xy,
            robot_yaw_rad=robot_yaw,
            person_xy=self.person_xy,
            limits=self.limits,
            reached=self.latch.latched,
        )
        drive = gate
        target_xy = self.person_xy
        if self.audio is not None and not gate.reached:
            self._audio_sensor_elapsed_s += self.control_dt_s
            estimate = self.audio.last_estimate
            if self.audio.sensor is not None and self._audio_sensor_elapsed_s >= 0.02:
                estimate = self.audio.observe(
                    robot_xy=robot_xy,
                    robot_yaw_rad=robot_yaw,
                    source_xy=self.person_xy,
                )
                self._audio_sensor_elapsed_s = 0.0
            if estimate is not None:
                target_xy = target_from_bearing(
                    robot_xy=robot_xy,
                    robot_yaw_rad=robot_yaw,
                    bearing_rad=estimate.bearing_rad,
                    surface_distance_m=gate.surface_distance_m,
                    limits=self.limits,
                )
                drive = approach_person(
                    robot_xy=robot_xy,
                    robot_yaw_rad=robot_yaw,
                    person_xy=target_xy,
                    limits=self.limits,
                )
        command = NavigationCommand(
            forward_mps=drive.forward_mps,
            lateral_mps=drive.lateral_mps,
            yaw_rps=drive.yaw_rps,
            surface_distance_m=gate.surface_distance_m,
            reached=gate.reached,
        )
        reached = self.latch.update(command.surface_distance_m, self.control_dt_s)
        self.last_command = command

        if self.audio is not None:
            self.audio.cue(
                dt_s=self.control_dt_s,
                bearing_rad=body_bearing(
                    robot_xy=robot_xy, robot_yaw_rad=robot_yaw, target_xy=target_xy
                ),
                distance_m=command.surface_distance_m,
            )
        if reached:
            if not self._latch_cued:
                self._latch_cued = True
                if self.audio is not None:
                    self.audio.mark_proximity_latched()
            image_jpeg = None
            if self.call_worker is not None:
                if not self.call_worker.submitted and image_supplier is not None:
                    try:
                        image_jpeg = image_supplier()
                        self.front_camera_status = "captured"
                        self.front_camera_bytes = len(image_jpeg)
                        self.audit_log.write(
                            "front_camera_captured",
                            simulator="mujoco",
                            simulation_id=self.simulation_id,
                            jpeg_bytes=self.front_camera_bytes,
                        )
                    except Exception as error:
                        self.front_camera_status = "capture_failed"
                        self.audit_log.write(
                            "front_camera_capture_failed",
                            simulator="mujoco",
                            simulation_id=self.simulation_id,
                            error_type=type(error).__name__,
                        )
                facts = RescueObservation(
                    self.simulation_id,
                    command.surface_distance_m,
                    image_jpeg=image_jpeg,
                )
                submitted = self.call_worker.submit_once(facts)
                if submitted and self.audio is not None:
                    self.audio.mark_call_submitted()
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

    @property
    def spatial_audio_path(self) -> Path | None:
        if self.audio is None or self.audio.renderer is None:
            return None
        return self.audio.settings.path_for(self.simulation_id)

    def close(self) -> None:
        if self.call_worker is not None:
            self.call_worker.close(timeout_s=50.0)
        if self.audio is not None:
            summary = self.audio.close(self.simulation_id)
            if summary is not None:
                self.audit_log.write(
                    "spatial_audio_written",
                    simulator="mujoco",
                    simulation_id=self.simulation_id,
                    **summary,
                )


class MujocoAudioMonitor:
    """Render passive casualty audio without taking motion authority."""

    def __init__(
        self,
        *,
        person_xy: tuple[float, float],
        control_dt_s: float,
        settings: SpatialAudioSettings,
        simulation_id: str = "",
        audit_log: Path = Path("runtime/everest-g1-events.jsonl"),
        mode: str,
    ) -> None:
        if control_dt_s <= 0:
            raise ValueError("control_dt_s must be positive")
        if not settings.enabled:
            raise ValueError("passive audio monitor requires enabled spatial audio settings")
        self.person_xy = person_xy
        self.control_dt_s = control_dt_s
        self.simulation_id = simulation_id or new_simulation_id()
        self.mode = mode
        self.audio = RescueAudio(settings)
        self.audit_log = JsonlAuditLog(audit_log)
        self._proximity_cued = False
        self._audio_sensor_elapsed_s = float("inf")
        self._closed = False
        self.audit_log.write(
            "spatial_audio_started",
            simulator="mujoco",
            simulation_id=self.simulation_id,
            mode=mode,
            motion_authority=False,
            acoustic_localization=self.audio.sensor is not None,
            cue_rendered=self.audio.renderer is not None,
        )

    @property
    def spatial_audio_path(self) -> Path | None:
        if self.audio.renderer is None:
            return None
        return self.audio.settings.path_for(self.simulation_id)

    def update(self, robot_qpos: np.ndarray) -> None:
        qpos = np.asarray(robot_qpos, dtype=np.float64)
        if qpos.shape != (7,) or not np.all(np.isfinite(qpos)):
            raise ValueError("MuJoCo audio pose must contain seven finite values")
        robot_xy = (float(qpos[0]), float(qpos[1]))
        robot_yaw = yaw_from_wxyz(qpos[3:7])
        self._audio_sensor_elapsed_s += self.control_dt_s
        estimate = self.audio.last_estimate
        if self.audio.sensor is not None and self._audio_sensor_elapsed_s >= 0.02:
            estimate = self.audio.observe(
                robot_xy=robot_xy,
                robot_yaw_rad=robot_yaw,
                source_xy=self.person_xy,
            )
            self._audio_sensor_elapsed_s = 0.0
        bearing = (
            estimate.bearing_rad
            if estimate is not None
            else body_bearing(
                robot_xy=robot_xy,
                robot_yaw_rad=robot_yaw,
                target_xy=self.person_xy,
            )
        )
        center_distance = math.dist(robot_xy, self.person_xy)
        surface_distance = max(0.0, center_distance - 0.25 - 0.30)
        self.audio.cue(
            dt_s=self.control_dt_s,
            bearing_rad=bearing,
            distance_m=surface_distance,
        )
        if surface_distance <= 0.15 and not self._proximity_cued:
            self._proximity_cued = True
            self.audio.mark_proximity_latched()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        summary = self.audio.close(self.simulation_id)
        if summary is not None:
            self.audit_log.write(
                "spatial_audio_written",
                simulator="mujoco",
                simulation_id=self.simulation_id,
                mode=self.mode,
                motion_authority=False,
                **summary,
            )


def yaw_from_wxyz(quaternion: np.ndarray) -> float:
    """Return planar yaw from a MuJoCo w, x, y, z quaternion."""

    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
