"""Deterministic proximity and approach logic for the rescue scenario."""

from __future__ import annotations

import math
from dataclasses import dataclass

from everest_g1.models import NavigationCommand


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class ApproachLimits:
    """Conservative commissioning limits for a loosely-coupled WBC."""

    touch_distance_m: float = 0.15
    robot_radius_m: float = 0.25
    person_radius_m: float = 0.30
    max_forward_mps: float = 0.20
    max_lateral_mps: float = 0.12
    max_yaw_rps: float = 0.30
    position_gain: float = 0.55
    yaw_gain: float = 0.80


class ProximityLatch:
    """Latches only after proximity is continuously true for a dwell period."""

    def __init__(self, threshold_m: float = 0.15, dwell_s: float = 0.25) -> None:
        if threshold_m < 0 or dwell_s <= 0:
            raise ValueError("threshold_m must be non-negative and dwell_s must be positive")
        self.threshold_m = threshold_m
        self.dwell_s = dwell_s
        self._inside_s = 0.0
        self.latched = False

    def update(self, distance_m: float, dt_s: float) -> bool:
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        if self.latched:
            return True
        if distance_m <= self.threshold_m:
            self._inside_s += dt_s
            self.latched = self._inside_s + 1e-9 >= self.dwell_s
        else:
            self._inside_s = 0.0
        return self.latched

    def reset(self) -> None:
        self._inside_s = 0.0
        self.latched = False


def approach_person(
    *,
    robot_xy: tuple[float, float],
    robot_yaw_rad: float,
    person_xy: tuple[float, float],
    limits: ApproachLimits,
    reached: bool = False,
) -> NavigationCommand:
    """Return a conservative body-frame command toward a fixed observed target."""

    dx = person_xy[0] - robot_xy[0]
    dy = person_xy[1] - robot_xy[1]
    center_distance = math.hypot(dx, dy)
    surface_distance = max(0.0, center_distance - limits.robot_radius_m - limits.person_radius_m)
    if reached or surface_distance <= limits.touch_distance_m:
        return NavigationCommand(0.0, 0.0, 0.0, surface_distance, True)

    cos_yaw = math.cos(robot_yaw_rad)
    sin_yaw = math.sin(robot_yaw_rad)
    body_x = cos_yaw * dx + sin_yaw * dy
    body_y = -sin_yaw * dx + cos_yaw * dy
    target_yaw = math.atan2(dy, dx)
    yaw_error = _wrap_angle(target_yaw - robot_yaw_rad)

    return NavigationCommand(
        forward_mps=_clamp(limits.position_gain * body_x, limits.max_forward_mps),
        lateral_mps=_clamp(limits.position_gain * body_y, limits.max_lateral_mps),
        yaw_rps=_clamp(limits.yaw_gain * yaw_error, limits.max_yaw_rps),
        surface_distance_m=surface_distance,
        reached=False,
    )


def body_bearing(
    *,
    robot_xy: tuple[float, float],
    robot_yaw_rad: float,
    target_xy: tuple[float, float],
) -> float:
    """Return the bearing to a target in the robot body frame (0 ahead, + left)."""

    dx = target_xy[0] - robot_xy[0]
    dy = target_xy[1] - robot_xy[1]
    return _wrap_angle(math.atan2(dy, dx) - robot_yaw_rad)


def target_from_bearing(
    *,
    robot_xy: tuple[float, float],
    robot_yaw_rad: float,
    bearing_rad: float,
    surface_distance_m: float,
    limits: ApproachLimits,
) -> tuple[float, float]:
    """Place a steering target at ``bearing_rad`` and the measured range.

    Bearing may come from a noisy sensor. The range never does: it is the same
    measured surface distance that gates the stop, re-expressed as a centre
    distance, so a wrong bearing can only steer badly and never fake arrival.
    """

    center_distance = surface_distance_m + limits.robot_radius_m + limits.person_radius_m
    heading = robot_yaw_rad + bearing_rad
    return (
        robot_xy[0] + center_distance * math.cos(heading),
        robot_xy[1] + center_distance * math.sin(heading),
    )
