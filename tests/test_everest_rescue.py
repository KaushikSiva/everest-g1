import math

import pytest

from everest_g1.rescue import ApproachLimits, ProximityLatch, approach_person


def test_approach_command_is_bounded_and_body_relative() -> None:
    limits = ApproachLimits()
    command = approach_person(
        robot_xy=(0.0, 0.0),
        robot_yaw_rad=math.pi / 2,
        person_xy=(2.6, 0.75),
        limits=limits,
    )

    assert abs(command.forward_mps) <= limits.max_forward_mps
    assert abs(command.lateral_mps) <= limits.max_lateral_mps
    assert abs(command.yaw_rps) <= limits.max_yaw_rps
    assert not command.reached
    assert command.lateral_mps < 0


def test_approach_stops_at_touching_distance() -> None:
    limits = ApproachLimits()
    center_distance = limits.robot_radius_m + limits.person_radius_m + 0.1
    command = approach_person(
        robot_xy=(0.0, 0.0),
        robot_yaw_rad=0.0,
        person_xy=(center_distance, 0.0),
        limits=limits,
    )

    assert command.reached
    assert command.forward_mps == command.lateral_mps == command.yaw_rps == 0.0
    assert command.surface_distance_m == pytest.approx(0.1)


def test_proximity_requires_continuous_dwell() -> None:
    latch = ProximityLatch(threshold_m=0.15, dwell_s=0.25)
    for _ in range(10):
        assert not latch.update(0.1, 0.02)
    assert not latch.update(0.2, 0.02)
    for _ in range(12):
        assert not latch.update(0.1, 0.02)
    assert latch.update(0.1, 0.02)
    assert latch.update(5.0, 0.02)


def test_invalid_proximity_timing_is_rejected() -> None:
    with pytest.raises(ValueError):
        ProximityLatch(dwell_s=0)
    with pytest.raises(ValueError):
        ProximityLatch().update(0.1, 0)
