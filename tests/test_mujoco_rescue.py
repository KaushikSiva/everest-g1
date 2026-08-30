import json
import math
from pathlib import Path

import numpy as np
import pytest

from everest_g1.beacon import ARM_VALUE, BeaconSettings
from everest_g1.models import RescueObservation
from everest_g1.mujoco import MujocoRescueController, yaw_from_wxyz


def test_mujoco_rescue_stops_dwells_and_logs_disarmed(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    controller = MujocoRescueController(
        person_xy=(0.65, 0.0),
        control_dt_s=0.01,
        audit_log=log_path,
        simulation_id="mujoco-dry",
    )
    qpos = np.asarray([0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0])

    for _ in range(24):
        np.testing.assert_array_equal(controller.update(qpos), 0.0)
        assert not controller.latch.latched
    np.testing.assert_array_equal(controller.update(qpos), 0.0)
    controller.close()

    assert controller.latch.latched
    assert not controller.live_call_armed
    assert not controller.call_submitted
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "simulation_started",
        "proximity_reached_call_disarmed",
    ]
    assert all(record.get("simulator") == "mujoco" for record in records)


def test_mujoco_rescue_armed_path_submits_exactly_one_beacon_incident(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import everest_g1.beacon as beacon_module

    monkeypatch.setenv("EVEREST_ARM_LIVE_CALL", ARM_VALUE)
    monkeypatch.setenv("BEACON_API_URL", "https://beacon.test")
    monkeypatch.setenv("BEACON_API_TOKEN", "test-token")
    calls: list[RescueObservation] = []

    def fake_post(_settings: BeaconSettings, observation: RescueObservation) -> dict[str, object]:
        calls.append(observation)
        return {"incident_id": "inc-mujoco", "status": "queued"}

    monkeypatch.setattr(beacon_module, "post_incident", fake_post)
    controller = MujocoRescueController(
        person_xy=(0.65, 0.0),
        control_dt_s=0.25,
        arm_live_call=True,
        audit_log=tmp_path / "events.jsonl",
        simulation_id="mujoco-live-test",
    )
    qpos = np.asarray([0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0])

    controller.update(qpos)
    controller.update(qpos)
    controller.close()

    assert controller.call_submitted
    assert len(calls) == 1
    assert calls[0].simulation_id == "mujoco-live-test"
    assert calls[0].distance_m == pytest.approx(0.1)
    assert calls[0].observed_state == "motionless_adult_in_snow"


def test_mujoco_yaw_and_pose_validation(tmp_path: Path) -> None:
    quaternion = np.asarray([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
    assert yaw_from_wxyz(quaternion) == pytest.approx(math.pi / 2)
    controller = MujocoRescueController(
        person_xy=(1.0, 0.0),
        control_dt_s=0.01,
        audit_log=tmp_path / "events.jsonl",
    )
    try:
        with pytest.raises(ValueError, match="seven finite"):
            controller.update(np.zeros(6))
    finally:
        controller.close()
