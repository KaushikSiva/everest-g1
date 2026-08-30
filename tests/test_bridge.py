import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from summit_sentinel.agent_runtime import (
    BridgeRuntimeWorker,
    RuntimeCommandApplier,
)
from summit_sentinel.bridge import SQLiteBridge, scenario_velocity, validate_command
from summit_sentinel.simulation import SummitSentinelEnv
from summit_sentinel.telemetry import TelemetryRecorder


def test_velocity_and_scenario_commands_are_allowlisted_and_clamped() -> None:
    velocity = validate_command(
        "velocity", {"vx": 9.0, "vy": -9.0, "yaw": 0.5, "duration_s": 100.0}
    )
    assert velocity == {"vx": 1.0, "vy": -1.0, "yaw": 0.5, "duration_s": 10.0}
    assert scenario_velocity({"name": "walk_forward", "speed_scale": 5.0, "duration_s": 0.1}) == (
        0.35,
        0.0,
        0.0,
    )

    with pytest.raises(ValueError, match="unexpected command fields"):
        validate_command(
            "velocity",
            {"vx": 0.0, "vy": 0.0, "yaw": 0.0, "duration_s": 1.0, "torque": 10},
        )
    with pytest.raises(ValueError, match="scenario name"):
        scenario_velocity({"name": "summit_sprint", "speed_scale": 1.0, "duration_s": 2.0})


def test_scenario_conditions_and_control_mode_are_strictly_validated_and_clamped() -> None:
    conditions = validate_command(
        "scenario_conditions",
        {
            "friction": 9.0,
            "wind_mps": 99.0,
            "visibility_m": 1.0,
            "snow_depth_m": 2.0,
            "approval_ref": "tf-run/conditions-1",
        },
    )
    assert conditions == {
        "friction": 1.5,
        "wind_mps": 30.0,
        "visibility_m": 10.0,
        "snow_depth_m": 0.5,
        "approval_ref": "tf-run/conditions-1",
    }
    assert validate_command(
        "control_mode", {"mode": "hold", "approval_ref": "tf-run/control-1"}
    ) == {"mode": "hold", "approval_ref": "tf-run/control-1"}

    invalid_conditions = {
        "friction": 0.8,
        "wind_mps": 2.0,
        "visibility_m": 1000.0,
        "snow_depth_m": 0.1,
        "approval_ref": "tf-run/conditions-2",
    }
    with pytest.raises(ValueError, match="unexpected command fields"):
        validate_command("scenario_conditions", {**invalid_conditions, "torque": 1.0})
    with pytest.raises(ValueError, match="finite number"):
        validate_command("scenario_conditions", {**invalid_conditions, "wind_mps": np.nan})
    with pytest.raises(ValueError, match="missing command fields"):
        payload = dict(invalid_conditions)
        del payload["snow_depth_m"]
        validate_command("scenario_conditions", payload)
    with pytest.raises(ValueError, match="external audit reference"):
        validate_command("scenario_conditions", {**invalid_conditions, "approval_ref": "bad ref"})
    with pytest.raises(ValueError, match="control mode"):
        validate_command(
            "control_mode", {"mode": "direct_torque", "approval_ref": "tf-run/control-2"}
        )


def test_sqlite_wal_queue_is_claimed_once_across_connections(tmp_path) -> None:
    path = tmp_path / "bridge.db"
    writer = SQLiteBridge(path)
    reader = SQLiteBridge(path)
    queued = writer.enqueue_command(
        "velocity", {"vx": 2.0, "vy": 0.0, "yaw": 0.0, "duration_s": 20.0}
    )

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    claimed = reader.claim_commands()
    assert [command.id for command in claimed] == [queued["command_id"]]
    assert writer.claim_commands() == []
    assert claimed[0].payload["vx"] == 1.0
    assert claimed[0].payload["duration_s"] == 10.0
    assert reader.complete_command(claimed[0].id, applied=True, message="done")
    assert writer.command_status(claimed[0].id)["status"] == "applied"


def test_runtime_worker_preserves_claim_and_completion_queue_semantics(tmp_path) -> None:
    bridge = SQLiteBridge(tmp_path / "runtime-worker.db")
    env = SummitSentinelEnv(use_policy=False, auto_reset=False)
    applier = RuntimeCommandApplier()
    queued = bridge.enqueue_command(
        "velocity", {"vx": 0.4, "vy": 0.0, "yaw": 0.0, "duration_s": 1.0}
    )
    worker = BridgeRuntimeWorker(bridge, poll_hz=20.0)
    worker.start()
    try:
        deadline = time.monotonic() + 1.0
        commands = []
        while not commands and time.monotonic() < deadline:
            commands = worker.take_commands()
            time.sleep(0.005)
        assert [command.id for command in commands] == [queued["command_id"]]
        velocity, outcome = applier.consume(env, commands, now=0.0)
        assert velocity is not None
        assert velocity[0] == pytest.approx(0.4)
        assert outcome is not None
        assert worker.submit_outcome(outcome)

        deadline = time.monotonic() + 1.0
        while bridge.command_status(queued["command_id"])["status"] != "applied":
            assert time.monotonic() < deadline
            time.sleep(0.005)
    finally:
        worker.close()


def test_telemetry_recorder_is_rate_limited_and_storage_is_bounded(tmp_path) -> None:
    bridge = SQLiteBridge(tmp_path / "telemetry.db", max_telemetry_rows=2)
    recorder = TelemetryRecorder(bridge, hz=15.0, run_id="test-run")
    env = SummitSentinelEnv(use_policy=False, auto_reset=False)

    first = env.step(np.zeros(3, dtype=np.float32))
    assert recorder.maybe_record(env, first, monotonic_now=0.0, recorded_at=100.0)
    second = env.step(np.zeros(3, dtype=np.float32))
    assert not recorder.maybe_record(env, second, monotonic_now=0.05, recorded_at=100.05)
    assert recorder.maybe_record(env, second, monotonic_now=0.07, recorded_at=100.07)
    third = env.step(np.zeros(3, dtype=np.float32))
    assert recorder.maybe_record(env, third, monotonic_now=0.14, recorded_at=100.14)

    replay = bridge.recent_telemetry(100)
    assert len(replay) == 2
    assert [frame["sequence"] for frame in replay] == [1, 2]
    with pytest.raises(ValueError, match="between 1 and 1000"):
        bridge.recent_telemetry(1001)


def test_command_applier_enforces_reset_before_resume(tmp_path) -> None:
    bridge = SQLiteBridge(tmp_path / "commands.db")
    env = SummitSentinelEnv(use_policy=False, auto_reset=False)
    applier = RuntimeCommandApplier()

    env.emergency_stop("local operator", authority="local")
    remote_reset = bridge.enqueue_command("reset", {"approval_ref": "tf-reset-1"})
    premature = bridge.enqueue_command("resume", {"approval_ref": "tf-resume-1"})
    _, outcome = applier.consume(env, bridge.claim_commands(), now=0.0)
    assert env.emergency_stop_latched
    assert env.stop_authority == "local"
    assert outcome is not None
    assert outcome.completions == (
        (
            remote_reset["command_id"],
            False,
            "remote reset cannot acknowledge a local or fault stop",
        ),
        (premature["command_id"], False, "remote resume cannot clear a local or fault stop"),
    )

    env.reset(authority="local")
    assert env.resume(authority="local")
    assert not env.emergency_stop_latched


def test_command_applier_supervisory_stop_preempts_batch(tmp_path) -> None:
    bridge = SQLiteBridge(tmp_path / "supervisory.db")
    env = SummitSentinelEnv(use_policy=False, auto_reset=False)
    applier = RuntimeCommandApplier()
    motion = bridge.enqueue_command(
        "velocity", {"vx": 0.5, "vy": 0.0, "yaw": 0.0, "duration_s": 2.0}
    )
    stop = bridge.enqueue_command("remote_stop", {"approval_ref": "tf-stop-1"})

    velocity, outcome = applier.consume(env, bridge.claim_commands(), now=0.0)
    assert velocity is None
    assert outcome is not None
    assert env.stop_authority == "supervisory"
    assert outcome.completions == (
        (stop["command_id"], True, "best-effort supervisory stop latched"),
    )
    assert outcome.rejected_ids == (motion["command_id"],)

    reset = bridge.enqueue_command("reset", {"approval_ref": "tf-reset-1"})
    _, reset_outcome = applier.consume(env, bridge.claim_commands(), now=0.1)
    assert env.emergency_stop_latched
    assert reset_outcome is not None
    assert reset_outcome.completions[0][:2] == (reset["command_id"], True)
    bridge.advance_run_epoch(reset_outcome.advance_epoch_reason or "test reset")
    resume = bridge.enqueue_command("resume", {"approval_ref": "tf-resume-1"})
    _, resume_outcome = applier.consume(env, bridge.claim_commands(), now=0.2)
    assert not env.emergency_stop_latched
    assert resume_outcome is not None
    assert resume_outcome.completions[0][:2] == (resume["command_id"], True)


def test_command_applier_applies_conditions_and_modes_only_at_simulator_boundary(tmp_path) -> None:
    bridge = SQLiteBridge(tmp_path / "conditions.db")
    env = SummitSentinelEnv(use_policy=False, auto_reset=False)
    applier = RuntimeCommandApplier()
    initial_control = env.data.ctrl.copy()

    conditions = bridge.enqueue_command(
        "scenario_conditions",
        {
            "friction": 1.0,
            "wind_mps": 12.0,
            "visibility_m": 500.0,
            "snow_depth_m": 0.5,
            "approval_ref": "tf-run/conditions-3",
        },
    )
    velocity, outcome = applier.consume(env, bridge.claim_commands(), now=0.0)
    assert velocity is None
    assert outcome is not None
    assert outcome.completions[0][:2] == (conditions["command_id"], True)
    assert env.scenario_conditions == {
        "friction": 1.0,
        "wind_mps": 12.0,
        "visibility_m": 500.0,
        "snow_depth_m": 0.5,
        "effective_friction": pytest.approx(0.4),
    }
    assert env.model.geom("everest_terrain").friction[0] == pytest.approx(0.4)
    np.testing.assert_array_equal(env.data.ctrl, initial_control)

    env.emergency_stop("operator", authority="local")
    hold = bridge.enqueue_command(
        "control_mode", {"mode": "hold", "approval_ref": "tf-run/control-3"}
    )
    _, outcome = applier.consume(env, bridge.claim_commands(), now=0.1)
    assert env.control_mode == "hold"
    assert outcome is not None
    assert outcome.completions[0][:2] == (hold["command_id"], True)
    assert env.emergency_stop_latched
    assert env.stop_authority == "local"

    supervisory = bridge.enqueue_command(
        "control_mode", {"mode": "supervisory", "approval_ref": "tf-run/control-4"}
    )
    _, outcome = applier.consume(env, bridge.claim_commands(), now=0.2)
    assert env.control_mode == "supervisory"
    assert outcome is not None
    assert outcome.completions[0][:2] == (supervisory["command_id"], True)
    assert env.emergency_stop_latched
    assert env.stop_authority == "local"


def test_command_ttl_priority_coalescing_backlog_and_epoch(tmp_path) -> None:
    bridge = SQLiteBridge(tmp_path / "bounds.db", max_pending_commands=2)
    expired = bridge.enqueue_command(
        "velocity",
        {"vx": 0.1, "vy": 0.0, "yaw": 0.0, "duration_s": 1.0},
        ttl_s=0.5,
        now=100.0,
    )
    assert bridge.claim_commands(now=101.0) == []
    assert bridge.command_status(expired["command_id"])["status"] == "expired"

    old_motion = bridge.enqueue_command(
        "velocity",
        {"vx": 0.1, "vy": 0.0, "yaw": 0.0, "duration_s": 1.0},
        now=200.0,
    )
    new_motion = bridge.enqueue_command(
        "scenario",
        {"name": "stand", "speed_scale": 0.0, "duration_s": 1.0},
        now=200.1,
    )
    assert bridge.command_status(old_motion["command_id"])["status"] == "superseded"
    bridge.advance_run_epoch()
    assert bridge.command_status(new_motion["command_id"])["status"] == "stale"

    reset = bridge.enqueue_command("reset", {"approval_ref": "tf-reset-2"}, now=300.0)
    resume = bridge.enqueue_command("resume", {"approval_ref": "tf-resume-2"}, now=300.0)
    with pytest.raises(RuntimeError, match="backlog is full"):
        bridge.enqueue_command(
            "velocity",
            {"vx": 0.0, "vy": 0.0, "yaw": 0.0, "duration_s": 1.0},
            now=300.0,
        )
    stop = bridge.enqueue_command("remote_stop", {"approval_ref": "tf-stop-2"}, now=300.0)
    assert bridge.command_status(resume["command_id"])["status"] == "preempted"
    claimed = bridge.claim_commands(now=300.1)
    assert claimed[0].id == stop["command_id"]
    assert claimed[1].id == reset["command_id"]


def test_atomic_claim_is_unique_across_concurrent_consumers(tmp_path) -> None:
    path = tmp_path / "concurrent.db"
    bridge = SQLiteBridge(path)
    bridge.enqueue_command("reset", {"approval_ref": "tf-reset-3"})
    bridge.enqueue_command("resume", {"approval_ref": "tf-resume-3"})
    bridge.enqueue_command("remote_stop", {"approval_ref": "tf-stop-3"})

    with ThreadPoolExecutor(max_workers=2) as pool:
        batches = list(pool.map(lambda _: SQLiteBridge(path).claim_commands(), range(2)))
    ids = [command.id for batch in batches for command in batch]
    assert len(ids) == 3
    assert len(set(ids)) == 3


def test_environment_reset_neutralizes_active_velocity_and_advances_epoch(tmp_path) -> None:
    bridge = SQLiteBridge(tmp_path / "active-reset.db")
    env = SummitSentinelEnv(use_policy=False, auto_reset=False)
    applier = RuntimeCommandApplier()
    bridge.enqueue_command("velocity", {"vx": 0.8, "vy": 0.0, "yaw": 0.0, "duration_s": 10.0})
    velocity, _ = applier.consume(env, bridge.claim_commands(), now=0.0)
    assert velocity is not None
    assert velocity[0] == pytest.approx(0.8)
    initial_epoch = bridge.current_run_epoch()

    outcome = applier.on_environment_reset()
    assert outcome.advance_epoch_reason == "environment reset invalidated prior commands"
    assert bridge.advance_run_epoch(outcome.advance_epoch_reason) == initial_epoch + 1
    assert applier.active_velocity(now=0.1) is None


def test_restart_invalidates_queued_and_claimed_commands_and_bounds_history(tmp_path) -> None:
    path = tmp_path / "restart.db"
    first_bridge = SQLiteBridge(path, max_pending_commands=2, max_command_history=2)
    assert first_bridge.begin_simulator_run() == 1
    claimed_command = first_bridge.enqueue_command("reset", {"approval_ref": "tf-reset-4"})
    queued_command = first_bridge.enqueue_command("resume", {"approval_ref": "tf-resume-4"})
    assert [item.id for item in first_bridge.claim_commands(limit=1)] == [
        claimed_command["command_id"]
    ]

    second_bridge = SQLiteBridge(path, max_pending_commands=2, max_command_history=2)
    assert second_bridge.begin_simulator_run() == 2
    assert second_bridge.claim_commands() == []
    assert second_bridge.command_status(claimed_command["command_id"])["status"] == "stale"
    assert second_bridge.command_status(queued_command["command_id"])["status"] == "stale"

    for _ in range(8):
        crashed = SQLiteBridge(path, max_pending_commands=2, max_command_history=2)
        crashed.enqueue_command("reset", {"approval_ref": "tf-reset-crash"})
        assert len(crashed.claim_commands(limit=1)) == 1
        restarted = SQLiteBridge(path, max_pending_commands=2, max_command_history=2)
        restarted.begin_simulator_run()

    status = restarted.status()
    assert status["total_commands"] <= status["max_command_rows"]
    assert status["total_commands"] <= 2
    assert not {"queued", "claimed"} & status["command_counts"].keys()
