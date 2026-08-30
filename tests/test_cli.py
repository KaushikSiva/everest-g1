import inspect
import threading
import time
from contextlib import nullcontext
from unittest.mock import Mock

import numpy as np
import pytest

from summit_sentinel import cli
from summit_sentinel.joystick import OperatorInput
from summit_sentinel.simulation import SummitSentinelEnv


def test_viewer_sync_is_throttled_to_sixty_hz() -> None:
    syncs = sum(cli._viewer_sync_due(step, 0.002) for step in range(1, 501))
    assert syncs == 60


def test_viewer_camera_opens_on_wide_interactive_summit_view() -> None:
    viewer = Mock()
    viewer.lock.return_value = nullcontext()
    viewer.cam.lookat = np.zeros(3, dtype=np.float64)

    cli._configure_viewer_camera(viewer)

    viewer.lock.assert_called_once_with()
    np.testing.assert_array_equal(viewer.cam.lookat, cli.VIEWER_CAMERA_LOOKAT)
    assert viewer.cam.distance == cli.VIEWER_CAMERA_DISTANCE
    assert viewer.cam.azimuth == cli.VIEWER_CAMERA_AZIMUTH
    assert viewer.cam.elevation == cli.VIEWER_CAMERA_ELEVATION
    assert cli.VIEWER_CAMERA_DISTANCE > 0.0
    assert -90.0 < cli.VIEWER_CAMERA_ELEVATION < 0.0


def test_macos_viewer_error_prints_exact_mjpython_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.delenv("MJPYTHON_BIN", raising=False)
    argv = ["--mode", "viewer", "--seconds", "12", "--joystick"]

    with pytest.raises(SystemExit) as error:
        cli.main(argv)

    assert error.value.code == 2
    assert (
        "uv run mjpython -m summit_sentinel --mode viewer --seconds 12 --joystick"
        in capsys.readouterr().err
    )


def test_estop_latency_is_zero_intervening_mocked_physics_steps() -> None:
    events: list[tuple[str, int]] = []
    env = Mock(spec=SummitSentinelEnv)
    physics_steps = 0

    def stop(*args, **kwargs) -> None:
        del args, kwargs
        events.append(("stop", physics_steps))

    def step(*args, **kwargs) -> None:
        nonlocal physics_steps
        del args, kwargs
        events.append(("step", physics_steps))
        physics_steps += 1

    env.emergency_stop.side_effect = stop
    env.step.side_effect = step
    operator = OperatorInput(
        command=np.zeros(3, dtype=np.float32),
        emergency_stop=True,
    )

    warning, did_reset = cli._apply_operator_safety(env, operator)
    env.step(operator.command)

    assert events == [("stop", 0), ("step", 0)]
    assert physics_steps == 1
    assert warning is None
    assert not did_reset


def test_joystick_is_closed_when_environment_initialization_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Mock()
    monkeypatch.setattr(cli, "PSJoystick", lambda *args, **kwargs: source)
    monkeypatch.setattr(
        cli,
        "SummitSentinelEnv",
        Mock(side_effect=RuntimeError("environment initialization failed")),
    )
    args = cli.build_parser().parse_args(["--mode", "headless", "--joystick", "--seconds", "0"])

    with pytest.raises(RuntimeError, match="environment initialization failed"):
        cli._run(args)

    source.close.assert_called_once_with()


def test_runtime_loop_contains_no_synchronous_bridge_storage_calls() -> None:
    source = inspect.getsource(cli._run_with_source)
    sample_index = source.index("operator = source.sample()")
    local_safety_index = source.index("_apply_operator_safety(env, operator)")
    queue_index = source.index("bridge_worker.take_commands()")
    assert sample_index < local_safety_index < queue_index
    for forbidden in (
        ".claim_commands(",
        ".complete_command(",
        ".advance_run_epoch(",
        ".append_telemetry(",
        ".update_joystick_state(",
    ):
        assert forbidden not in source


def test_bridge_worker_handoff_and_shutdown_are_bounded_when_storage_blocks() -> None:
    from summit_sentinel.agent_runtime import BridgeRuntimeWorker

    entered = threading.Event()
    release = threading.Event()

    class BlockingBridge:
        def claim_commands(self):
            entered.set()
            release.wait(timeout=2.0)
            return []

    worker = BridgeRuntimeWorker(BlockingBridge())
    worker.start()
    assert entered.wait(timeout=0.5)

    env = SummitSentinelEnv(use_policy=False, auto_reset=False)
    stopped_time = float(env.data.time)
    operator = OperatorInput(
        command=np.ones(3, dtype=np.float32),
        emergency_stop=True,
        connected=False,
    )
    cli._apply_operator_safety(env, operator)
    stopped = env.step(operator.command)
    assert not stopped.physics_advanced
    assert env.data.time == stopped_time
    assert env.emergency_stop_reason == "joystick disconnected"

    started = time.monotonic()
    worker.publish_latest({"sequence": 0}, {"connected": False})
    assert time.monotonic() - started < 0.05
    worker.close(timeout=0.01)
    assert "shutdown deadline" in str(worker.failure)
    release.set()


def test_background_storage_failure_latches_fault_before_physics() -> None:
    env = SummitSentinelEnv(use_policy=False, auto_reset=False)
    initial_time = float(env.data.time)

    assert cli._apply_bridge_failure(env, "database locked")
    stopped = env.step(np.ones(3, dtype=np.float32))

    assert not stopped.physics_advanced
    assert stopped.stop_authority == "fault"
    assert env.data.time == initial_time
    assert "database locked" in str(env.emergency_stop_reason)
