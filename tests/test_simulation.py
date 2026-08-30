import xml.etree.ElementTree as ET

import numpy as np
import pytest

import summit_sentinel.simulation as simulation_module
from summit_sentinel.config import SCENE_PATH
from summit_sentinel.simulation import SummitSentinelEnv


def test_model_compiles_with_named_g1_contract() -> None:
    env = SummitSentinelEnv(use_policy=False)
    assert env.model.nq == 19
    assert env.model.nv == 18
    assert env.model.nu == 12
    assert env.model.opt.timestep == 0.002
    assert env.policy_hz == 50.0
    assert env.model.joint("floating_base_joint").name == "floating_base_joint"
    assert env.model.geom("everest_terrain").name == "everest_terrain"


def test_terrain_visual_has_no_checker_grid_and_preserves_physics_contract() -> None:
    scene = ET.parse(SCENE_PATH).getroot()
    texture = scene.find("./asset/texture[@name='snow_rock']")
    terrain = scene.find("./worldbody/geom[@name='everest_terrain']")

    assert texture is not None
    assert texture.get("builtin") == "flat"
    assert all(item.get("builtin") != "checker" for item in scene.findall("./asset/texture"))
    assert terrain is not None
    assert terrain.get("type") == "hfield"
    assert terrain.get("hfield") == "everest_compressed"
    assert terrain.get("friction") == "0.9 0.01 0.001"
    assert terrain.get("condim") == "4"


def test_short_headless_hold_stability_smoke() -> None:
    env = SummitSentinelEnv(use_policy=False, auto_reset=False)
    results = [env.step() for _ in range(1000)]
    assert not any(result.fell for result in results)
    assert env.data.time == pytest.approx(2.0)
    assert np.all(np.isfinite(env.data.qpos))
    assert env.data.joint("floating_base_joint").qpos[2] > 0.65
    np.testing.assert_allclose(env.data.joint("floating_base_joint").qpos[:2], 0.0)


def test_bundled_policy_steps_with_47_by_12_contract() -> None:
    env = SummitSentinelEnv(use_policy=True, auto_reset=False)
    assert env.policy_mode == "torchscript", env.policy_warning
    result = env.step(np.asarray([0.0, 0.0, 0.0], dtype=np.float32))
    assert result.observation.shape == (47,)
    assert result.action.shape == (12,)
    assert result.torque.shape == (12,)
    assert np.all(np.isfinite(result.action))


def test_first_policy_action_after_reset_matches_fresh_environment() -> None:
    command = np.asarray([0.15, -0.1, 0.05], dtype=np.float32)
    reused = SummitSentinelEnv(use_policy=True, auto_reset=False)
    for _ in range(25):
        reused.step(command)
    reused.reset()
    after_reset = reused.step(command).action

    fresh = SummitSentinelEnv(use_policy=True, auto_reset=False)
    first_fresh = fresh.step(command).action

    np.testing.assert_array_equal(after_reset, first_fresh)


def test_emergency_stop_freezes_before_physics_and_requires_reset_then_resume() -> None:
    env = SummitSentinelEnv(use_policy=False, auto_reset=False)
    env.step(np.asarray([0.2, 0.0, 0.0], dtype=np.float32))
    stopped_qpos = env.data.qpos.copy()
    stopped_qvel = env.data.qvel.copy()
    stopped_time = float(env.data.time)

    assert env.emergency_stop("test")
    stopped = env.step(np.asarray([1.0, 1.0, 1.0], dtype=np.float32))

    assert stopped.emergency_stop_latched
    assert not stopped.physics_advanced
    assert stopped.time == stopped_time
    np.testing.assert_array_equal(env.data.qpos, stopped_qpos)
    np.testing.assert_array_equal(env.data.qvel, stopped_qvel)
    np.testing.assert_array_equal(env.data.ctrl, 0.0)
    np.testing.assert_array_equal(stopped.torque, 0.0)
    with pytest.raises(RuntimeError, match="reset before resume"):
        env.resume()

    env.reset()
    assert env.emergency_stop_latched
    assert not env.step().physics_advanced
    assert env.resume()
    resumed = env.step()
    assert resumed.physics_advanced
    assert not resumed.emergency_stop_latched
    assert env.data.time == pytest.approx(env.model.opt.timestep)


def test_latched_stop_is_first_branch_even_for_nan_command() -> None:
    env = SummitSentinelEnv(use_policy=False, auto_reset=False)
    env.emergency_stop("test")
    stopped_time = float(env.data.time)

    result = env.step(np.asarray([0.0, np.nan, 0.0], dtype=np.float32))

    assert result.emergency_stop_latched
    assert not result.physics_advanced
    assert env.data.time == stopped_time
    np.testing.assert_array_equal(env.data.ctrl, 0.0)


@pytest.mark.parametrize(
    "command",
    [np.zeros(2), np.zeros(4), np.asarray([0.0, np.nan, 0.0])],
)
def test_velocity_command_rejects_invalid_shape_or_nonfinite_values(command: np.ndarray) -> None:
    env = SummitSentinelEnv(use_policy=False)
    with pytest.raises(ValueError, match="three finite"):
        env.step(command)


def test_nonfinite_state_fails_closed_before_control_or_physics() -> None:
    env = SummitSentinelEnv(use_policy=False, auto_reset=False)
    initial_time = float(env.data.time)
    env.data.qvel[0] = np.nan

    result = env.step()

    assert not result.physics_advanced
    assert result.emergency_stop_latched
    assert result.stop_authority == "fault"
    assert env.data.time == initial_time
    np.testing.assert_array_equal(env.data.ctrl, 0.0)


def test_nonfinite_policy_action_fails_closed_before_control_or_physics() -> None:
    class NonFinitePolicy:
        mode = "test-nonfinite"

        def reset(self) -> None:
            return None

        def infer(self, observation: np.ndarray) -> np.ndarray:
            del observation
            return np.full(12, np.nan, dtype=np.float32)

    env = SummitSentinelEnv(use_policy=False, auto_reset=False)
    env.policy = NonFinitePolicy()
    result = env.step()

    assert not result.physics_advanced
    assert result.stop_authority == "fault"
    assert env.data.time == 0.0
    np.testing.assert_array_equal(env.data.ctrl, 0.0)


def test_nonfinite_target_or_torque_path_fails_closed_before_next_physics_step() -> None:
    env = SummitSentinelEnv(use_policy=False, auto_reset=False)
    env.step()
    initial_time = float(env.data.time)
    env.target_joint_position[0] = np.inf

    result = env.step()

    assert not result.physics_advanced
    assert result.stop_authority == "fault"
    assert env.data.time == initial_time
    np.testing.assert_array_equal(env.data.ctrl, 0.0)


def test_fall_reset_inhibits_locomotion_until_fresh_local_reset() -> None:
    env = SummitSentinelEnv(use_policy=False, auto_reset=True)
    env._fallen = lambda: True
    fell = env.step(np.asarray([0.7, 0.0, 0.0], dtype=np.float32))
    assert fell.reset
    assert fell.locomotion_inhibited
    env._fallen = lambda: False

    inhibited = env.step(np.asarray([0.7, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(inhibited.command, 0.0)
    env.reset(authority="remote")
    still_inhibited = env.step(np.asarray([0.7, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(still_inhibited.command, 0.0)

    env.reset(authority="local")
    accepted = env.step(np.asarray([0.7, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(accepted.command, [0.7, 0.0, 0.0])


def test_runtime_policy_failure_and_auto_reset_use_no_io_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InferenceFails:
        mode = "inference-fails"

        def reset(self) -> None:
            return None

        def infer(self, observation: np.ndarray) -> np.ndarray:
            del observation
            raise RuntimeError("inference failed")

    class ResetFails:
        mode = "reset-fails"

        def reset(self) -> None:
            raise RuntimeError("reset failed")

        def infer(self, observation: np.ndarray) -> np.ndarray:
            del observation
            return np.zeros(12, dtype=np.float32)

    env = SummitSentinelEnv(use_policy=False, auto_reset=True)

    def forbidden_loader(*args, **kwargs):
        del args, kwargs
        raise AssertionError("load_policy/filesystem path reached after initialization")

    monkeypatch.setattr(simulation_module, "load_policy", forbidden_loader)
    env.policy = InferenceFails()
    inference_result = env.step()
    assert inference_result.physics_advanced
    assert env.policy_mode == "hold-braced"
    assert "preconstructed" in env.policy_warning

    env.policy = ResetFails()
    env._fallen = lambda: True
    reset_result = env.step()
    assert reset_result.reset
    assert env.policy_mode == "hold-braced"
    assert "policy memory reset failure" in env.policy_warning
