import numpy as np
import pytest

from summit_sentinel import control
from summit_sentinel.config import POLICY_PATH, POLICY_SHA256, load_config
from summit_sentinel.control import ObservationBuilder, pd_control


def test_bundled_policy_digest_matches_trusted_constant() -> None:
    assert control.policy_sha256(POLICY_PATH) == POLICY_SHA256


def test_mismatched_or_untrusted_policy_is_never_deserialized(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy_path = tmp_path / "untrusted.pt"
    policy_path.write_bytes(b"not torchscript")

    def forbidden_loader(*args, **kwargs):
        del args, kwargs
        raise AssertionError("deserializer reached")

    monkeypatch.setattr(control, "TorchScriptPolicy", forbidden_loader)
    policy, warning = control.load_policy(
        policy_path,
        12,
        expected_sha256="0" * 64,
    )
    assert policy.mode == "hold-braced"
    assert "mismatch before deserialization" in str(warning)

    policy, warning = control.load_policy(policy_path, 12, expected_sha256=None)
    assert policy.mode == "hold-braced"
    assert "no trusted policy SHA-256" in str(warning)


def test_exact_observation_layout_and_dimensions() -> None:
    config = load_config().controller
    builder = ObservationBuilder(config)
    command = np.asarray([0.5, -0.25, 0.4], dtype=np.float32)
    previous = np.linspace(-0.5, 0.5, 12, dtype=np.float32)
    obs = builder.build(
        angular_velocity=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        quaternion_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        command=command,
        joint_position=config.default_angles.copy(),
        joint_velocity=np.ones(12, dtype=np.float32),
        previous_action=previous,
        sim_time=0.0,
    )
    assert obs.shape == (47,)
    np.testing.assert_allclose(obs[0:3], [0.25, 0.5, 0.75])
    np.testing.assert_allclose(obs[3:6], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(obs[6:9], command * config.command_scale)
    np.testing.assert_allclose(obs[9:21], 0.0)
    np.testing.assert_allclose(obs[21:33], 0.05)
    np.testing.assert_allclose(obs[33:45], previous)
    np.testing.assert_allclose(obs[45:47], [0.0, 1.0], atol=1e-7)


def test_pd_controller_has_twelve_bounded_outputs() -> None:
    config = load_config().controller
    torque = pd_control(
        target_q=np.full(12, 100.0, dtype=np.float32),
        q=np.zeros(12, dtype=np.float32),
        kp=config.kps,
        dq=np.zeros(12, dtype=np.float32),
        kd=config.kds,
        torque_limits=config.torque_limits,
    )
    assert torque.shape == (12,)
    np.testing.assert_allclose(torque, config.torque_limits)
