from pathlib import Path

import numpy as np
import pytest

from everest_g1.autonomy.controller import AutonomousMujocoController
from everest_g1.autonomy.planning import (
    EnvironmentProfile,
    GeminiRoutePlanner,
    PlannerError,
    build_route_options,
)
from summit_sentinel.simulation import SummitSentinelEnv


def test_scan_routes_expose_every_requested_factor_and_unsafe_grade() -> None:
    options = build_route_options("scan", EnvironmentProfile())
    samples = [sample for option in options for sample in option.assessments]

    assert len(options) == 4
    assert all(sample.temperature_c < 0 for sample in samples)
    assert all(sample.wind_mps > 0 for sample in samples)
    assert all(sample.visibility_m > 0 for sample in samples)
    assert all(sample.snow_depth_m > 0 for sample in samples)
    assert all(sample.effective_friction > 0 for sample in samples)
    assert max(sample.slope_deg for sample in samples) > 14.0
    assert any(not option.hard_safe for option in options)
    prompt = GeminiRoutePlanner._prompt("scan", options)
    for factor in (
        "slope_deg",
        "temperature_c",
        "wind_mps",
        "visibility_m",
        "snow_depth_m",
        "effective_friction",
        "distance_from_start_m",
    ):
        assert factor in prompt


def test_offline_plan_is_explicit_and_selects_lowest_safe_risk() -> None:
    options = build_route_options("scan", EnvironmentProfile())
    planner = GeminiRoutePlanner(offline=True)

    selected = planner.select("scan", options, image_jpeg=None)

    assert planner.provider == "offline-deterministic"
    assert selected.hard_safe
    assert selected.route_id == "scan-sheltered-low-grade"


def test_live_planner_requires_key_and_camera() -> None:
    options = build_route_options("rescue", EnvironmentProfile())
    with pytest.raises(PlannerError, match="GEMINI_API_KEY"):
        GeminiRoutePlanner(api_key="").select("rescue", options, image_jpeg=None)
    with pytest.raises(PlannerError, match="front-camera JPEG"):
        GeminiRoutePlanner(api_key="test").select("rescue", options, image_jpeg=None)


def test_gemini_selects_only_named_locally_safe_route(monkeypatch: pytest.MonkeyPatch) -> None:
    from google import genai

    options = build_route_options("scan", EnvironmentProfile())
    selected_id = "scan-central-observation"

    class FakeResponse:
        text = (
            '{"route_id":"scan-central-observation",'
            '"reason":"moderate slope and good friction",'
            '"observations":"camera and environmental table checked"}'
        )

    class FakeModels:
        def generate_content(self, **kwargs):
            assert kwargs["model"] == "gemini-robotics-er-2-preview"
            assert len(kwargs["contents"]) == 2
            return FakeResponse()

    class FakeClient:
        def __init__(self, *, api_key: str):
            assert api_key == "test-key"
            self.models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(genai, "Client", FakeClient)
    planner = GeminiRoutePlanner(api_key="test-key")
    selected = planner.select("scan", options, image_jpeg=b"\xff\xd8frame\xff\xd9")

    assert selected.route_id == selected_id
    assert selected.hard_safe
    assert planner.provider == "gemini-er-2"


def test_carry_mode_attaches_only_after_proximity_dwell(tmp_path: Path) -> None:
    env = SummitSentinelEnv(use_policy=False, auto_reset=False)
    planner = GeminiRoutePlanner(offline=True)
    route = planner.select(
        "carry",
        build_route_options("carry", EnvironmentProfile()),
        image_jpeg=None,
    )
    controller = AutonomousMujocoController(
        env=env,
        mode="carry",
        route=route,
        planner=planner,
        audit_log=tmp_path / "events.jsonl",
    )
    root = env.data.joint("floating_base_joint").qpos
    try:
        assert not env.casualty_carrying
        for waypoint in route.approach_waypoints:
            root[:2] = waypoint
            controller.update(env)
        root[:2] = env.rescue_target_xy
        for _ in range(125):
            controller.update(env)

        assert controller.carry_proxy_attached
        assert env.casualty_carrying
        before = env.data.mocap_pos[env._casualty_mocap_id].copy()
        root[0] += 0.5
        env._update_carried_casualty_pose()
        after = env.data.mocap_pos[env._casualty_mocap_id]
        assert after[0] - before[0] == pytest.approx(0.5)
    finally:
        controller.close()


def test_autonomy_commands_remain_inside_conservative_bounds(tmp_path: Path) -> None:
    env = SummitSentinelEnv(use_policy=False, auto_reset=False)
    planner = GeminiRoutePlanner(offline=True)
    route = planner.select(
        "scan",
        build_route_options("scan", EnvironmentProfile()),
        image_jpeg=None,
    )
    controller = AutonomousMujocoController(
        env=env,
        mode="scan",
        route=route,
        planner=planner,
        audit_log=tmp_path / "events.jsonl",
    )
    try:
        command = controller.update(env)
    finally:
        controller.close()

    assert np.all(np.isfinite(command))
    assert abs(command[0]) <= 0.18
    assert abs(command[1]) <= 0.10
    assert abs(command[2]) <= 0.25
