import asyncio
import inspect

import pytest
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError

import summit_sentinel.mcp_server as mcp_server
from summit_sentinel.bridge import SQLiteBridge
from summit_sentinel.mcp_server import build_mcp_server

REQUIRED_TOOLS = {
    "get_sim_state",
    "get_joystick_state",
    "get_run_telemetry",
    "export_replay",
    "apply_scenario_conditions",
    "set_control_mode",
    "reset_simulation",
}
READ_TOOLS = {
    "get_sim_state",
    "get_joystick_state",
    "get_run_telemetry",
    "export_replay",
}
WRITE_TOOLS = {
    "apply_scenario_conditions",
    "set_control_mode",
    "reset_simulation",
}


def test_exact_connector_tools_schemas_annotations_and_enqueue_only_writes(tmp_path) -> None:
    path = tmp_path / "mcp.db"
    server = build_mcp_server(path)

    async def exercise() -> None:
        tools = await server.list_tools()
        names = {tool.name for tool in tools}
        tools_by_name = {tool.name: tool for tool in tools}
        assert names >= REQUIRED_TOOLS
        assert "queue_scenario" not in names
        assert "queue_velocity_command" not in names

        for name in READ_TOOLS:
            annotations = tools_by_name[name].annotations
            assert annotations is not None
            assert annotations.read_only_hint is True
            assert annotations.destructive_hint is False
            assert annotations.idempotent_hint is True
            assert annotations.open_world_hint is False
        for name in WRITE_TOOLS:
            tool = tools_by_name[name]
            assert tool.input_schema["additionalProperties"] is False
            assert tool.annotations is not None
            assert tool.annotations.read_only_hint is False
            assert tool.annotations.destructive_hint is True
            assert tool.annotations.idempotent_hint is False
            assert tool.annotations.open_world_hint is False
            assert "operator" in (tool.description or "")

        stop_annotations = tools_by_name["request_remote_stop"].annotations
        assert stop_annotations is not None
        assert stop_annotations.read_only_hint is False
        assert stop_annotations.destructive_hint is False
        assert "best-effort" in (tools_by_name["request_remote_stop"].description or "")
        assert "local emergency stop" in (tools_by_name["request_remote_stop"].description or "")

        conditions_schema = tools_by_name["apply_scenario_conditions"].input_schema
        assert set(conditions_schema["required"]) == {
            "friction",
            "wind_mps",
            "visibility_m",
            "snow_depth_m",
            "approval_ref",
        }
        control_schema = tools_by_name["set_control_mode"].input_schema
        assert set(control_schema["properties"]["mode"]["enum"]) == {"hold", "supervisory"}
        assert set(tools_by_name["reset_simulation"].input_schema["required"]) == {"approval_ref"}

        async with Client(server) as client:
            conditions = await client.call_tool(
                "apply_scenario_conditions",
                {
                    "friction": 9.0,
                    "wind_mps": 99.0,
                    "visibility_m": 1.0,
                    "snow_depth_m": 3.0,
                    "approval_ref": "tf-approval/conditions-1",
                },
            )
            assert not conditions.is_error
            assert conditions.structured_content["payload"] == {
                "friction": 1.5,
                "wind_mps": 30.0,
                "visibility_m": 10.0,
                "snow_depth_m": 0.5,
                "approval_ref": "tf-approval/conditions-1",
            }
            control = await client.call_tool(
                "set_control_mode",
                {"mode": "hold", "approval_ref": "tf-approval/control-1"},
            )
            reset = await client.call_tool(
                "reset_simulation", {"approval_ref": "tf-approval/reset-1"}
            )
            assert not control.is_error
            assert not reset.is_error

    asyncio.run(exercise())
    commands = SQLiteBridge(path).claim_commands()
    assert [command.kind for command in commands] == [
        "control_mode",
        "reset",
        "scenario_conditions",
    ]
    assert "MjData" not in inspect.getsource(mcp_server)
    assert "SummitSentinelEnv" not in inspect.getsource(mcp_server)


def test_mcp_boundary_rejects_unknown_fields_without_enqueuing(tmp_path) -> None:
    path = tmp_path / "strict.db"
    server = build_mcp_server(path)

    async def exercise() -> None:
        with pytest.raises(ToolError, match="joint_target"):
            await server.call_tool(
                "apply_scenario_conditions",
                {
                    "friction": 0.8,
                    "wind_mps": 5.0,
                    "visibility_m": 1000.0,
                    "snow_depth_m": 0.1,
                    "approval_ref": "tf-approval/strict-1",
                    "joint_target": [0.0],
                },
            )
        with pytest.raises(ToolError, match="bypass"):
            await server.call_tool(
                "set_control_mode",
                {
                    "mode": "hold",
                    "approval_ref": "tf-approval/strict-2",
                    "bypass": True,
                },
            )
        with pytest.raises(ToolError, match="clear_latch"):
            await server.call_tool(
                "reset_simulation",
                {"approval_ref": "tf-approval/strict-3", "clear_latch": True},
            )
        async with Client(server) as client:
            result = await client.call_tool(
                "apply_scenario_conditions",
                {
                    "friction": 0.8,
                    "wind_mps": 5.0,
                    "visibility_m": 1000.0,
                    "snow_depth_m": 0.1,
                    "approval_ref": "tf-approval/strict-4",
                    "joint_target": [0.0],
                },
            )
            assert result.is_error

    asyncio.run(exercise())
    status = SQLiteBridge(path).status()
    assert status["total_commands"] == 0


def test_joystick_telemetry_and_replay_reads_are_bounded_and_structured(tmp_path) -> None:
    path = tmp_path / "reads.db"
    bridge = SQLiteBridge(path)
    bridge.update_joystick_state(
        {
            "connected": False,
            "calibrated": True,
            "profile_name": "dual-sense.json",
            "device_name": "Wireless Controller",
            "normalized_axes": {"vx": 0.25, "vy": -0.5, "yaw": 0.75},
            "safety": {
                "reset": False,
                "emergency_stop": True,
                "resume": False,
                "quit": False,
                "stop_latched": True,
            },
        }
    )
    for sequence, position in enumerate(([0.0, 0.0, 2.0], [3.0, 4.0, 2.0], [6.0, 4.0, 2.0])):
        bridge.append_telemetry(
            {
                "recorded_at": 100.0 + sequence,
                "sim_time": float(sequence),
                "run_id": "mission-1",
                "sequence": sequence,
                "base_position": position,
                "physics_advanced": True,
                "fell": sequence == 1,
                "reset": sequence == 2,
                "emergency_stop_latched": sequence == 2,
            }
        )
    server = build_mcp_server(path)

    async def exercise() -> None:
        async with Client(server) as client:
            joystick = await client.call_tool("get_joystick_state", {})
            assert joystick.structured_content["connected"] is False
            assert joystick.structured_content["calibrated"] is True
            assert joystick.structured_content["normalized_axes"]["yaw"] == 0.75

            telemetry = await client.call_tool(
                "get_run_telemetry", {"limit": 2, "run_id": "mission-1"}
            )
            assert telemetry.structured_content["frame_count"] == 2
            assert len(telemetry.structured_content["frames"]) == 2

            replay = await client.call_tool("export_replay", {"limit": 10, "run_id": "mission-1"})
            content = replay.structured_content
            assert content["frame_count"] == 3
            assert content["mission_metrics"] == {
                "simulated_duration_s": 2.0,
                "horizontal_distance_m": 8.0,
                "falls": 1,
                "resets": 1,
                "stopped_frames": 1,
                "physics_steps_observed": 3,
            }
            assert content["storage"]["filesystem_write"] is False
            assert content["storage"]["logical_path"] == "telemetry://runs/mission-1"
            assert str(path) not in str(content)

    asyncio.run(exercise())


def test_streamable_http_app_has_documented_mcp_and_health_routes(tmp_path) -> None:
    server = build_mcp_server(tmp_path / "routes.db")
    app = server.streamable_http_app(
        streamable_http_path="/mcp", json_response=True, stateless_http=True
    )
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/mcp" in paths
    assert "/health" in paths
