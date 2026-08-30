"""Command-line runner for headless and interactive Summit Sentinel modes."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
import time
from pathlib import Path

import numpy as np

from everest_g1.autonomy import (
    GEMINI_ROBOTICS_MODEL,
    AutonomousMujocoController,
    EnvironmentProfile,
    GeminiRoutePlanner,
    build_route_options,
)
from everest_g1.mujoco import MujocoRescueController
from summit_sentinel.agent_runtime import (
    BridgeRuntimeWorker,
    RuntimeCommandApplier,
    StorageOutcome,
)
from summit_sentinel.bridge import SQLiteBridge
from summit_sentinel.config import load_config
from summit_sentinel.joystick import (
    FixedCommandSource,
    JoystickCalibration,
    OperatorInput,
    PSJoystick,
    capture_calibration,
    list_joysticks,
)
from summit_sentinel.simulation import SummitSentinelEnv
from summit_sentinel.telemetry import RuntimeTelemetrySampler, validate_telemetry_hz
from summit_sentinel.terrain import TERRAIN_SEED

VIEWER_HZ = 60.0
VIEWER_CAMERA_LOOKAT = (0.0, 0.0, 0.85)
VIEWER_CAMERA_DISTANCE = 7.0
VIEWER_CAMERA_AZIMUTH = 135.0
VIEWER_CAMERA_ELEVATION = -22.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("headless", "viewer"), default="viewer")
    parser.add_argument(
        "--seconds", type=float, default=30.0, help="wall/simulation seconds to run"
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=(TERRAIN_SEED,),
        default=TERRAIN_SEED,
        help=f"fixed checked-terrain seed (currently {TERRAIN_SEED})",
    )
    parser.add_argument("--joystick", action="store_true", help="enable a connected PS joystick")
    parser.add_argument("--joystick-index", type=int, default=0)
    parser.add_argument("--list-joysticks", action="store_true")
    parser.add_argument(
        "--joystick-calibration",
        type=Path,
        help="JSON center/range profile created by --calibrate-joystick",
    )
    parser.add_argument(
        "--calibrate-joystick",
        type=Path,
        metavar="OUTPUT.json",
        help="interactively calibrate command axes and exit",
    )
    parser.add_argument("--calibration-seconds", type=float, default=6.0)
    parser.add_argument("--vx", type=float, default=0.0, help="fixed forward velocity command")
    parser.add_argument("--vy", type=float, default=0.0, help="fixed lateral velocity command")
    parser.add_argument("--yaw", type=float, default=0.0, help="fixed yaw-rate command")
    parser.add_argument(
        "--rescue",
        action="store_true",
        help="approach the prone person and run the proximity/call gate",
    )
    parser.add_argument(
        "--autonomy",
        choices=("rescue", "carry", "scan"),
        help="Gemini ER 2 high-level planning with locally bounded MuJoCo execution",
    )
    parser.add_argument(
        "--gemini-model",
        default=GEMINI_ROBOTICS_MODEL,
        help="Gemini Robotics ER model code",
    )
    parser.add_argument(
        "--offline-plan",
        action="store_true",
        help="explicitly replace Gemini with deterministic lowest-risk selection",
    )
    parser.add_argument("--temperature-c", type=float, default=-18.0)
    parser.add_argument("--wind-mps", type=float, default=8.0)
    parser.add_argument("--visibility-m", type=float, default=900.0)
    parser.add_argument("--snow-depth-m", type=float, default=0.14)
    parser.add_argument("--terrain-friction", type=float, default=0.82)
    parser.add_argument(
        "--arm-live-call",
        action="store_true",
        help="allow one BeaconCall after proximity (also requires environment arming)",
    )
    parser.add_argument("--simulation-id", help="optional idempotent rescue run identifier")
    parser.add_argument(
        "--audit-log",
        type=Path,
        default=Path("runtime/everest-g1-events.jsonl"),
        help="redacted rescue event JSONL",
    )
    parser.add_argument("--no-policy", action="store_true", help="use stationary PD hold fallback")
    parser.add_argument("--no-auto-reset", action="store_true")
    parser.add_argument(
        "--bridge-db",
        type=Path,
        help="enable local SQLite telemetry and queued agent commands",
    )
    parser.add_argument(
        "--telemetry-hz",
        type=float,
        default=15.0,
        help="bounded SQLite telemetry rate (10-20 Hz)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print a machine-readable final summary"
    )
    return parser


def _viewer_sync_due(completed_steps: int, timestep: float) -> bool:
    """Throttle GUI copies while preserving every 500 Hz physics step."""

    current_frame = math.floor(completed_steps * timestep * VIEWER_HZ + 1e-12)
    previous_frame = math.floor((completed_steps - 1) * timestep * VIEWER_HZ + 1e-12)
    return current_frame > previous_frame


def _configure_viewer_camera(viewer) -> None:
    """Set a wide initial summit composition without disabling free-camera input."""

    with viewer.lock():
        viewer.cam.lookat[:] = VIEWER_CAMERA_LOOKAT
        viewer.cam.distance = VIEWER_CAMERA_DISTANCE
        viewer.cam.azimuth = VIEWER_CAMERA_AZIMUTH
        viewer.cam.elevation = VIEWER_CAMERA_ELEVATION


def _macos_viewer_command(argv: list[str]) -> str:
    return shlex.join(["uv", "run", "mjpython", "-m", "summit_sentinel", *argv])


def _apply_operator_safety(
    env: SummitSentinelEnv, operator: OperatorInput
) -> tuple[str | None, bool]:
    """Apply local safety events synchronously before the caller invokes step."""

    warning: str | None = None
    if operator.emergency_stop:
        reason = "joystick disconnected" if not operator.connected else "joystick button"
        env.emergency_stop(reason, authority="local")
    did_reset = False
    if operator.reset:
        env.reset(authority="local")
        did_reset = True
    if operator.resume and not operator.emergency_stop:
        try:
            env.resume(authority="local")
        except RuntimeError as error:
            warning = str(error)
    return warning, did_reset


def _structured_joystick_state(
    source, operator: OperatorInput, env: SummitSentinelEnv, profile_path: Path | None
) -> dict[str, object]:
    command = np.clip(np.asarray(operator.command, dtype=np.float64), -1.0, 1.0)
    return {
        "connected": bool(operator.connected and isinstance(source, PSJoystick)),
        "calibrated": bool(source.calibrated),
        "profile_name": profile_path.name if profile_path is not None else None,
        "device_name": source.device_name,
        "normalized_axes": {
            "vx": float(command[0]),
            "vy": float(command[1]),
            "yaw": float(command[2]),
        },
        "safety": {
            "reset": operator.reset,
            "emergency_stop": operator.emergency_stop,
            "resume": operator.resume,
            "quit": operator.quit,
            "stop_latched": env.emergency_stop_latched,
        },
    }


def _apply_bridge_failure(env: SummitSentinelEnv, failure: str | None) -> bool:
    """Convert a background I/O failure into a synchronous fault latch."""

    if failure is None:
        return False
    env.emergency_stop(f"bridge background failure: {failure}", authority="fault")
    return True


def _run(args: argparse.Namespace) -> dict[str, object]:
    config = load_config()
    calibration = (
        JoystickCalibration.load(args.joystick_calibration)
        if args.joystick_calibration is not None
        else None
    )
    source = (
        PSJoystick(config.joystick, args.joystick_index, calibration)
        if args.joystick
        else FixedCommandSource(np.asarray([args.vx, args.vy, args.yaw], dtype=np.float32))
    )
    try:
        return _run_with_source(args, config, source)
    finally:
        # Covers environment, policy, bridge, telemetry, viewer, and loop
        # initialization failures as well as normal/exceptional loop exit.
        source.close()


def _run_with_source(args, config, source) -> dict[str, object]:
    env = SummitSentinelEnv(
        config=config,
        seed=args.seed,
        use_policy=not args.no_policy,
        auto_reset=not args.no_auto_reset,
    )
    profile = EnvironmentProfile(
        temperature_c=args.temperature_c,
        wind_mps=args.wind_mps,
        visibility_m=args.visibility_m,
        snow_depth_m=args.snow_depth_m,
        friction=args.terrain_friction,
    )
    profile.validate()
    if args.autonomy is not None:
        env.apply_scenario_conditions(
            {
                "friction": profile.friction,
                "wind_mps": profile.wind_mps,
                "visibility_m": profile.visibility_m,
                "snow_depth_m": profile.snow_depth_m,
            }
        )
    initial_resets = env.reset_count
    physics_steps = max(0, round(args.seconds / config.simulation.timestep))
    falls = 0
    start = time.monotonic()
    bridge = SQLiteBridge(args.bridge_db) if args.bridge_db is not None else None
    if bridge is not None:
        bridge.begin_simulator_run()
    bridge_worker = BridgeRuntimeWorker(bridge) if bridge is not None else None
    command_applier = RuntimeCommandApplier() if bridge is not None else None
    telemetry = (
        RuntimeTelemetrySampler(hz=validate_telemetry_hz(args.telemetry_hz))
        if bridge is not None
        else None
    )
    safety_warning: str | None = None
    rescue = (
        MujocoRescueController(
            person_xy=env.rescue_target_xy,
            control_dt_s=config.simulation.timestep,
            arm_live_call=args.arm_live_call,
            simulation_id=args.simulation_id or "",
            audit_log=args.audit_log,
        )
        if args.rescue
        else None
    )
    autonomy = None
    if args.autonomy is not None:
        root = env.data.joint("floating_base_joint").qpos
        options = build_route_options(
            args.autonomy,
            profile,
            start_xy=(float(root[0]), float(root[1])),
        )
        planner = GeminiRoutePlanner(
            model=args.gemini_model,
            offline=args.offline_plan,
        )
        planning_frame = None if args.offline_plan else env.front_camera_jpeg()
        route = planner.select(args.autonomy, options, image_jpeg=planning_frame)
        autonomy = AutonomousMujocoController(
            env=env,
            mode=args.autonomy,
            route=route,
            planner=planner,
            arm_live_call=args.arm_live_call,
            simulation_id=args.simulation_id or "",
            audit_log=args.audit_log,
            planning_frame_bytes=len(planning_frame) if planning_frame is not None else 0,
        )

    def advance() -> tuple[object, bool]:
        nonlocal falls, safety_warning
        operator = source.sample()

        # This is the first operation after sampling. No simulator-thread queue
        # or storage handoff may intervene before a local stop or disconnect is
        # latched synchronously in the simulator process.
        latest_warning, did_local_reset = _apply_operator_safety(env, operator)
        if latest_warning is not None:
            safety_warning = latest_warning
        if operator.quit:
            return operator, False

        agent_command = None
        if bridge_worker is not None and command_applier is not None:
            if did_local_reset:
                rejected_ids = bridge_worker.discard_commands()
                bridge_worker.submit_outcome(command_applier.on_environment_reset(rejected_ids))
            else:
                commands = bridge_worker.take_commands()
                if commands:
                    agent_command, outcome = command_applier.consume(env, commands)
                    if outcome is not None:
                        bridge_worker.submit_outcome(outcome)
                else:
                    agent_command = command_applier.active_velocity()

            if _apply_bridge_failure(env, bridge_worker.failure):
                agent_command = None

        effective_command = operator.command
        if not args.joystick and agent_command is not None:
            effective_command = agent_command
        if rescue is not None and not env.emergency_stop_latched:
            effective_command = rescue.update(
                env.data.joint("floating_base_joint").qpos.copy(),
                image_supplier=env.front_camera_jpeg if args.arm_live_call else None,
            )
        if autonomy is not None and not env.emergency_stop_latched:
            effective_command = autonomy.update(env)
        result = env.step(effective_command)
        falls += int(result.fell)
        if result.reset and bridge_worker is not None and command_applier is not None:
            rejected_ids = bridge_worker.discard_commands()
            bridge_worker.submit_outcome(command_applier.on_environment_reset(rejected_ids))
        if telemetry is not None and bridge_worker is not None:
            frame = telemetry.maybe_build(env, result)
            if frame is not None:
                bridge_worker.publish_latest(
                    frame,
                    _structured_joystick_state(source, operator, env, args.joystick_calibration),
                )
        return result, True

    try:
        if bridge_worker is not None:
            bridge_worker.start()
        if args.mode == "viewer":
            import mujoco.viewer

            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                _configure_viewer_camera(viewer)
                for step_index in range(physics_steps):
                    if not viewer.is_running():
                        break
                    step_start = time.monotonic()
                    _, should_continue = advance()
                    if not should_continue:
                        break
                    if _viewer_sync_due(step_index + 1, config.simulation.timestep):
                        viewer.sync()
                    remaining = config.simulation.timestep - (time.monotonic() - step_start)
                    if remaining > 0:
                        time.sleep(remaining)
        else:
            for _ in range(physics_steps):
                step_start = time.monotonic()
                _, should_continue = advance()
                if not should_continue:
                    break
                if args.joystick or bridge is not None:
                    remaining = config.simulation.timestep - (time.monotonic() - step_start)
                    if remaining > 0:
                        time.sleep(remaining)
    finally:
        if rescue is not None:
            rescue.close()
        if autonomy is not None:
            autonomy.close()
        if bridge_worker is not None:
            rejected_ids = bridge_worker.discard_commands()
            if rejected_ids:
                bridge_worker.submit_outcome(
                    StorageOutcome(
                        rejected_ids=rejected_ids,
                        reject_message="simulator stopped before command application",
                    )
                )
            bridge_worker.close()

    root = env.data.joint("floating_base_joint").qpos
    return {
        "policy_mode": env.policy_mode,
        "policy_warning": env.policy_warning,
        "physics_hz": 1.0 / config.simulation.timestep,
        "policy_hz": env.policy_hz,
        "simulated_seconds": float(env.data.time),
        "wall_seconds": time.monotonic() - start,
        "falls": falls,
        "automatic_or_operator_resets": env.reset_count - initial_resets,
        "base_position": [float(value) for value in root[:3]],
        "seed": args.seed,
        "emergency_stop_latched": env.emergency_stop_latched,
        "emergency_stop_reason": env.emergency_stop_reason,
        "safety_warning": safety_warning,
        "telemetry_frames": bridge_worker.records_written if bridge_worker is not None else 0,
        "bridge_background_failure": bridge_worker.failure if bridge_worker is not None else None,
        "bridge_db": str(bridge.path) if bridge is not None else None,
        "rescue_enabled": rescue is not None or args.autonomy in {"rescue", "carry"},
        "rescue_reached": (
            rescue.latch.latched
            if rescue is not None
            else autonomy.rescue.latch.latched
            if autonomy is not None and autonomy.rescue is not None
            else False
        ),
        "rescue_distance_m": (
            rescue.last_command.surface_distance_m
            if rescue is not None
            else autonomy.rescue.last_command.surface_distance_m
            if autonomy is not None and autonomy.rescue is not None
            else None
        ),
        "live_call_armed": (
            rescue.live_call_armed
            if rescue is not None
            else autonomy.live_call_armed
            if autonomy is not None
            else False
        ),
        "call_submitted": (
            rescue.call_submitted
            if rescue is not None
            else autonomy.call_submitted
            if autonomy is not None
            else False
        ),
        "front_camera_status": (
            rescue.front_camera_status
            if rescue is not None
            else autonomy.rescue.front_camera_status
            if autonomy is not None and autonomy.rescue is not None
            else "planning-only"
            if autonomy is not None
            else "disabled"
        ),
        "front_camera_bytes": (
            rescue.front_camera_bytes
            if rescue is not None
            else autonomy.rescue.front_camera_bytes
            if autonomy is not None and autonomy.rescue is not None
            else 0
        ),
        "simulation_id": (
            rescue.simulation_id
            if rescue is not None
            else autonomy.rescue.simulation_id
            if autonomy is not None and autonomy.rescue is not None
            else None
        ),
        "autonomy_mode": args.autonomy,
        "autonomy_route": autonomy.route.route_id if autonomy is not None else None,
        "autonomy_planner": autonomy.planner.provider if autonomy is not None else None,
        "autonomy_complete": autonomy.completed if autonomy is not None else False,
        "autonomy_scan_cycles": autonomy.scan_cycles if autonomy is not None else 0,
        "carry_proxy_attached": (autonomy.carry_proxy_attached if autonomy is not None else False),
        "autonomy_live_call_armed": (autonomy.live_call_armed if autonomy is not None else False),
        "autonomy_call_submitted": (autonomy.call_submitted if autonomy is not None else False),
        "autonomy_planning_camera_bytes": (
            autonomy.planning_frame_bytes if autonomy is not None else 0
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parsed_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(parsed_argv)
    if args.seconds < 0:
        parser.error("--seconds must be non-negative")
    if not 10.0 <= args.telemetry_hz <= 20.0:
        parser.error("--telemetry-hz must be between 10 and 20")
    if args.rescue and args.autonomy is not None:
        parser.error("--rescue and --autonomy are mutually exclusive")
    if args.autonomy is not None and args.joystick:
        parser.error("--autonomy and --joystick are mutually exclusive")
    if args.offline_plan and args.autonomy is None:
        parser.error("--offline-plan requires --autonomy")
    if args.arm_live_call and not (args.rescue or args.autonomy in {"rescue", "carry"}):
        parser.error("--arm-live-call requires --rescue or rescue/carry autonomy")
    if args.list_joysticks:
        devices = list_joysticks()
        if not devices:
            print("No joysticks detected")
        for index, name in enumerate(devices):
            print(f"{index}: {name}")
        return 0
    if args.calibrate_joystick is not None:
        try:
            profile = capture_calibration(
                load_config().joystick,
                args.calibrate_joystick,
                device_index=args.joystick_index,
                sweep_seconds=args.calibration_seconds,
            )
        except (OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(f"Saved calibration for {profile.device_name}: {args.calibrate_joystick}")
        return 0
    if args.mode == "viewer" and sys.platform == "darwin" and "MJPYTHON_BIN" not in os.environ:
        parser.error(
            "viewer mode on macOS requires MuJoCo's mjpython launcher; rerun exactly: "
            f"{_macos_viewer_command(parsed_argv)}"
        )
    try:
        summary = _run(args)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        warning = f" ({summary['policy_warning']})" if summary["policy_warning"] else ""
        print(f"policy={summary['policy_mode']}{warning}")
        print(
            f"physics={summary['physics_hz']:.0f} Hz policy={summary['policy_hz']:.0f} Hz "
            f"simulated={summary['simulated_seconds']:.3f} s falls={summary['falls']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
