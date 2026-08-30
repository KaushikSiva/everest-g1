"""Local dependency-free checks for the rescue trigger and arming gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from everest_g1.beacon import BeaconCallWorker, BeaconSettings, new_simulation_id
from everest_g1.models import RescueObservation
from everest_g1.rescue import (
    ApproachLimits,
    ProximityLatch,
    approach_person,
    body_bearing,
    target_from_bearing,
)
from everest_g1.spatial_audio import RescueAudio, SpatialAudioSettings


def _dry_run(args: argparse.Namespace) -> int:
    simulation_id = args.simulation_id or new_simulation_id()
    limits = ApproachLimits()
    latch = ProximityLatch(limits.touch_distance_m, dwell_s=0.25)
    person_xy = (2.6, 0.75)
    robot_xy = [0.0, 0.0]
    dt_s = 0.02
    audio_settings = SpatialAudioSettings(
        acoustic_localization=args.acoustic_localization,
        render_cue=args.spatial_audio,
        output_path=args.spatial_audio_out,
    )
    audio = RescueAudio(audio_settings) if audio_settings.enabled else None
    command = None
    for _ in range(2_000):
        # The gate always uses observed geometry; audio only ever steers.
        command = approach_person(
            robot_xy=(robot_xy[0], robot_xy[1]),
            robot_yaw_rad=0.0,
            person_xy=person_xy,
            limits=limits,
            reached=latch.latched,
        )
        drive = command
        target_xy = person_xy
        if audio is not None:
            estimate = audio.observe(
                robot_xy=(robot_xy[0], robot_xy[1]), robot_yaw_rad=0.0, source_xy=person_xy
            )
            if estimate is not None and not command.reached:
                target_xy = target_from_bearing(
                    robot_xy=(robot_xy[0], robot_xy[1]),
                    robot_yaw_rad=0.0,
                    bearing_rad=estimate.bearing_rad,
                    surface_distance_m=command.surface_distance_m,
                    limits=limits,
                )
                drive = approach_person(
                    robot_xy=(robot_xy[0], robot_xy[1]),
                    robot_yaw_rad=0.0,
                    person_xy=target_xy,
                    limits=limits,
                )
            audio.cue(
                dt_s=dt_s,
                bearing_rad=body_bearing(
                    robot_xy=(robot_xy[0], robot_xy[1]),
                    robot_yaw_rad=0.0,
                    target_xy=target_xy,
                ),
                distance_m=command.surface_distance_m,
            )
        if latch.update(command.surface_distance_m, dt_s):
            if audio is not None:
                audio.mark_proximity_latched()
            break
        robot_xy[0] += drive.forward_mps * dt_s
        robot_xy[1] += drive.lateral_mps * dt_s
    assert command is not None
    result = {
        "simulation_id": simulation_id,
        "distance_m": round(command.surface_distance_m, 3),
        "reached": latch.latched,
        "live_call_armed": False,
    }
    if args.arm_live_call:
        worker = BeaconCallWorker(BeaconSettings.from_env(arm_requested=True))
        worker.submit_once(RescueObservation(simulation_id, command.surface_distance_m))
        worker.close(timeout_s=0.1)
        result["live_call_armed"] = True
        if audio is not None:
            audio.mark_call_submitted()
    if audio is not None:
        summary = audio.close(simulation_id)
        result["acoustic_localization"] = audio.sensor is not None
        if summary is not None:
            result["spatial_audio"] = summary
    print(json.dumps(result, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Everest G1 rescue simulation utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    dry_run = subparsers.add_parser("dry-run", help="exercise proximity without Isaac Lab")
    dry_run.add_argument("--simulation-id")
    dry_run.add_argument("--arm-live-call", action="store_true")
    dry_run.add_argument(
        "--spatial-audio",
        action="store_true",
        help="render the stereo operator cue for this approach to a WAV file",
    )
    dry_run.add_argument(
        "--spatial-audio-out",
        type=Path,
        default=Path("runtime/everest-g1-rescue.wav"),
        help="stereo cue destination; the simulation id is appended to the stem",
    )
    dry_run.add_argument(
        "--acoustic-localization",
        action="store_true",
        help="steer the approach from a simulated torso microphone array",
    )
    dry_run.set_defaults(handler=_dry_run)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)
