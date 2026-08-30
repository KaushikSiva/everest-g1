"""Local dependency-free checks for the rescue trigger and arming gate."""

from __future__ import annotations

import argparse
import json

from everest_g1.beacon import BeaconCallWorker, BeaconSettings, new_simulation_id
from everest_g1.models import RescueObservation
from everest_g1.rescue import ApproachLimits, ProximityLatch, approach_person


def _dry_run(args: argparse.Namespace) -> int:
    simulation_id = args.simulation_id or new_simulation_id()
    limits = ApproachLimits()
    latch = ProximityLatch(limits.touch_distance_m, dwell_s=0.25)
    person_xy = (2.6, 0.75)
    robot_xy = [0.0, 0.0]
    command = None
    for _ in range(2_000):
        command = approach_person(
            robot_xy=(robot_xy[0], robot_xy[1]),
            robot_yaw_rad=0.0,
            person_xy=person_xy,
            limits=limits,
            reached=latch.latched,
        )
        if latch.update(command.surface_distance_m, 0.02):
            break
        robot_xy[0] += command.forward_mps * 0.02
        robot_xy[1] += command.lateral_mps * 0.02
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
    print(json.dumps(result, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Everest G1 rescue simulation utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    dry_run = subparsers.add_parser("dry-run", help="exercise proximity without Isaac Lab")
    dry_run.add_argument("--simulation-id")
    dry_run.add_argument("--arm-live-call", action="store_true")
    dry_run.set_defaults(handler=_dry_run)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)
