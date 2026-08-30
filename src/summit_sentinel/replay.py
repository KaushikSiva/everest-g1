"""Replay bounded local telemetry as timestamped JSON lines."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from summit_sentinel.bridge import MAX_REPLAY_ROWS, SQLiteBridge
from summit_sentinel.telemetry import validate_telemetry_hz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-db", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--hz", type=float, default=15.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= MAX_REPLAY_ROWS:
        parser.error(f"--limit must be between 1 and {MAX_REPLAY_ROWS}")
    try:
        hz = validate_telemetry_hz(args.hz)
        frames = SQLiteBridge(args.bridge_db).recent_telemetry(args.limit)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    period = 1.0 / hz
    for index, frame in enumerate(frames):
        if index:
            time.sleep(period)
        print(json.dumps(frame, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
