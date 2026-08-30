"""Process-safe SQLite boundary between the simulator and asynchronous agents."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

COMMAND_KINDS = frozenset(
    {
        "velocity",
        "scenario",
        "scenario_conditions",
        "control_mode",
        "remote_stop",
        "reset",
        "resume",
    }
)
CONTROL_MODES = frozenset({"hold", "supervisory"})
SCENARIO_CONDITION_BOUNDS = {
    "friction": (0.2, 1.5),
    "wind_mps": (0.0, 30.0),
    "visibility_m": (10.0, 10_000.0),
    "snow_depth_m": (0.0, 0.5),
}
SCENARIO_VELOCITIES: dict[str, tuple[float, float, float]] = {
    "stand": (0.0, 0.0, 0.0),
    "walk_forward": (0.35, 0.0, 0.0),
    "sidestep_left": (0.0, 0.25, 0.0),
    "sidestep_right": (0.0, -0.25, 0.0),
    "turn_left": (0.0, 0.0, 0.35),
    "turn_right": (0.0, 0.0, -0.35),
}
MAX_COMMAND_BATCH = 50
MAX_REPLAY_ROWS = 1_000
MAX_TELEMETRY_BYTES = 64 * 1024
DEFAULT_MAX_PENDING_COMMANDS = 256
DEFAULT_MAX_COMMAND_HISTORY = 1_000
COMMAND_PRIORITIES = {
    "remote_stop": 100,
    "control_mode": 60,
    "reset": 40,
    "resume": 30,
    "scenario_conditions": 20,
    "velocity": 10,
    "scenario": 10,
}
COMMAND_TTL_SECONDS = {
    "remote_stop": 5.0,
    "reset": 2.0,
    "resume": 2.0,
    "control_mode": 2.0,
    "scenario_conditions": 5.0,
    "velocity": 2.0,
    "scenario": 5.0,
}
_SOURCE_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_APPROVAL_PATTERN = re.compile(r"^[A-Za-z0-9_.:/-]{1,128}$")


@dataclass(frozen=True)
class QueuedCommand:
    id: int
    kind: str
    payload: dict[str, Any]
    source: str
    created_at: float
    expires_at: float
    run_epoch: int


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _clamp(value: object, name: str, minimum: float, maximum: float) -> float:
    return min(max(_finite_number(value, name), minimum), maximum)


def _no_extra_keys(payload: Mapping[str, object], allowed: set[str]) -> None:
    extras = set(payload) - allowed
    if extras:
        raise ValueError(f"unexpected command fields: {', '.join(sorted(extras))}")


def _approval_ref(payload: Mapping[str, object]) -> str:
    value = payload.get("approval_ref")
    if not isinstance(value, str) or not _APPROVAL_PATTERN.fullmatch(value):
        raise ValueError("approval_ref must be a 1-128 character external audit reference")
    return value


def validate_command(kind: str, payload: Mapping[str, object]) -> dict[str, Any]:
    """Validate and clamp every command at the process boundary.

    Command values are high-level velocity/scenario requests. No accepted
    schema contains a joint, actuator, position target, or torque field.
    """

    if kind not in COMMAND_KINDS:
        raise ValueError(f"unsupported command kind: {kind!r}")
    if not isinstance(payload, Mapping):
        raise ValueError("command payload must be an object")
    if kind in {"remote_stop", "reset", "resume"}:
        _no_extra_keys(payload, {"approval_ref"})
        return {"approval_ref": _approval_ref(payload)}
    if kind == "control_mode":
        _no_extra_keys(payload, {"mode", "approval_ref"})
        mode = payload.get("mode")
        if not isinstance(mode, str) or mode not in CONTROL_MODES:
            raise ValueError(f"control mode must be one of: {', '.join(sorted(CONTROL_MODES))}")
        return {"mode": mode, "approval_ref": _approval_ref(payload)}
    if kind == "scenario_conditions":
        allowed = {*SCENARIO_CONDITION_BOUNDS, "approval_ref"}
        _no_extra_keys(payload, allowed)
        if missing := allowed - set(payload):
            raise ValueError(f"missing command fields: {', '.join(sorted(missing))}")
        result = {
            name: _clamp(payload[name], name, minimum, maximum)
            for name, (minimum, maximum) in SCENARIO_CONDITION_BOUNDS.items()
        }
        result["approval_ref"] = _approval_ref(payload)
        return result
    if kind == "velocity":
        _no_extra_keys(payload, {"vx", "vy", "yaw", "duration_s"})
        required = {"vx", "vy", "yaw", "duration_s"}
        if missing := required - set(payload):
            raise ValueError(f"missing command fields: {', '.join(sorted(missing))}")
        return {
            "vx": _clamp(payload["vx"], "vx", -1.0, 1.0),
            "vy": _clamp(payload["vy"], "vy", -1.0, 1.0),
            "yaw": _clamp(payload["yaw"], "yaw", -1.0, 1.0),
            "duration_s": _clamp(payload["duration_s"], "duration_s", 0.1, 10.0),
        }

    _no_extra_keys(payload, {"name", "speed_scale", "duration_s"})
    required = {"name", "speed_scale", "duration_s"}
    if missing := required - set(payload):
        raise ValueError(f"missing command fields: {', '.join(sorted(missing))}")
    name = payload["name"]
    if not isinstance(name, str) or name not in SCENARIO_VELOCITIES:
        allowed = ", ".join(sorted(SCENARIO_VELOCITIES))
        raise ValueError(f"scenario name must be one of: {allowed}")
    return {
        "name": name,
        "speed_scale": _clamp(payload["speed_scale"], "speed_scale", 0.0, 1.0),
        "duration_s": _clamp(payload["duration_s"], "duration_s", 0.5, 30.0),
    }


def scenario_velocity(payload: Mapping[str, object]) -> tuple[float, float, float]:
    validated = validate_command("scenario", payload)
    base = SCENARIO_VELOCITIES[validated["name"]]
    scale = validated["speed_scale"]
    return tuple(component * scale for component in base)


class SQLiteBridge:
    """Small SQLite WAL store safe to open from simulator and MCP processes."""

    def __init__(
        self,
        path: Path,
        *,
        max_telemetry_rows: int = 10_000,
        max_pending_commands: int = DEFAULT_MAX_PENDING_COMMANDS,
        max_command_history: int = DEFAULT_MAX_COMMAND_HISTORY,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if isinstance(max_telemetry_rows, bool) or not 1 <= max_telemetry_rows <= 100_000:
            raise ValueError("max_telemetry_rows must be between 1 and 100000")
        self.max_telemetry_rows = max_telemetry_rows
        if isinstance(max_pending_commands, bool) or not 1 <= max_pending_commands <= 10_000:
            raise ValueError("max_pending_commands must be between 1 and 10000")
        self.max_pending_commands = max_pending_commands
        if isinstance(max_command_history, bool) or not 0 <= max_command_history <= 100_000:
            raise ValueError("max_command_history must be between 0 and 100000")
        self.max_command_history = max_command_history
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # Runtime simulator I/O is isolated in a fail-safe worker. Keep lock
        # waits short so worker failure becomes visible promptly at 500 Hz.
        connection = sqlite3.connect(self.path, timeout=0.1, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 100")
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if str(mode).lower() != "wal":
                raise RuntimeError(f"SQLite refused WAL mode for {self.path}")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at REAL NOT NULL,
                    sim_time REAL NOT NULL,
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS telemetry_recent_idx ON telemetry(id DESC);

                CREATE TABLE IF NOT EXISTS commands (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    created_at REAL NOT NULL,
                    claimed_at REAL,
                    completed_at REAL,
                    message TEXT,
                    priority INTEGER NOT NULL DEFAULT 0,
                    expires_at REAL,
                    run_epoch INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS bridge_meta (
                    key TEXT PRIMARY KEY,
                    value INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO bridge_meta(key, value) VALUES ('run_epoch', 0);
                CREATE TABLE IF NOT EXISTS joystick_state (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    updated_at REAL NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(commands)").fetchall()
            }
            migrations = {
                "priority": "ALTER TABLE commands ADD COLUMN priority INTEGER NOT NULL DEFAULT 0",
                "expires_at": "ALTER TABLE commands ADD COLUMN expires_at REAL",
                "run_epoch": "ALTER TABLE commands ADD COLUMN run_epoch INTEGER NOT NULL DEFAULT 0",
            }
            for column, statement in migrations.items():
                if column not in columns:
                    connection.execute(statement)
            connection.execute(
                "UPDATE commands SET kind = 'remote_stop' "
                "WHERE kind = 'emergency_stop' AND status = 'queued'"
            )
            connection.execute(
                "UPDATE commands SET priority = CASE kind "
                "WHEN 'remote_stop' THEN 100 WHEN 'control_mode' THEN 60 "
                "WHEN 'reset' THEN 40 WHEN 'resume' THEN 30 "
                "WHEN 'scenario_conditions' THEN 20 ELSE 10 END WHERE priority = 0"
            )
            connection.execute(
                "UPDATE commands SET expires_at = created_at + 2.0 WHERE expires_at IS NULL"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS commands_queue_idx "
                "ON commands(status, priority DESC, id)"
            )

    def health(self) -> dict[str, object]:
        with self._connect() as connection:
            connection.execute("SELECT 1").fetchone()
            mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        return {"status": "ok", "database": "ready", "journal_mode": mode}

    def append_telemetry(self, frame: Mapping[str, object]) -> int:
        required = {"recorded_at", "sim_time", "run_id", "sequence"}
        if missing := required - set(frame):
            raise ValueError(f"telemetry missing fields: {', '.join(sorted(missing))}")
        recorded_at = _finite_number(frame["recorded_at"], "recorded_at")
        sim_time = _finite_number(frame["sim_time"], "sim_time")
        run_id = frame["run_id"]
        sequence = frame["sequence"]
        if not isinstance(run_id, str) or not _SOURCE_PATTERN.fullmatch(run_id):
            raise ValueError("run_id must contain 1-64 safe identifier characters")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        try:
            encoded = json.dumps(dict(frame), allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"telemetry must be finite JSON: {error}") from error
        if len(encoded.encode("utf-8")) > MAX_TELEMETRY_BYTES:
            raise ValueError("telemetry frame exceeds 64 KiB")

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                INSERT INTO telemetry(recorded_at, sim_time, run_id, sequence, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (recorded_at, sim_time, run_id, sequence, encoded),
            )
            connection.execute(
                """
                DELETE FROM telemetry
                WHERE id <= COALESCE(
                    (SELECT id FROM telemetry ORDER BY id DESC LIMIT 1 OFFSET ?), 0
                )
                """,
                (self.max_telemetry_rows,),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def recent_telemetry(self, limit: int = 100) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not 1 <= limit <= MAX_REPLAY_ROWS:
            raise ValueError(f"telemetry limit must be between 1 and {MAX_REPLAY_ROWS}")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM telemetry ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in reversed(rows)]

    def run_telemetry(self, limit: int = 100, *, run_id: str | None = None) -> dict[str, Any]:
        if isinstance(limit, bool) or not 1 <= limit <= MAX_REPLAY_ROWS:
            raise ValueError(f"telemetry limit must be between 1 and {MAX_REPLAY_ROWS}")
        if run_id is not None and (
            not isinstance(run_id, str) or not _SOURCE_PATTERN.fullmatch(run_id)
        ):
            raise ValueError("run_id must contain 1-64 safe identifier characters")
        with self._connect() as connection:
            selected_run = run_id
            if selected_run is None:
                row = connection.execute(
                    "SELECT run_id FROM telemetry ORDER BY id DESC LIMIT 1"
                ).fetchone()
                selected_run = str(row["run_id"]) if row is not None else None
            rows = (
                connection.execute(
                    """
                    SELECT payload_json FROM telemetry
                    WHERE run_id = ? ORDER BY id DESC LIMIT ?
                    """,
                    (selected_run, limit),
                ).fetchall()
                if selected_run is not None
                else []
            )
        frames = [json.loads(row["payload_json"]) for row in reversed(rows)]
        return {
            "run_id": selected_run,
            "frame_count": len(frames),
            "truncated": len(frames) == limit,
            "frames": frames,
            "mission_metrics": self._mission_metrics(frames),
        }

    @staticmethod
    def _mission_metrics(frames: list[dict[str, Any]]) -> dict[str, object]:
        distance = 0.0
        positions = [frame.get("base_position") for frame in frames]
        valid_positions = [
            position
            for position in positions
            if isinstance(position, list)
            and len(position) == 3
            and all(isinstance(value, int | float) and math.isfinite(value) for value in position)
        ]
        for previous, current in pairwise(valid_positions):
            distance += math.hypot(current[0] - previous[0], current[1] - previous[1])
        sim_times = [
            float(frame["sim_time"])
            for frame in frames
            if isinstance(frame.get("sim_time"), int | float)
            and math.isfinite(float(frame["sim_time"]))
        ]
        return {
            "simulated_duration_s": max(sim_times) - min(sim_times) if sim_times else 0.0,
            "horizontal_distance_m": distance,
            "falls": sum(bool(frame.get("fell")) for frame in frames),
            "resets": sum(bool(frame.get("reset")) for frame in frames),
            "stopped_frames": sum(bool(frame.get("emergency_stop_latched")) for frame in frames),
            "physics_steps_observed": sum(bool(frame.get("physics_advanced")) for frame in frames),
        }

    def update_joystick_state(self, state: Mapping[str, object]) -> None:
        required = {
            "connected",
            "calibrated",
            "profile_name",
            "device_name",
            "normalized_axes",
            "safety",
        }
        _no_extra_keys(state, required)
        if missing := required - set(state):
            raise ValueError(f"joystick state missing fields: {', '.join(sorted(missing))}")
        for key in ("connected", "calibrated"):
            if not isinstance(state[key], bool):
                raise ValueError(f"joystick {key} must be boolean")
        for key in ("profile_name", "device_name"):
            value = state[key]
            if value is not None and (
                not isinstance(value, str) or not 1 <= len(value) <= 128 or "/" in value
            ):
                raise ValueError(f"joystick {key} must be a bounded basename or null")
        axes = state["normalized_axes"]
        if not isinstance(axes, Mapping) or set(axes) != {"vx", "vy", "yaw"}:
            raise ValueError("normalized_axes must contain exactly vx, vy, and yaw")
        normalized_axes = {name: _clamp(value, name, -1.0, 1.0) for name, value in axes.items()}
        safety = state["safety"]
        safety_keys = {"reset", "emergency_stop", "resume", "quit", "stop_latched"}
        if (
            not isinstance(safety, Mapping)
            or set(safety) != safety_keys
            or not all(isinstance(value, bool) for value in safety.values())
        ):
            raise ValueError("joystick safety state has invalid fields")
        payload = {
            "connected": state["connected"],
            "calibrated": state["calibrated"],
            "profile_name": state["profile_name"],
            "device_name": state["device_name"],
            "normalized_axes": normalized_axes,
            "safety": dict(safety),
        }
        updated_at = time.time()
        payload["updated_at"] = updated_at
        encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO joystick_state(singleton, updated_at, payload_json)
                VALUES (1, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    payload_json = excluded.payload_json
                """,
                (updated_at, encoded),
            )

    def joystick_state(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM joystick_state WHERE singleton = 1"
            ).fetchone()
        return (
            json.loads(row["payload_json"])
            if row is not None
            else {"available": False, "connected": False}
        )

    def enqueue_command(
        self,
        kind: str,
        payload: Mapping[str, object],
        *,
        source: str = "mcp",
        ttl_s: float | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        if not isinstance(source, str) or not _SOURCE_PATTERN.fullmatch(source):
            raise ValueError("source must contain 1-64 safe identifier characters")
        normalized = validate_command(kind, payload)
        created_at = time.time() if now is None else _finite_number(now, "now")
        ttl = _clamp(
            COMMAND_TTL_SECONDS[kind] if ttl_s is None else ttl_s,
            "ttl_s",
            0.1,
            30.0,
        )
        expires_at = created_at + ttl
        encoded = json.dumps(normalized, allow_nan=False, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run_epoch = self._current_run_epoch(connection)
            self._expire_commands(connection, created_at, run_epoch)

            coalesce_kinds = (
                ("velocity", "scenario") if kind in {"velocity", "scenario"} else (kind,)
            )
            replaceable_kinds = {"velocity", "scenario", "scenario_conditions", "control_mode"}
            placeholders = ",".join("?" for _ in coalesce_kinds)
            existing = connection.execute(
                f"""
                SELECT id, kind, payload_json, source, expires_at, run_epoch
                FROM commands
                WHERE status = 'queued' AND run_epoch = ? AND kind IN ({placeholders})
                ORDER BY priority DESC, id DESC LIMIT 1
                """,
                (run_epoch, *coalesce_kinds),
            ).fetchone()
            if existing is not None and kind not in replaceable_kinds:
                self._prune_command_history(connection)
                connection.commit()
                return {
                    "command_id": int(existing["id"]),
                    "kind": str(existing["kind"]),
                    "payload": json.loads(existing["payload_json"]),
                    "status": "queued",
                    "source": str(existing["source"]),
                    "expires_at": float(existing["expires_at"]),
                    "run_epoch": int(existing["run_epoch"]),
                    "coalesced": True,
                }
            if existing is not None:
                connection.execute(
                    f"""
                    UPDATE commands SET status = 'superseded', completed_at = ?,
                                        message = 'newer motion command coalesced'
                    WHERE status = 'queued' AND run_epoch = ? AND kind IN ({placeholders})
                    """,
                    (created_at, run_epoch, *coalesce_kinds),
                )

            pending_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM commands WHERE status IN ('queued', 'claimed')"
                ).fetchone()[0]
            )
            if pending_count >= self.max_pending_commands:
                if kind != "remote_stop":
                    connection.rollback()
                    raise RuntimeError("command backlog is full; request rejected")
                preempted = connection.execute(
                    """
                    SELECT id FROM commands WHERE status = 'queued'
                    ORDER BY priority ASC, id ASC LIMIT 1
                    """
                ).fetchone()
                if preempted is None:
                    connection.rollback()
                    raise RuntimeError("command backlog is full; stop request could not be queued")
                connection.execute(
                    """
                    UPDATE commands SET status = 'preempted', completed_at = ?,
                                        message = 'preempted by supervisory stop'
                    WHERE id = ? AND status = 'queued'
                    """,
                    (created_at, preempted["id"]),
                )

            cursor = connection.execute(
                """
                INSERT INTO commands(
                    kind, payload_json, source, status, created_at,
                    priority, expires_at, run_epoch
                )
                VALUES (?, ?, ?, 'queued', ?, ?, ?, ?)
                """,
                (
                    kind,
                    encoded,
                    source,
                    created_at,
                    COMMAND_PRIORITIES[kind],
                    expires_at,
                    run_epoch,
                ),
            )
            self._prune_command_history(connection)
            connection.commit()
        return {
            "command_id": int(cursor.lastrowid),
            "kind": kind,
            "payload": normalized,
            "status": "queued",
            "source": source,
            "expires_at": expires_at,
            "run_epoch": run_epoch,
            "coalesced": False,
        }

    @staticmethod
    def _current_run_epoch(connection: sqlite3.Connection) -> int:
        return int(
            connection.execute("SELECT value FROM bridge_meta WHERE key = 'run_epoch'").fetchone()[
                0
            ]
        )

    @staticmethod
    def _expire_commands(connection: sqlite3.Connection, now: float, run_epoch: int) -> None:
        connection.execute(
            """
            UPDATE commands SET status = 'stale', completed_at = ?,
                                message = 'invalidated by run reset'
            WHERE status = 'queued' AND run_epoch != ?
            """,
            (now, run_epoch),
        )
        connection.execute(
            """
            UPDATE commands SET status = 'expired', completed_at = ?,
                                message = 'command TTL expired'
            WHERE status = 'queued' AND expires_at <= ?
            """,
            (now, now),
        )

    def _prune_command_history(self, connection: sqlite3.Connection) -> None:
        terminal = (
            "applied",
            "rejected",
            "expired",
            "stale",
            "superseded",
            "preempted",
        )
        placeholders = ",".join("?" for _ in terminal)
        if self.max_command_history == 0:
            connection.execute(f"DELETE FROM commands WHERE status IN ({placeholders})", terminal)
            return
        connection.execute(
            f"""
            DELETE FROM commands
            WHERE status IN ({placeholders})
              AND id NOT IN (
                  SELECT id FROM commands
                  WHERE status IN ({placeholders})
                  ORDER BY id DESC LIMIT ?
              )
            """,
            (*terminal, *terminal, self.max_command_history),
        )

    def current_run_epoch(self) -> int:
        with self._connect() as connection:
            return self._current_run_epoch(connection)

    def begin_simulator_run(self) -> int:
        """Advance the durable epoch before consuming any persisted command."""

        return self.advance_run_epoch("simulator restart invalidated prior commands")

    def claim_commands(self, limit: int = 20, *, now: float | None = None) -> list[QueuedCommand]:
        if isinstance(limit, bool) or not 1 <= limit <= MAX_COMMAND_BATCH:
            raise ValueError(f"command batch must be between 1 and {MAX_COMMAND_BATCH}")
        claimed_at = time.time() if now is None else _finite_number(now, "now")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run_epoch = self._current_run_epoch(connection)
            self._expire_commands(connection, claimed_at, run_epoch)
            rows = connection.execute(
                """
                SELECT id, kind, payload_json, source, created_at, expires_at, run_epoch
                FROM commands
                WHERE status = 'queued' AND run_epoch = ? AND expires_at > ?
                ORDER BY priority DESC, id ASC LIMIT ?
                """,
                (run_epoch, claimed_at, limit),
            ).fetchall()
            if rows:
                connection.executemany(
                    "UPDATE commands SET status = 'claimed', claimed_at = ? "
                    "WHERE id = ? AND status = 'queued'",
                    ((claimed_at, row["id"]) for row in rows),
                )
            self._prune_command_history(connection)
            connection.commit()
        return [
            QueuedCommand(
                id=int(row["id"]),
                kind=str(row["kind"]),
                payload=json.loads(row["payload_json"]),
                source=str(row["source"]),
                created_at=float(row["created_at"]),
                expires_at=float(row["expires_at"]),
                run_epoch=int(row["run_epoch"]),
            )
            for row in rows
        ]

    def reject_claimed(self, command_ids: list[int], message: str) -> int:
        if not command_ids:
            return 0
        if len(command_ids) > MAX_COMMAND_BATCH or any(
            isinstance(command_id, bool) or not isinstance(command_id, int) or command_id < 1
            for command_id in command_ids
        ):
            raise ValueError("claimed command ids must be positive integers in one bounded batch")
        placeholders = ",".join("?" for _ in command_ids)
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE commands SET status = 'rejected', completed_at = ?, message = ?
                WHERE status = 'claimed' AND id IN ({placeholders})
                """,
                (time.time(), message[:500], *command_ids),
            )
            self._prune_command_history(connection)
        return cursor.rowcount

    def advance_run_epoch(self, message: str = "simulation reset") -> int:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            next_epoch = self._current_run_epoch(connection) + 1
            connection.execute(
                "UPDATE bridge_meta SET value = ? WHERE key = 'run_epoch'", (next_epoch,)
            )
            connection.execute(
                """
                UPDATE commands SET status = 'stale', completed_at = ?, message = ?
                WHERE status IN ('queued', 'claimed') AND run_epoch != ?
                """,
                (now, message[:500], next_epoch),
            )
            self._prune_command_history(connection)
            connection.commit()
        return next_epoch

    def complete_command(self, command_id: int, *, applied: bool, message: str = "") -> bool:
        if isinstance(command_id, bool) or not isinstance(command_id, int) or command_id < 1:
            raise ValueError("command_id must be a positive integer")
        status = "applied" if applied else "rejected"
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE commands SET status = ?, completed_at = ?, message = ?
                WHERE id = ? AND status = 'claimed'
                """,
                (status, time.time(), message[:500], command_id),
            )
            self._prune_command_history(connection)
        return cursor.rowcount == 1

    def command_status(self, command_id: int) -> dict[str, Any] | None:
        if isinstance(command_id, bool) or not isinstance(command_id, int) or command_id < 1:
            raise ValueError("command_id must be a positive integer")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, kind, payload_json, source, status, created_at,
                       claimed_at, completed_at, message, priority, expires_at, run_epoch
                FROM commands WHERE id = ?
                """,
                (command_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        return result

    def status(self) -> dict[str, Any]:
        with self._connect() as connection:
            counts = {
                str(row["status"]): int(row["count"])
                for row in connection.execute(
                    "SELECT status, COUNT(*) AS count FROM commands GROUP BY status"
                )
            }
            row = connection.execute(
                "SELECT payload_json FROM telemetry ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return {
            "health": self.health(),
            "run_epoch": self.current_run_epoch(),
            "max_pending_commands": self.max_pending_commands,
            "max_command_history": self.max_command_history,
            "max_command_rows": self.max_pending_commands + self.max_command_history,
            "total_commands": sum(counts.values()),
            "command_counts": counts,
            "latest_telemetry": json.loads(row["payload_json"]) if row is not None else None,
        }
