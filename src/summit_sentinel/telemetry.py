"""Bounded local telemetry sampling kept outside the MuJoCo control step."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from summit_sentinel.bridge import SQLiteBridge

if TYPE_CHECKING:
    from summit_sentinel.simulation import StepResult, SummitSentinelEnv

MIN_TELEMETRY_HZ = 10.0
MAX_TELEMETRY_HZ = 20.0


def validate_telemetry_hz(hz: float) -> float:
    value = float(hz)
    if not MIN_TELEMETRY_HZ <= value <= MAX_TELEMETRY_HZ:
        raise ValueError(
            f"telemetry/replay rate must be between {MIN_TELEMETRY_HZ:g} "
            f"and {MAX_TELEMETRY_HZ:g} Hz"
        )
    return value


def build_frame(
    env: SummitSentinelEnv,
    result: StepResult,
    *,
    run_id: str,
    sequence: int,
    recorded_at: float | None = None,
) -> dict[str, Any]:
    """Copy a compact, JSON-safe snapshot after ``env.step`` has returned."""

    root = env.data.joint("floating_base_joint")
    return {
        "recorded_at": time.time() if recorded_at is None else float(recorded_at),
        "sim_time": float(result.time),
        "run_id": run_id,
        "sequence": sequence,
        "physics_steps": env.physics_steps,
        "policy_mode": env.policy_mode,
        "control_mode": env.control_mode,
        "scenario_conditions": dict(env.scenario_conditions),
        "command": [float(value) for value in result.command],
        "base_position": [float(value) for value in root.qpos[:3]],
        "base_quaternion_wxyz": [float(value) for value in root.qpos[3:7]],
        "fell": result.fell,
        "reset": result.reset,
        "emergency_stop_latched": result.emergency_stop_latched,
        "emergency_stop_reason": env.emergency_stop_reason,
        "stop_authority": result.stop_authority,
        "locomotion_inhibited": result.locomotion_inhibited,
        "physics_advanced": result.physics_advanced,
    }


@dataclass
class TelemetryRecorder:
    """Publish at a wall-clock rate in the explicitly allowed 10-20 Hz band."""

    bridge: SQLiteBridge
    hz: float = 15.0
    run_id: str = field(default_factory=lambda: f"sim-{uuid.uuid4().hex[:12]}")
    _next_due: float | None = field(default=None, init=False)
    _sequence: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.hz = validate_telemetry_hz(self.hz)

    @property
    def records_written(self) -> int:
        return self._sequence

    def maybe_record(
        self,
        env: SummitSentinelEnv,
        result: StepResult,
        *,
        monotonic_now: float | None = None,
        recorded_at: float | None = None,
    ) -> bool:
        now = time.monotonic() if monotonic_now is None else float(monotonic_now)
        if self._next_due is not None and now + 1e-12 < self._next_due:
            return False
        frame = build_frame(
            env,
            result,
            run_id=self.run_id,
            sequence=self._sequence,
            recorded_at=recorded_at,
        )
        self.bridge.append_telemetry(frame)
        self._sequence += 1
        period = 1.0 / self.hz
        self._next_due = now + period
        return True


@dataclass
class RuntimeTelemetrySampler:
    """Build copied snapshots at a bounded rate; never performs storage I/O."""

    hz: float = 15.0
    run_id: str = field(default_factory=lambda: f"sim-{uuid.uuid4().hex[:12]}")
    _next_due: float | None = field(default=None, init=False)
    _sequence: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.hz = validate_telemetry_hz(self.hz)

    def maybe_build(
        self,
        env: SummitSentinelEnv,
        result: StepResult,
        *,
        monotonic_now: float | None = None,
        recorded_at: float | None = None,
    ) -> dict[str, Any] | None:
        now = time.monotonic() if monotonic_now is None else float(monotonic_now)
        if self._next_due is not None and now + 1e-12 < self._next_due:
            return None
        frame = build_frame(
            env,
            result,
            run_id=self.run_id,
            sequence=self._sequence,
            recorded_at=recorded_at,
        )
        self._sequence += 1
        self._next_due = now + 1.0 / self.hz
        return frame
