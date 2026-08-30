"""Apply queued high-level commands outside the physics/control loop."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from summit_sentinel.bridge import SQLiteBridge, scenario_velocity, validate_command
from summit_sentinel.simulation import SummitSentinelEnv


@dataclass(frozen=True)
class StorageOutcome:
    """One bounded, ordered storage update produced by the simulator thread."""

    completions: tuple[tuple[int, bool, str], ...] = ()
    rejected_ids: tuple[int, ...] = ()
    reject_message: str = ""
    advance_epoch_reason: str | None = None


class RuntimeCommandApplier:
    """Apply already-copied commands without performing storage or network I/O."""

    def __init__(self) -> None:
        self._active_velocity: np.ndarray | None = None
        self._active_until = 0.0
        self._active_epoch: int | None = None

    def _clear_active_velocity(self) -> None:
        self._active_velocity = None
        self._active_epoch = None
        self._active_until = 0.0

    def on_environment_reset(self, rejected_ids: tuple[int, ...] = ()) -> StorageOutcome:
        self._clear_active_velocity()
        return StorageOutcome(
            rejected_ids=rejected_ids,
            reject_message="environment reset invalidated prior commands",
            advance_epoch_reason="environment reset invalidated prior commands",
        )

    def consume(
        self,
        env: SummitSentinelEnv,
        commands: list[Any],
        *,
        now: float | None = None,
    ) -> tuple[np.ndarray | None, StorageOutcome | None]:
        """Consume one claimed batch and return a nonblocking storage receipt."""

        current = time.monotonic() if now is None else float(now)
        completions: list[tuple[int, bool, str]] = []
        rejected_ids: tuple[int, ...] = ()
        reject_message = ""
        advance_reason: str | None = None
        for index, command in enumerate(commands):
            try:
                message, stop_batch, reset_epoch = self._apply(
                    env,
                    command.kind,
                    command.payload,
                    current,
                    command.run_epoch,
                )
            except (RuntimeError, ValueError) as error:
                completions.append((command.id, False, str(error)))
                if command.kind == "remote_stop":
                    rejected_ids = tuple(item.id for item in commands[index + 1 :])
                    reject_message = "batch stopped after supervisory stop request"
                    break
            else:
                completions.append((command.id, True, message))
                if reset_epoch:
                    advance_reason = "remote simulation reset"
                    self._clear_active_velocity()
                if stop_batch or reset_epoch:
                    rejected_ids = tuple(item.id for item in commands[index + 1 :])
                    reject_message = "batch processing stopped by higher-priority safety command"
                    break
        outcome = StorageOutcome(
            completions=tuple(completions),
            rejected_ids=rejected_ids,
            reject_message=reject_message,
            advance_epoch_reason=advance_reason,
        )
        return self.active_velocity(now=current), outcome

    def active_velocity(self, *, now: float | None = None) -> np.ndarray | None:
        current = time.monotonic() if now is None else float(now)
        if self._active_velocity is not None and current < self._active_until:
            return self._active_velocity.copy()
        self._clear_active_velocity()
        return None

    def _apply(
        self,
        env: SummitSentinelEnv,
        kind: str,
        payload: dict[str, object],
        now: float,
        run_epoch: int,
    ) -> tuple[str, bool, bool]:
        payload = validate_command(kind, payload)
        if kind == "remote_stop":
            if not env.request_supervisory_stop("remote supervisory queue"):
                raise RuntimeError("local or fault stop already owns the latch")
            self._clear_active_velocity()
            return "best-effort supervisory stop latched", True, False
        if kind == "scenario_conditions":
            env.apply_scenario_conditions(
                {
                    name: float(payload[name])
                    for name in ("friction", "wind_mps", "visibility_m", "snow_depth_m")
                }
            )
            return "validated scenario conditions applied", False, False
        if kind == "control_mode":
            mode = str(payload["mode"])
            env.set_control_mode(mode)
            if mode == "hold":
                self._clear_active_velocity()
                return "hold mode applied; active motion cleared", True, False
            return "supervisory mode applied without changing stop latch", False, False
        if kind == "reset":
            env.reset(authority="remote")
            self._clear_active_velocity()
            return "remote state reset; local acknowledgement unchanged", False, True
        if kind == "resume":
            if not env.resume(authority="remote"):
                raise RuntimeError("simulation is not supervisory-stopped")
            return "supervisory stop resumed after remote reset", False, False
        if kind == "velocity":
            velocity = np.asarray([payload["vx"], payload["vy"], payload["yaw"]], dtype=np.float32)
            duration = float(payload["duration_s"])
        elif kind == "scenario":
            velocity = np.asarray(scenario_velocity(payload), dtype=np.float32)
            duration = float(payload["duration_s"])
        else:
            raise ValueError(f"unsupported queued command kind: {kind!r}")
        self._active_velocity = velocity
        self._active_until = now + duration
        self._active_epoch = run_epoch
        return f"bounded velocity active for {duration:g} seconds", False, False


class BridgeRuntimeWorker:
    """Own every SQLite operation used while the 500 Hz simulator is running.

    The main thread hands over immutable copies through bounded, nonblocking
    queues. This worker deliberately has no environment or MuJoCo references.
    """

    def __init__(
        self,
        bridge: SQLiteBridge,
        *,
        poll_hz: float = 20.0,
        operation_capacity: int = 32,
    ) -> None:
        if not 1.0 <= poll_hz <= 20.0:
            raise ValueError("command poll rate must be between 1 and 20 Hz")
        if not 1 <= operation_capacity <= 256:
            raise ValueError("operation queue capacity must be between 1 and 256")
        self._bridge = bridge
        self._poll_period = 1.0 / poll_hz
        self._operations: queue.Queue[StorageOutcome] = queue.Queue(operation_capacity)
        self._commands: queue.Queue[list[Any]] = queue.Queue(1)
        self._snapshot_lock = threading.Lock()
        self._snapshot: tuple[int, dict[str, Any], dict[str, object]] | None = None
        self._snapshot_version = 0
        self._records_written = 0
        self._failure_lock = threading.Lock()
        self._failure: str | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="summit-sentinel-sqlite",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    @property
    def failure(self) -> str | None:
        with self._failure_lock:
            return self._failure

    @property
    def records_written(self) -> int:
        with self._snapshot_lock:
            return self._records_written

    def take_commands(self) -> list[Any]:
        try:
            return self._commands.get_nowait()
        except queue.Empty:
            return []

    def discard_commands(self) -> tuple[int, ...]:
        return tuple(command.id for command in self.take_commands())

    def submit_outcome(self, outcome: StorageOutcome) -> bool:
        if self.failure is not None:
            return False
        try:
            self._operations.put_nowait(outcome)
        except queue.Full:
            self._set_failure("bounded bridge operation queue overflow")
            return False
        self._wake.set()
        return True

    def publish_latest(
        self,
        frame: dict[str, Any],
        joystick_state: dict[str, object],
    ) -> None:
        """Replace any unwritten telemetry/state pair without blocking physics."""

        with self._snapshot_lock:
            self._snapshot_version += 1
            self._snapshot = (
                self._snapshot_version,
                dict(frame),
                dict(joystick_state),
            )
        self._wake.set()

    def close(self, *, timeout: float = 1.0) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=max(0.0, timeout))
        if self._thread.is_alive():
            self._set_failure("bridge worker did not stop within shutdown deadline")

    def _set_failure(self, message: str) -> None:
        with self._failure_lock:
            if self._failure is None:
                self._failure = message[:300]

    def _process_outcome(self, outcome: StorageOutcome) -> None:
        for command_id, applied, message in outcome.completions:
            self._bridge.complete_command(command_id, applied=applied, message=message)
        if outcome.rejected_ids:
            self._bridge.reject_claimed(list(outcome.rejected_ids), outcome.reject_message)
        if outcome.advance_epoch_reason is not None:
            self._bridge.advance_run_epoch(outcome.advance_epoch_reason)

    def _write_latest_snapshot(self) -> None:
        with self._snapshot_lock:
            snapshot = self._snapshot
        if snapshot is None:
            return
        version, frame, joystick_state = snapshot
        self._bridge.append_telemetry(frame)
        self._bridge.update_joystick_state(joystick_state)
        with self._snapshot_lock:
            self._records_written += 1
            if self._snapshot is not None and self._snapshot[0] == version:
                self._snapshot = None

    def _has_pending_snapshot(self) -> bool:
        with self._snapshot_lock:
            return self._snapshot is not None

    def _run(self) -> None:
        next_poll = time.monotonic()
        batch_inflight = False
        try:
            while (
                not self._stop.is_set()
                or not self._operations.empty()
                or self._has_pending_snapshot()
            ):
                while True:
                    try:
                        outcome = self._operations.get_nowait()
                    except queue.Empty:
                        break
                    self._process_outcome(outcome)
                    batch_inflight = False
                self._write_latest_snapshot()
                now = time.monotonic()
                if not batch_inflight and now + 1e-12 >= next_poll:
                    commands = self._bridge.claim_commands()
                    next_poll = now + self._poll_period
                    if commands:
                        self._commands.put_nowait(commands)
                        batch_inflight = True
                self._wake.wait(timeout=min(0.01, max(0.0, next_poll - time.monotonic())))
                self._wake.clear()
        except Exception as error:  # fail-safe status is observed by the simulator thread
            self._set_failure(f"{type(error).__name__}: {error}")
