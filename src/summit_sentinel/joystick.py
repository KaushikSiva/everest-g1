"""PlayStation-style joystick command input with fail-safe zero commands."""

from __future__ import annotations

import importlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from summit_sentinel.config import JoystickConfig

# pygame prints a greeting on import unless this is set. Keep CLI --json output
# machine-readable even when joystick support is enabled.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


@dataclass(frozen=True)
class OperatorInput:
    command: np.ndarray
    reset: bool = False
    emergency_stop: bool = False
    resume: bool = False
    quit: bool = False
    connected: bool = True


@dataclass(frozen=True)
class AxisCalibration:
    """Observed raw range and resting center for one SDL joystick axis."""

    minimum: float = -1.0
    center: float = 0.0
    maximum: float = 1.0

    def __post_init__(self) -> None:
        values = np.asarray([self.minimum, self.center, self.maximum], dtype=np.float64)
        if not np.all(np.isfinite(values)) or not self.minimum < self.center < self.maximum:
            raise ValueError("axis calibration requires finite minimum < center < maximum")

    def normalize(self, value: float, deadzone: float) -> float:
        denominator = (
            self.maximum - self.center if value >= self.center else self.center - self.minimum
        )
        centered = (value - self.center) / denominator
        return apply_deadzone(float(np.clip(centered, -1.0, 1.0)), deadzone)


@dataclass(frozen=True)
class JoystickCalibration:
    """Portable JSON calibration profile for the axes used as velocity commands."""

    device_name: str
    axes: dict[int, AxisCalibration]
    schema_version: int = 1

    @classmethod
    def load(cls, path: Path) -> JoystickCalibration:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("schema_version") != 1:
                raise ValueError("unsupported joystick calibration schema")
            axes = {
                int(index): AxisCalibration(
                    minimum=float(values["minimum"]),
                    center=float(values["center"]),
                    maximum=float(values["maximum"]),
                )
                for index, values in raw["axes"].items()
            }
            return cls(device_name=str(raw["device_name"]), axes=axes)
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid joystick calibration file {path}: {error}") from error

    def save(self, path: Path) -> None:
        payload = {
            "schema_version": self.schema_version,
            "device_name": self.device_name,
            "axes": {
                str(index): {
                    "minimum": calibration.minimum,
                    "center": calibration.center,
                    "maximum": calibration.maximum,
                }
                for index, calibration in sorted(self.axes.items())
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def apply_deadzone(value: float, deadzone: float) -> float:
    """Remove center drift while preserving the remaining stick range."""

    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0
    scaled = (magnitude - deadzone) / (1.0 - deadzone)
    return float(np.copysign(scaled, value))


class FixedCommandSource:
    def __init__(self, command: np.ndarray) -> None:
        self._command = np.asarray(command, dtype=np.float32)

    def sample(self) -> OperatorInput:
        return OperatorInput(command=self._command.copy())

    @property
    def device_name(self) -> str:
        return "fixed-command"

    @property
    def calibrated(self) -> bool:
        return False

    def close(self) -> None:
        return None


class PSJoystick:
    """Read normalized velocity commands from a pygame-supported PS controller.

    Sticks create velocity commands only. They never bypass the locomotion
    policy or write torque controls directly.
    """

    def __init__(
        self,
        config: JoystickConfig,
        device_index: int = 0,
        calibration: JoystickCalibration | None = None,
        reconnect_interval_s: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        import pygame

        self._pygame = pygame
        self._controller_api = importlib.import_module("pygame._sdl2.controller_old")
        self._config = config
        self._device = None
        self._controller = None
        self._connected = False
        self._reset_down = False
        self._resume_down = False
        self._instance_id = None
        self._calibration = calibration
        self._device_index = device_index
        self._expected_name: str | None = None
        if not 0.1 <= reconnect_interval_s <= 10.0:
            raise ValueError("reconnect interval must be between 0.1 and 10 seconds")
        self._reconnect_interval_s = reconnect_interval_s
        self._clock = clock
        self._next_reconnect_at = 0.0
        self._closed = False
        # mjpython runs Python on a worker thread so macOS can reserve its main
        # thread for MuJoCo's Cocoa viewer. pygame.init() would also initialize
        # SDL video; pumping that Cocoa event loop here aborts the process. The
        # controller updater polls SDL input without initializing video.
        pygame.joystick.init()
        self._controller_api.init()
        count = pygame.joystick.get_count()
        if count == 0:
            self.close()
            raise RuntimeError("no joystick found; pair/connect the PS controller and retry")
        if device_index >= count:
            self.close()
            raise RuntimeError(
                f"joystick index {device_index} not found; detected {count} device(s)"
            )
        if not self._controller_api.is_controller(device_index):
            self.close()
            raise RuntimeError(
                f"joystick index {device_index} is not a supported PlayStation-style controller"
            )
        self._device = pygame.joystick.Joystick(device_index)
        try:
            self._device.init()
            self._controller = self._controller_api.Controller(device_index)
            self._instance_id = self._device.get_instance_id()
        except pygame.error:
            self.close()
            raise
        self._connected = True
        connected_name = self.name
        if calibration is not None and calibration.device_name != connected_name:
            self.close()
            raise RuntimeError(
                "joystick calibration device mismatch: "
                f"profile={calibration.device_name!r}, connected={connected_name!r}"
            )
        self._expected_name = connected_name

    @property
    def name(self) -> str:
        if self._device is None:
            return "disconnected joystick"
        return self._device.get_name()

    @property
    def device_name(self) -> str:
        return self._expected_name or self.name

    @property
    def calibrated(self) -> bool:
        return self._calibration is not None

    def _axis(self, index: int) -> float:
        assert self._device is not None
        if index >= self._device.get_numaxes():
            return 0.0
        raw = float(self._device.get_axis(index))
        calibration = (
            self._calibration.axes.get(index, AxisCalibration())
            if self._calibration is not None
            else AxisCalibration()
        )
        return calibration.normalize(raw, self._config.deadzone)

    def _button(self, index: int) -> bool:
        assert self._device is not None
        return index < self._device.get_numbuttons() and bool(self._device.get_button(index))

    def _disconnect(self) -> OperatorInput:
        """Clear latched input and return the only safe disconnected command."""

        if self._connected:
            self._next_reconnect_at = self._clock() + self._reconnect_interval_s
        self._connected = False
        self._reset_down = False
        self._resume_down = False
        self._release_device()
        return self._stopped_input()

    @staticmethod
    def _stopped_input() -> OperatorInput:
        return OperatorInput(
            command=np.zeros(3, dtype=np.float32),
            emergency_stop=True,
            connected=False,
        )

    def _release_device(self) -> None:
        controller = self._controller
        self._controller = None
        if controller is not None:
            try:
                if controller.get_init():
                    controller.quit()
            except self._pygame.error:
                pass
        if self._device is None:
            return
        device = self._device
        self._device = None
        try:
            if device.get_init():
                device.quit()
        except self._pygame.error:
            # A driver can invalidate the object between removal notification
            # and cleanup; commands are already cleared by _disconnect().
            pass

    def _attempt_reconnect(self) -> bool:
        if self._closed or self._clock() < self._next_reconnect_at:
            return False
        self._next_reconnect_at = self._clock() + self._reconnect_interval_s
        try:
            self._controller_api.update()
            count = self._pygame.joystick.get_count()
            indices = list(range(count))
            if self._device_index in indices:
                indices.remove(self._device_index)
                indices.insert(0, self._device_index)
            for index in indices:
                candidate = self._pygame.joystick.Joystick(index)
                candidate.init()
                if self._expected_name is not None and candidate.get_name() != self._expected_name:
                    candidate.quit()
                    continue
                if not self._controller_api.is_controller(index):
                    candidate.quit()
                    continue
                controller = self._controller_api.Controller(index)
                self._device = candidate
                self._controller = controller
                self._instance_id = candidate.get_instance_id()
                self._connected = True
                self._reset_down = False
                self._resume_down = False
                return True
        except self._pygame.error:
            self._release_device()
        return False

    def _device_was_removed(self) -> bool:
        assert self._device is not None
        return self._controller is None or not bool(self._controller.attached())

    def sample(self) -> OperatorInput:
        if (self._device is None or not self._connected) and not self._attempt_reconnect():
            return self._stopped_input()
        try:
            self._controller_api.update()
            removed = self._device_was_removed()
            initialized = self._device.get_init()
        except self._pygame.error:
            return self._disconnect()
        if removed or not initialized:
            return self._disconnect()
        try:
            command = np.asarray(
                [
                    -self._axis(self._config.axis_forward) * self._config.max_forward,
                    -self._axis(self._config.axis_lateral) * self._config.max_lateral,
                    -self._axis(self._config.axis_yaw) * self._config.max_yaw,
                ],
                dtype=np.float32,
            )
            reset_down = self._button(self._config.reset_button)
            emergency_stop_down = self._button(self._config.emergency_stop_button)
            resume_down = self._button(self._config.resume_button)
            quit_down = self._button(self._config.quit_button)
        except self._pygame.error:
            return self._disconnect()
        reset = reset_down and not self._reset_down
        resume = resume_down and not self._resume_down
        self._reset_down = reset_down
        self._resume_down = resume_down
        return OperatorInput(
            command=command,
            reset=reset,
            emergency_stop=emergency_stop_down,
            resume=resume,
            quit=quit_down,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._connected = False
        self._reset_down = False
        self._resume_down = False
        self._release_device()
        self._controller_api.quit()
        self._pygame.joystick.quit()


def list_joysticks() -> list[str]:
    import pygame

    pygame.joystick.init()
    names = []
    try:
        for index in range(pygame.joystick.get_count()):
            device = pygame.joystick.Joystick(index)
            device.init()
            try:
                names.append(device.get_name())
            finally:
                device.quit()
        return names
    finally:
        pygame.joystick.quit()


def capture_calibration(
    config: JoystickConfig,
    output_path: Path,
    *,
    device_index: int = 0,
    sweep_seconds: float = 6.0,
    prompt: Callable[[str], str] = input,
) -> JoystickCalibration:
    """Interactively measure center and range for the three command axes.

    Calibration changes normalized ``vx``, ``vy``, and yaw commands only. It
    cannot address joints, actuator controls, position targets, or torques.
    """

    if not 2.0 <= sweep_seconds <= 30.0:
        raise ValueError("calibration sweep must be between 2 and 30 seconds")
    joystick = PSJoystick(config, device_index)
    axes = sorted({config.axis_lateral, config.axis_forward, config.axis_yaw})
    try:
        assert joystick._device is not None  # Internal access is confined to calibration.
        prompt("Release both sticks to center, then press Enter.")
        center_samples: dict[int, list[float]] = {axis: [] for axis in axes}
        center_deadline = time.monotonic() + 0.75
        while time.monotonic() < center_deadline:
            joystick._controller_api.update()
            for axis in axes:
                center_samples[axis].append(float(joystick._device.get_axis(axis)))
            time.sleep(0.01)
        centers = {axis: float(np.median(values)) for axis, values in center_samples.items()}

        prompt(
            "Press Enter, then move both sticks through their full range in circles "
            f"for {sweep_seconds:g} seconds."
        )
        minima = centers.copy()
        maxima = centers.copy()
        sweep_deadline = time.monotonic() + sweep_seconds
        while time.monotonic() < sweep_deadline:
            joystick._controller_api.update()
            for axis in axes:
                raw = float(joystick._device.get_axis(axis))
                minima[axis] = min(minima[axis], raw)
                maxima[axis] = max(maxima[axis], raw)
            time.sleep(0.01)

        calibrations = {
            axis: AxisCalibration(minima[axis], centers[axis], maxima[axis]) for axis in axes
        }
        for axis, calibration in calibrations.items():
            if (
                min(
                    calibration.center - calibration.minimum,
                    calibration.maximum - calibration.center,
                )
                < 0.25
            ):
                raise RuntimeError(
                    f"axis {axis} did not reach enough range; repeat calibration and move fully"
                )
        profile = JoystickCalibration(device_name=joystick.name, axes=calibrations)
        profile.save(output_path)
        return profile
    finally:
        joystick.close()
