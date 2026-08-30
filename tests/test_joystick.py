import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from summit_sentinel.config import load_config
from summit_sentinel.joystick import (
    AxisCalibration,
    JoystickCalibration,
    PSJoystick,
    apply_deadzone,
    list_joysticks,
)
from summit_sentinel.simulation import SummitSentinelEnv


class FakeDevice:
    def __init__(self) -> None:
        self.axes = [0.55, -1.0, 0.5]
        self.buttons = [0] * 10
        self.initialized = False
        self.attached = True
        self.quit_calls = 0

    def init(self) -> None:
        self.initialized = True

    def quit(self) -> None:
        self.initialized = False
        self.quit_calls += 1

    def get_init(self) -> bool:
        return self.initialized

    def get_name(self) -> str:
        return "Mock DualSense"

    def get_instance_id(self) -> int:
        return 42

    def get_numaxes(self) -> int:
        return len(self.axes)

    def get_axis(self, index: int) -> float:
        return self.axes[index]

    def get_numbuttons(self) -> int:
        return len(self.buttons)

    def get_button(self, index: int) -> int:
        return self.buttons[index]


class FakeJoystickModule:
    def __init__(self, device: FakeDevice) -> None:
        self.device = device
        self.quit_calls = 0
        self.count = 1
        self.get_count_calls = 0

    def init(self) -> None:
        return None

    def quit(self) -> None:
        self.quit_calls += 1

    def get_count(self) -> int:
        self.get_count_calls += 1
        return self.count

    def Joystick(self, index: int) -> FakeDevice:
        assert index == 0
        return self.device


class FakeEventQueue:
    def __init__(self) -> None:
        self.events: list[SimpleNamespace] = []
        self.pump_calls = 0

    def pump(self) -> None:
        self.pump_calls += 1

    def get(self, event_types: list[int]) -> list[SimpleNamespace]:
        selected = [event for event in self.events if event.type in event_types]
        self.events = [event for event in self.events if event.type not in event_types]
        return selected


class FakeController:
    def __init__(self, device: FakeDevice) -> None:
        self.device = device
        self.initialized = True
        self.quit_calls = 0

    def attached(self) -> bool:
        return self.initialized and self.device.attached

    def get_init(self) -> bool:
        return self.initialized

    def quit(self) -> None:
        self.initialized = False
        self.quit_calls += 1


class FakeControllerModule:
    def __init__(self, pygame: "FakePygame") -> None:
        self.pygame = pygame
        self.initialized = False
        self.controllers: list[FakeController] = []
        self.init_calls = 0
        self.quit_calls = 0
        self.update_calls = 0

    def init(self) -> None:
        self.initialized = True
        self.init_calls += 1

    def quit(self) -> None:
        self.initialized = False
        self.quit_calls += 1

    def update(self) -> None:
        self.update_calls += 1
        removed = self.pygame.event.get([self.pygame.JOYDEVICEREMOVED])
        if any(event.instance_id == 42 for event in removed):
            self.pygame.device.attached = False

    def is_controller(self, index: int) -> bool:
        return index == 0

    def Controller(self, index: int) -> FakeController:
        assert index == 0
        self.pygame.device.attached = True
        controller = FakeController(self.pygame.device)
        self.controllers.append(controller)
        return controller


class FakeDisplay:
    def __init__(self) -> None:
        self.init_calls = 0

    def init(self) -> None:
        self.init_calls += 1


class FakePygame:
    error = RuntimeError
    JOYDEVICEREMOVED = 1542

    def __init__(self) -> None:
        self.device = FakeDevice()
        self.joystick = FakeJoystickModule(self.device)
        self.event = FakeEventQueue()
        self.display = FakeDisplay()
        self.quit_calls = 0
        self.init_calls = 0

    def init(self) -> None:
        self.init_calls += 1

    def quit(self) -> None:
        self.quit_calls += 1


@pytest.fixture
def fake_pygame(monkeypatch: pytest.MonkeyPatch) -> FakePygame:
    pygame = FakePygame()
    controller = FakeControllerModule(pygame)
    monkeypatch.setitem(sys.modules, "pygame", pygame)
    monkeypatch.setitem(sys.modules, "pygame._sdl2.controller_old", controller)
    return pygame


@pytest.mark.parametrize("value", [-0.1, -0.05, 0.0, 0.05, 0.1])
def test_deadzone_suppresses_center_drift(value: float) -> None:
    assert apply_deadzone(value, 0.1) == 0.0


def test_deadzone_rescales_remaining_range() -> None:
    assert apply_deadzone(1.0, 0.1) == 1.0
    assert apply_deadzone(-1.0, 0.1) == -1.0
    assert apply_deadzone(0.55, 0.1) == pytest.approx(0.5)


def test_mapping_buttons_and_reset_rising_edge(fake_pygame: FakePygame) -> None:
    joystick = PSJoystick(load_config().joystick)
    fake_pygame.device.buttons[0] = 1
    fake_pygame.device.buttons[1] = 1
    fake_pygame.device.buttons[3] = 1
    fake_pygame.device.buttons[9] = 1

    first = joystick.sample()
    np.testing.assert_allclose(first.command, [1.0, -0.5, -(0.4 / 0.9)])
    assert first.reset
    assert first.emergency_stop
    assert first.resume
    assert first.quit
    assert not joystick.sample().reset
    assert not joystick.sample().resume

    fake_pygame.device.buttons[0] = 0
    assert not joystick.sample().reset
    fake_pygame.device.buttons[0] = 1
    assert joystick.sample().reset


def test_hot_unplug_returns_zero_and_clears_latched_state(fake_pygame: FakePygame) -> None:
    joystick = PSJoystick(load_config().joystick)
    fake_pygame.device.buttons[0] = 1
    assert joystick.sample().reset
    fake_pygame.event.events.append(
        SimpleNamespace(type=fake_pygame.JOYDEVICEREMOVED, instance_id=42)
    )

    disconnected = joystick.sample()
    np.testing.assert_array_equal(disconnected.command, np.zeros(3, dtype=np.float32))
    assert not disconnected.reset
    assert not disconnected.quit
    assert disconnected.emergency_stop
    assert not disconnected.connected
    assert not fake_pygame.device.get_init()

    again = joystick.sample()
    np.testing.assert_array_equal(again.command, np.zeros(3, dtype=np.float32))
    assert not again.reset


def test_hot_unplug_throttles_discovery_then_reconnects_without_clearing_local_latch(
    fake_pygame: FakePygame,
) -> None:
    now = [0.0]
    joystick = PSJoystick(
        load_config().joystick,
        reconnect_interval_s=1.0,
        clock=lambda: now[0],
    )
    env = SummitSentinelEnv(use_policy=False, auto_reset=False)
    fake_pygame.event.events.append(
        SimpleNamespace(type=fake_pygame.JOYDEVICEREMOVED, instance_id=42)
    )
    disconnected = joystick.sample()
    assert disconnected.emergency_stop
    env.emergency_stop("joystick disconnected", authority="local")
    initial_discovery_calls = fake_pygame.joystick.get_count_calls

    fake_pygame.joystick.count = 0
    now[0] = 0.9
    assert not joystick.sample().connected
    assert fake_pygame.joystick.get_count_calls == initial_discovery_calls
    now[0] = 1.0
    assert not joystick.sample().connected
    assert fake_pygame.joystick.get_count_calls == initial_discovery_calls + 1

    fake_pygame.joystick.count = 1
    fake_pygame.device.buttons[0] = 1
    fake_pygame.device.buttons[3] = 1
    now[0] = 1.9
    assert not joystick.sample().connected
    assert fake_pygame.joystick.get_count_calls == initial_discovery_calls + 1
    now[0] = 2.0
    reconnected = joystick.sample()
    assert reconnected.connected
    assert not reconnected.emergency_stop
    assert reconnected.reset
    assert reconnected.resume
    assert env.emergency_stop_latched
    assert env.stop_authority == "local"

    env.reset(authority="local")
    assert env.resume(authority="local")
    assert not env.emergency_stop_latched
    joystick.close()


def test_close_and_listing_release_pygame_resources(fake_pygame: FakePygame) -> None:
    joystick = PSJoystick(load_config().joystick)
    joystick.close()
    assert fake_pygame.device.quit_calls == 1
    assert fake_pygame.joystick.quit_calls == 1
    assert fake_pygame.quit_calls == 0

    assert list_joysticks() == ["Mock DualSense"]
    assert fake_pygame.device.quit_calls == 2
    assert fake_pygame.joystick.quit_calls == 2
    assert fake_pygame.quit_calls == 0


def test_joystick_polling_never_initializes_global_pygame_or_video(
    fake_pygame: FakePygame,
) -> None:
    joystick = PSJoystick(load_config().joystick)
    joystick.sample()

    assert fake_pygame.init_calls == 0
    assert fake_pygame.display.init_calls == 0
    joystick.close()


def test_pygame_support_prompt_is_suppressed_for_json_output() -> None:
    assert os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] == "1"


def test_calibration_profile_round_trip_and_asymmetric_normalization(tmp_path) -> None:
    path = tmp_path / "dualsense.json"
    profile = JoystickCalibration(
        device_name="Mock DualSense",
        axes={0: AxisCalibration(minimum=-0.5, center=0.1, maximum=0.9)},
    )
    profile.save(path)
    loaded = JoystickCalibration.load(path)

    assert loaded == profile
    assert loaded.axes[0].normalize(0.5, 0.0) == pytest.approx(0.5)
    assert loaded.axes[0].normalize(-0.2, 0.0) == pytest.approx(-0.5)


def test_calibration_profile_must_match_connected_device(
    fake_pygame: FakePygame,
) -> None:
    profile = JoystickCalibration(device_name="Different Controller", axes={})
    with pytest.raises(RuntimeError, match="device mismatch"):
        PSJoystick(load_config().joystick, calibration=profile)
    assert not fake_pygame.device.get_init()
