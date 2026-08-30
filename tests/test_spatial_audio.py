import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest

from everest_g1.beacon import ARM_VALUE, BeaconSettings
from everest_g1.models import RescueObservation
from everest_g1.mujoco import MujocoAudioMonitor, MujocoRescueController
from everest_g1.spatial_audio import (
    AcousticBeaconSensor,
    AcousticSensorConfig,
    BearingEstimate,
    MicArray,
    SpatialAudioSettings,
    SpatialCueConfig,
    SpatialCueRenderer,
)

QUIET_SENSOR = AcousticSensorConfig(tdoa_jitter_s=0.0, level_noise=0.0, smoothing=1.0)


def _source_at(robot_xy: tuple[float, float], bearing_rad: float, range_m: float):
    return (
        robot_xy[0] + range_m * math.cos(bearing_rad),
        robot_xy[1] + range_m * math.sin(bearing_rad),
    )


@pytest.mark.parametrize("true_bearing", [0.0, 0.6, -1.2, 2.4, math.pi - 0.05, -2.9])
@pytest.mark.parametrize("robot_yaw", [0.0, 1.1, -2.2])
def test_noise_free_array_recovers_bearing_in_the_body_frame(
    true_bearing: float, robot_yaw: float
) -> None:
    sensor = AcousticBeaconSensor(QUIET_SENSOR)
    robot_xy = (1.5, -0.75)
    source = _source_at(robot_xy, robot_yaw + true_bearing, 3.0)

    estimate = sensor.sense(robot_xy=robot_xy, robot_yaw_rad=robot_yaw, source_xy=source)

    error = abs(
        math.atan2(
            math.sin(estimate.bearing_rad - true_bearing),
            math.cos(estimate.bearing_rad - true_bearing),
        )
    )
    # One grid cell is 2*pi/720; near-field curvature costs a little more.
    assert error < math.radians(3.0)


def test_planar_array_separates_a_source_behind_from_one_ahead() -> None:
    sensor = AcousticBeaconSensor(QUIET_SENSOR)
    robot_xy = (0.0, 0.0)

    ahead = sensor.sense(
        robot_xy=robot_xy, robot_yaw_rad=0.0, source_xy=_source_at(robot_xy, 0.3, 4.0)
    )
    sensor.reset()
    behind = sensor.sense(
        robot_xy=robot_xy, robot_yaw_rad=0.0, source_xy=_source_at(robot_xy, math.pi - 0.3, 4.0)
    )

    assert math.cos(ahead.bearing_rad) > 0.0
    assert math.cos(behind.bearing_rad) < 0.0


def test_jittered_bearing_stays_usable_and_reports_confidence() -> None:
    sensor = AcousticBeaconSensor(AcousticSensorConfig(seed=7))
    robot_xy = (0.0, 0.0)
    source = _source_at(robot_xy, 0.5, 3.0)

    errors = []
    for _ in range(40):
        estimate = sensor.sense(robot_xy=robot_xy, robot_yaw_rad=0.0, source_xy=source)
        errors.append(abs(estimate.bearing_rad - 0.5))
        assert 0.0 <= estimate.confidence <= 1.0

    # Smoothing needs a few frames; judge the settled tail, not the first frame.
    assert max(errors[10:]) < math.radians(12.0)


def test_range_from_level_is_recovered_without_noise_and_never_collapses() -> None:
    sensor = AcousticBeaconSensor(QUIET_SENSOR)
    estimate = sensor.sense(robot_xy=(0.0, 0.0), robot_yaw_rad=0.0, source_xy=(2.5, 0.0))
    assert estimate.range_m == pytest.approx(2.5, rel=1e-6)

    touching = sensor.sense(robot_xy=(0.0, 0.0), robot_yaw_rad=0.0, source_xy=(0.0, 0.0))
    assert touching.range_m >= QUIET_SENSOR.minimum_range_m


def test_degenerate_microphone_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="three non-collinear"):
        MicArray(offsets_m=((0.0, 0.0), (0.1, 0.0)))
    with pytest.raises(ValueError, match="collinear"):
        MicArray(offsets_m=((0.0, 0.0), (0.1, 0.0), (0.2, 0.0)))


def _channel_energy(renderer: SpatialCueRenderer) -> tuple[float, float]:
    samples = renderer.samples()
    return float(np.sum(samples[:, 0] ** 2)), float(np.sum(samples[:, 1] ** 2))


def test_cue_pans_left_for_a_source_on_the_left_and_right_for_one_on_the_right() -> None:
    left = SpatialCueRenderer()
    right = SpatialCueRenderer()

    left.update(dt_s=0.02, bearing_rad=math.pi / 2, distance_m=2.0)
    right.update(dt_s=0.02, bearing_rad=-math.pi / 2, distance_m=2.0)

    left_l, left_r = _channel_energy(left)
    right_l, right_r = _channel_energy(right)
    assert left_l > left_r * 4
    assert right_r > right_l * 4


def test_cue_applies_an_interaural_delay_toward_the_source() -> None:
    config = SpatialCueConfig()
    renderer = SpatialCueRenderer(config)
    # A partial pan keeps both channels audible, so both onsets are measurable.
    bearing = math.pi / 6
    renderer.update(dt_s=0.02, bearing_rad=bearing, distance_m=2.0)
    samples = renderer.samples()

    left_onset = int(np.argmax(np.abs(samples[:, 0]) > 1e-4))
    right_onset = int(np.argmax(np.abs(samples[:, 1]) > 1e-4))
    expected = round(math.sin(bearing) * config.max_itd_s * config.sample_rate_hz)
    assert right_onset - left_onset == expected


def test_cue_repeats_faster_as_the_person_gets_closer() -> None:
    far = SpatialCueRenderer()
    near = SpatialCueRenderer()
    for _ in range(150):
        far.update(dt_s=0.02, bearing_rad=0.0, distance_m=4.0)
        near.update(dt_s=0.02, bearing_rad=0.0, distance_m=0.2)

    assert near.beeps > far.beeps * 2


def _dominant_hz(samples: np.ndarray, sample_rate_hz: int) -> float:
    mono = samples.sum(axis=1)
    spectrum = np.abs(np.fft.rfft(mono))
    return float(np.fft.rfftfreq(mono.size, 1.0 / sample_rate_hz)[int(np.argmax(spectrum))])


def test_a_source_behind_the_robot_drops_the_cue_an_octave() -> None:
    config = SpatialCueConfig()
    ahead = SpatialCueRenderer(config)
    behind = SpatialCueRenderer(config)
    ahead.update(dt_s=0.02, bearing_rad=0.2, distance_m=2.0)
    behind.update(dt_s=0.02, bearing_rad=math.pi - 0.2, distance_m=2.0)

    ahead_hz = _dominant_hz(ahead.samples(), config.sample_rate_hz)
    behind_hz = _dominant_hz(behind.samples(), config.sample_rate_hz)
    assert ahead_hz == pytest.approx(config.tone_hz, rel=0.1)
    assert behind_hz == pytest.approx(config.tone_hz * config.behind_tone_ratio, rel=0.1)


def test_latch_and_call_tones_are_sequenced_rather_than_stacked() -> None:
    config = SpatialCueConfig()
    renderer = SpatialCueRenderer(config)
    renderer.mark_proximity_latched()
    renderer.mark_call_submitted()

    assert renderer.events == ["proximity_latched", "call_submitted"]
    expected = config.latch_seconds + 2.0 * config.call_seconds
    assert renderer.seconds == pytest.approx(expected, rel=0.02)


def test_ranging_beeps_stop_once_proximity_latches() -> None:
    renderer = SpatialCueRenderer()
    for _ in range(100):
        renderer.update(dt_s=0.02, bearing_rad=0.0, distance_m=1.0)
    before = renderer.beeps
    renderer.mark_proximity_latched()
    for _ in range(400):
        assert not renderer.update(dt_s=0.02, bearing_rad=0.0, distance_m=1.0)

    assert before > 0
    assert renderer.beeps == before
    # Time still advances, so the track keeps its true duration.
    assert renderer.seconds == pytest.approx(10.0, rel=0.05)


def test_cue_track_writes_a_playable_stereo_wav(tmp_path: Path) -> None:
    renderer = SpatialCueRenderer()
    for step in range(200):
        renderer.update(dt_s=0.02, bearing_rad=0.9 - step * 0.004, distance_m=4.0 - step * 0.018)
    renderer.mark_proximity_latched()
    path = renderer.write_wav(tmp_path / "cue.wav")

    with wave.open(str(path), "rb") as stream:
        assert stream.getnchannels() == 2
        assert stream.getsampwidth() == 2
        assert stream.getframerate() == SpatialCueConfig().sample_rate_hz
        frames = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
    assert frames.size > 0
    assert np.abs(frames).max() > 1000


def test_cue_buffer_is_bounded_by_max_seconds() -> None:
    renderer = SpatialCueRenderer(SpatialCueConfig(max_seconds=1.0))
    for _ in range(500):
        renderer.update(dt_s=0.02, bearing_rad=0.0, distance_m=1.0)

    assert renderer.truncated
    assert renderer.samples().shape[0] <= SpatialCueConfig().sample_rate_hz


def test_controller_renders_a_cue_and_logs_it_without_leaking_configuration(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "events.jsonl"
    controller = MujocoRescueController(
        person_xy=(0.65, 0.0),
        control_dt_s=0.01,
        audit_log=log_path,
        simulation_id="cue-run",
        spatial_audio=SpatialAudioSettings(render_cue=True, output_path=tmp_path / "rescue.wav"),
    )
    qpos = np.asarray([0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0])
    for _ in range(30):
        controller.update(qpos)
    controller.close()

    assert controller.latch.latched
    assert controller.spatial_audio_path == tmp_path / "rescue-cue-run.wav"
    assert controller.spatial_audio_path.exists()

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "simulation_started",
        "spatial_audio_started",
        "proximity_reached_call_disarmed",
        "spatial_audio_written",
    ]
    written = records[-1]
    assert written["events"] == ["proximity_latched"]
    assert written["beeps"] > 0


def test_acoustic_localization_steers_but_never_gates_proximity(tmp_path: Path) -> None:
    controller = MujocoRescueController(
        person_xy=(5.0, 0.0),
        control_dt_s=0.02,
        audit_log=tmp_path / "events.jsonl",
        simulation_id="steer-only",
        spatial_audio=SpatialAudioSettings(acoustic_localization=True),
    )
    assert controller.audio is not None and controller.audio.sensor is not None

    # A microphone that claims the person is underfoot must not stop the robot.
    controller.audio.sensor.sense = lambda **_: BearingEstimate(  # type: ignore[method-assign]
        bearing_rad=0.0, range_m=0.0, confidence=1.0
    )
    qpos = np.asarray([0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0])
    for _ in range(50):
        command = controller.update(qpos)
    controller.close()

    assert not controller.latch.latched
    assert not controller.call_submitted
    assert controller.last_command.surface_distance_m == pytest.approx(4.45)
    assert command[0] > 0.0


def test_acoustic_localization_still_reaches_the_person(tmp_path: Path) -> None:
    person_xy = (2.6, 0.75)
    controller = MujocoRescueController(
        person_xy=person_xy,
        control_dt_s=0.02,
        audit_log=tmp_path / "events.jsonl",
        simulation_id="acoustic-approach",
        spatial_audio=SpatialAudioSettings(
            acoustic_localization=True, render_cue=True, output_path=tmp_path / "rescue.wav"
        ),
    )
    qpos = np.asarray([0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for _ in range(2_000):
        command = controller.update(qpos)
        if controller.latch.latched:
            break
        qpos[0] += float(command[0]) * 0.02
        qpos[1] += float(command[1]) * 0.02
    controller.close()

    assert controller.latch.latched
    assert controller.last_command.surface_distance_m <= 0.15
    assert (tmp_path / "rescue-acoustic-approach.wav").exists()


def test_passive_monitor_writes_controller_or_scan_audio_without_motion_authority(
    tmp_path: Path,
) -> None:
    monitor = MujocoAudioMonitor(
        person_xy=(1.35, 0.30),
        control_dt_s=0.002,
        settings=SpatialAudioSettings(
            acoustic_localization=True,
            render_cue=True,
            output_path=tmp_path / "passive.wav",
        ),
        simulation_id="scan-audio",
        audit_log=tmp_path / "events.jsonl",
        mode="gemini-scan",
    )
    qpos = np.asarray([0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0])
    for _ in range(100):
        monitor.update(qpos)
    monitor.close()

    assert monitor.spatial_audio_path == tmp_path / "passive-scan-audio.wav"
    assert monitor.spatial_audio_path.exists()
    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert records[0]["motion_authority"] is False
    assert records[-1]["event"] == "spatial_audio_written"


def test_controller_monitor_queues_one_call_without_taking_motion_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import everest_g1.beacon as beacon_module

    monkeypatch.setenv("EVEREST_ARM_LIVE_CALL", ARM_VALUE)
    monkeypatch.setenv("BEACON_API_URL", "https://beacon.test")
    monkeypatch.setenv("BEACON_API_TOKEN", "test-token")
    calls: list[RescueObservation] = []

    def fake_post(_settings: BeaconSettings, observation: RescueObservation) -> dict[str, object]:
        calls.append(observation)
        return {"incident_id": "inc-controller", "status": "queued"}

    monkeypatch.setattr(beacon_module, "post_incident", fake_post)
    monitor = MujocoAudioMonitor(
        person_xy=(0.65, 0.0),
        control_dt_s=0.25,
        settings=SpatialAudioSettings(),
        simulation_id="controller-live-test",
        audit_log=tmp_path / "events.jsonl",
        mode="controller",
        arm_live_call=True,
    )
    qpos = np.asarray([0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0])
    jpeg = b"\xff\xd8controller-camera\xff\xd9"

    assert monitor.update(qpos, image_supplier=lambda: jpeg) is None
    assert monitor.update(qpos, image_supplier=lambda: jpeg) is None
    monitor.close()

    assert monitor.latch.latched
    assert monitor.live_call_armed
    assert monitor.call_submitted
    assert monitor.front_camera_status == "captured"
    assert monitor.front_camera_bytes == len(jpeg)
    assert len(calls) == 1
    assert calls[0].simulation_id == "controller-live-test"
    assert calls[0].distance_m == pytest.approx(0.1)
    assert calls[0].image_jpeg == jpeg
