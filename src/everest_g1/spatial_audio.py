"""Spatial audio for the rescue event.

Two independent halves share this module:

* :class:`AcousticBeaconSensor` simulates a small planar microphone array on the
  G1 and recovers the *bearing* of the downed person from time differences of
  arrival. It is a sensing modality, and it is deliberately allowed to steer the
  approach only. Range recovered from sound level is coarse and never gates the
  proximity latch or a BeaconCall.
* :class:`SpatialCueRenderer` turns the same relative geometry into a stereo
  operator cue: pan and interaural delay follow bearing, repetition rate and
  gain follow distance, and the proximity latch and call submission each get a
  distinct tone. It renders to a buffer and writes a WAV at shutdown, so nothing
  here needs an audio device or blocks the control loop.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SPEED_OF_SOUND_MPS = 343.0

#: Four torso microphones on a 0.12 m square. Three non-collinear microphones
#: are the minimum that resolves front from back; four keeps the geometry
#: symmetric so bearing error does not depend on which way the robot faces.
DEFAULT_MIC_OFFSETS_M: tuple[tuple[float, float], ...] = (
    (0.06, 0.06),
    (0.06, -0.06),
    (-0.06, -0.06),
    (-0.06, 0.06),
)


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class MicArray:
    """Body-frame microphone offsets in metres (x forward, y left)."""

    offsets_m: tuple[tuple[float, float], ...] = DEFAULT_MIC_OFFSETS_M

    def __post_init__(self) -> None:
        if len(self.offsets_m) < 3:
            raise ValueError("bearing needs at least three non-collinear microphones")
        positions = self.positions
        if np.linalg.matrix_rank(positions - positions.mean(axis=0)) < 2:
            raise ValueError("microphone offsets must not be collinear")

    @property
    def positions(self) -> np.ndarray:
        return np.asarray(self.offsets_m, dtype=np.float64)

    @property
    def aperture_m(self) -> float:
        positions = self.positions
        spread = positions[:, None, :] - positions[None, :, :]
        return float(np.linalg.norm(spread, axis=-1).max())

    def world_positions(self, robot_xy: tuple[float, float], robot_yaw_rad: float) -> np.ndarray:
        cos_yaw = math.cos(robot_yaw_rad)
        sin_yaw = math.sin(robot_yaw_rad)
        rotation = np.asarray([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)
        return self.positions @ rotation.T + np.asarray(robot_xy, dtype=np.float64)


@dataclass(frozen=True)
class AcousticSensorConfig:
    """Noise model for the simulated array and its bearing estimator."""

    array: MicArray = field(default_factory=MicArray)
    #: Per-microphone arrival-time jitter. 8 us is a realistic GCC-PHAT residual
    #: for a 48 kHz array with interpolation, and is not a claim about hardware.
    tdoa_jitter_s: float = 8e-6
    #: Multiplicative level error, driving the coarse range estimate only.
    level_noise: float = 0.12
    reference_level_at_1m: float = 1.0
    minimum_range_m: float = 0.05
    #: Exponential smoothing applied to the bearing unit vector, 1.0 disables it.
    smoothing: float = 0.35
    bearing_grid: int = 720
    seed: int = 0

    def __post_init__(self) -> None:
        if self.tdoa_jitter_s < 0 or self.level_noise < 0:
            raise ValueError("noise magnitudes must be non-negative")
        if not 0.0 < self.smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        if self.bearing_grid < 8:
            raise ValueError("bearing_grid must resolve the array aperture")


@dataclass(frozen=True)
class BearingEstimate:
    """What the array believes, in the robot body frame.

    ``range_m`` is a coarse level-based figure kept for telemetry only. Nothing
    in the control path may position the robot from it; use
    :func:`everest_g1.rescue.target_from_bearing` with the measured range.
    """

    bearing_rad: float
    range_m: float
    confidence: float


class AcousticBeaconSensor:
    """Recover a body-frame bearing to a sound source from simulated TDOAs."""

    def __init__(self, config: AcousticSensorConfig | None = None) -> None:
        self.config = config or AcousticSensorConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._smoothed: np.ndarray | None = None
        angles = np.linspace(-math.pi, math.pi, self.config.bearing_grid, endpoint=False)
        self._grid_angles = angles
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        # Far-field plane wave: a microphone displaced along the arrival
        # direction hears the source earlier, hence the negative sign.
        self._grid_delays = -(self.config.array.positions @ directions.T) / SPEED_OF_SOUND_MPS
        self._grid_delays -= self._grid_delays.mean(axis=0, keepdims=True)

    def reset(self) -> None:
        self._smoothed = None

    def sense(
        self,
        *,
        robot_xy: tuple[float, float],
        robot_yaw_rad: float,
        source_xy: tuple[float, float],
    ) -> BearingEstimate:
        measured = self._measure_delays(robot_xy, robot_yaw_rad, source_xy)
        residuals = self._grid_delays - measured[:, None]
        errors = np.sqrt(np.mean(residuals**2, axis=0))
        best = int(np.argmin(errors))
        raw_bearing = float(self._grid_angles[best])

        aperture_delay = self.config.array.aperture_m / SPEED_OF_SOUND_MPS
        confidence = float(np.clip(1.0 - errors[best] / max(aperture_delay, 1e-9), 0.0, 1.0))
        bearing = self._smooth(raw_bearing)
        return BearingEstimate(
            bearing_rad=bearing,
            range_m=self._measure_range(robot_xy, source_xy),
            confidence=confidence,
        )

    def _measure_delays(
        self,
        robot_xy: tuple[float, float],
        robot_yaw_rad: float,
        source_xy: tuple[float, float],
    ) -> np.ndarray:
        mics = self.config.array.world_positions(robot_xy, robot_yaw_rad)
        distances = np.linalg.norm(np.asarray(source_xy, dtype=np.float64) - mics, axis=1)
        delays = distances / SPEED_OF_SOUND_MPS
        if self.config.tdoa_jitter_s > 0:
            delays = delays + self._rng.normal(0.0, self.config.tdoa_jitter_s, size=delays.shape)
        return delays - delays.mean()

    def _measure_range(
        self, robot_xy: tuple[float, float], source_xy: tuple[float, float]
    ) -> float:
        true_range = max(
            math.dist(robot_xy, source_xy),
            self.config.minimum_range_m,
        )
        level = self.config.reference_level_at_1m / true_range
        if self.config.level_noise > 0:
            level *= float(np.exp(self._rng.normal(0.0, self.config.level_noise)))
        level = max(level, 1e-9)
        return max(self.config.reference_level_at_1m / level, self.config.minimum_range_m)

    def _smooth(self, bearing_rad: float) -> float:
        sample = np.asarray([math.cos(bearing_rad), math.sin(bearing_rad)])
        if self._smoothed is None:
            self._smoothed = sample
        else:
            alpha = self.config.smoothing
            self._smoothed = alpha * sample + (1.0 - alpha) * self._smoothed
        return float(math.atan2(self._smoothed[1], self._smoothed[0]))


@dataclass(frozen=True)
class SpatialCueConfig:
    """Stereo operator cue: bearing to pan/ITD, distance to rate/gain."""

    sample_rate_hz: int = 22050
    tone_hz: float = 660.0
    #: Sources behind the robot drop an octave, so front and back never sound
    #: alike even though interaural level differences alone are ambiguous.
    behind_tone_ratio: float = 0.5
    beep_seconds: float = 0.06
    fade_seconds: float = 0.006
    near_interval_s: float = 0.10
    far_interval_s: float = 0.75
    interval_slope_s_per_m: float = 0.16
    max_itd_s: float = 0.00065
    reference_distance_m: float = 1.5
    minimum_gain: float = 0.15
    latch_tone_hz: float = 880.0
    latch_seconds: float = 0.6
    call_tone_hz: tuple[float, float] = (660.0, 990.0)
    call_seconds: float = 0.22
    max_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.sample_rate_hz < 8000:
            raise ValueError("sample_rate_hz must be at least 8000")
        if self.near_interval_s <= 0 or self.far_interval_s < self.near_interval_s:
            raise ValueError("beep intervals must be positive and ordered")
        if self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive")


class SpatialCueRenderer:
    """Accumulate a stereo cue track; write it once at shutdown."""

    def __init__(self, config: SpatialCueConfig | None = None) -> None:
        self.config = config or SpatialCueConfig()
        self._capacity = int(self.config.sample_rate_hz * min(self.config.max_seconds, 10.0))
        self._buffer = np.zeros((self._capacity, 2), dtype=np.float64)
        self._elapsed_s = 0.0
        self._written = 0
        self._until_next_beep_s = 0.0
        self._reserved_until = 0
        self._latched = False
        self.beeps = 0
        self.events: list[str] = []

    @property
    def _cursor(self) -> int:
        # Derived from elapsed time so per-tick rounding never accumulates.
        return round(self._elapsed_s * self.config.sample_rate_hz)

    @property
    def seconds(self) -> float:
        return max(self._cursor, self._written) / self.config.sample_rate_hz

    @property
    def truncated(self) -> bool:
        return self._cursor >= int(self.config.max_seconds * self.config.sample_rate_hz)

    def update(self, *, dt_s: float, bearing_rad: float, distance_m: float) -> bool:
        """Advance the track by ``dt_s`` and emit a beep when one is due.

        Ranging beeps stop once proximity latches: the robot is stationary, the
        geometry no longer changes, and the operator has already heard the
        arrival chime. Continuing to beep would add nothing but noise.
        """

        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        if self._latched:
            self._elapsed_s += dt_s
            return False
        emitted = False
        self._until_next_beep_s -= dt_s
        if self._until_next_beep_s <= 0.0:
            self._mix_directional_beep(bearing_rad, distance_m)
            self._until_next_beep_s = self._beep_interval_s(distance_m)
            self.beeps += 1
            emitted = True
        self._elapsed_s += dt_s
        return emitted

    def mark_proximity_latched(self) -> None:
        self._latched = True
        start = self._reserve(self.config.latch_seconds)
        self._mix_tone(start, self.config.latch_tone_hz, self.config.latch_seconds, 0.0, 0.9)
        self.events.append("proximity_latched")

    def mark_call_submitted(self) -> None:
        low, high = self.config.call_tone_hz
        start = self._reserve(2.0 * self.config.call_seconds)
        step = int(self.config.call_seconds * self.config.sample_rate_hz)
        self._mix_tone(start, low, self.config.call_seconds, 0.0, 0.85)
        self._mix_tone(start + step, high, self.config.call_seconds, 0.0, 0.85)
        self.events.append("call_submitted")

    def _reserve(self, seconds: float) -> int:
        """Place an event tone after any earlier one so cues never overlap."""

        start = max(self._cursor, self._reserved_until)
        self._reserved_until = start + int(seconds * self.config.sample_rate_hz)
        return start

    def _beep_interval_s(self, distance_m: float) -> float:
        interval = self.config.near_interval_s + self.config.interval_slope_s_per_m * max(
            distance_m, 0.0
        )
        return float(min(max(interval, self.config.near_interval_s), self.config.far_interval_s))

    def _distance_gain(self, distance_m: float) -> float:
        gain = 1.0 / (1.0 + max(distance_m, 0.0) / self.config.reference_distance_m)
        return float(max(gain, self.config.minimum_gain))

    def _mix_directional_beep(self, bearing_rad: float, distance_m: float) -> None:
        bearing = _wrap_angle(bearing_rad)
        tone = self.config.tone_hz
        if math.cos(bearing) < 0.0:
            tone *= self.config.behind_tone_ratio
        self._mix_tone(
            self._cursor,
            tone,
            self.config.beep_seconds,
            math.sin(bearing),
            self._distance_gain(distance_m),
        )

    def _mix_tone(
        self, start_sample: int, frequency_hz: float, seconds: float, pan: float, gain: float
    ) -> None:
        """Mix one tone at ``pan`` (-1 right .. +1 left) using equal power and ITD."""

        rate = self.config.sample_rate_hz
        count = max(int(seconds * rate), 1)
        times = np.arange(count, dtype=np.float64) / rate
        envelope = self._envelope(count)
        mono = gain * envelope * np.sin(2.0 * math.pi * frequency_hz * times)

        pan = float(np.clip(pan, -1.0, 1.0))
        angle = (pan + 1.0) * (math.pi / 4.0)  # -1 -> 0 rad (right), +1 -> pi/2 (left)
        left_gain = math.sin(angle)
        right_gain = math.cos(angle)
        lead = round(abs(pan) * self.config.max_itd_s * rate)
        left_start = start_sample + (0 if pan >= 0 else lead)
        right_start = start_sample + (lead if pan >= 0 else 0)

        self._mix_channel(left_start, mono * left_gain, 0)
        self._mix_channel(right_start, mono * right_gain, 1)

    def _envelope(self, count: int) -> np.ndarray:
        fade = min(int(self.config.fade_seconds * self.config.sample_rate_hz), count // 2)
        envelope = np.ones(count, dtype=np.float64)
        if fade > 0:
            ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, fade)))
            envelope[:fade] = ramp
            envelope[count - fade :] = ramp[::-1]
        return envelope

    def _mix_channel(self, start_sample: int, samples: np.ndarray, channel: int) -> None:
        limit = int(self.config.max_seconds * self.config.sample_rate_hz)
        if start_sample >= limit:
            return
        end = min(start_sample + samples.size, limit)
        self._ensure_capacity(end)
        self._buffer[start_sample:end, channel] += samples[: end - start_sample]
        self._written = max(self._written, end)

    def _ensure_capacity(self, required: int) -> None:
        if required <= self._capacity:
            return
        capacity = self._capacity
        while capacity < required:
            capacity *= 2
        limit = int(self.config.max_seconds * self.config.sample_rate_hz)
        capacity = min(capacity, limit)
        grown = np.zeros((capacity, 2), dtype=np.float64)
        grown[: self._capacity] = self._buffer
        self._buffer = grown
        self._capacity = capacity

    def samples(self) -> np.ndarray:
        """Return the rendered stereo track clipped to [-1, 1]."""

        end = max(self._written, self._cursor)
        return np.clip(self._buffer[:end], -1.0, 1.0)

    def write_wav(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pcm = np.round(self.samples() * 32767.0).astype("<i2")
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(2)
            stream.setsampwidth(2)
            stream.setframerate(self.config.sample_rate_hz)
            stream.writeframes(pcm.tobytes())
        return path


@dataclass(frozen=True)
class SpatialAudioSettings:
    """One switch set shared by the MuJoCo and Isaac rescue controllers."""

    #: Steer the approach from the simulated array instead of ground-truth xy.
    acoustic_localization: bool = False
    #: Render the stereo operator cue and write it at shutdown.
    render_cue: bool = False
    output_path: Path = Path("runtime/everest-g1-rescue.wav")
    sensor: AcousticSensorConfig | None = None
    cue: SpatialCueConfig | None = None

    @property
    def enabled(self) -> bool:
        return self.acoustic_localization or self.render_cue

    def path_for(self, simulation_id: str) -> Path:
        path = Path(self.output_path)
        if not simulation_id:
            return path
        return path.with_name(f"{path.stem}-{simulation_id}{path.suffix}")


class RescueAudio:
    """Bundle the array sensor and the operator cue for one rescue run."""

    def __init__(self, settings: SpatialAudioSettings) -> None:
        self.settings = settings
        self.sensor = (
            AcousticBeaconSensor(settings.sensor) if settings.acoustic_localization else None
        )
        self.renderer = SpatialCueRenderer(settings.cue) if settings.render_cue else None
        self.last_estimate: BearingEstimate | None = None
        self.last_bearing_rad: float | None = None

    @property
    def enabled(self) -> bool:
        return self.sensor is not None or self.renderer is not None

    def observe(
        self,
        *,
        robot_xy: tuple[float, float],
        robot_yaw_rad: float,
        source_xy: tuple[float, float],
    ) -> BearingEstimate | None:
        """Return the array's belief about the source, or None when disabled."""

        if self.sensor is None:
            return None
        self.last_estimate = self.sensor.sense(
            robot_xy=robot_xy, robot_yaw_rad=robot_yaw_rad, source_xy=source_xy
        )
        return self.last_estimate

    def cue(self, *, dt_s: float, bearing_rad: float, distance_m: float) -> None:
        self.last_bearing_rad = _wrap_angle(bearing_rad)
        if self.renderer is not None:
            self.renderer.update(
                dt_s=dt_s, bearing_rad=self.last_bearing_rad, distance_m=distance_m
            )

    def mark_proximity_latched(self) -> None:
        if self.renderer is not None:
            self.renderer.mark_proximity_latched()

    def mark_call_submitted(self) -> None:
        if self.renderer is not None:
            self.renderer.mark_call_submitted()

    def reset(self) -> None:
        if self.sensor is not None:
            self.sensor.reset()

    def close(self, simulation_id: str = "") -> dict[str, object] | None:
        """Write the cue track and return redacted summary fields for the audit log."""

        if self.renderer is None:
            return None
        path = self.settings.path_for(simulation_id)
        self.renderer.write_wav(path)
        return {
            "path": str(path),
            "seconds": round(self.renderer.seconds, 3),
            "beeps": self.renderer.beeps,
            "events": list(self.renderer.events),
            "truncated": self.renderer.truncated,
        }
