# Spatial audio across the four MuJoCo modes

Everest G1 renders the casualty direction as sound in two independent ways.
The raw CLI flags are optional and neither can place a call. The four macOS
launchers enable both by default so controller, rescue, carry, and scan runs
produce comparable audio evidence.

1. **Acoustic localization** (`--acoustic-localization`) simulates a torso
   microphone array and recovers the *bearing* of the downed person from time
   differences of arrival. It steers only the final rescue/carry approach.
2. **The operator cue** (`--spatial-audio`) renders a stereo track in which pan
   and interaural delay follow bearing, repetition rate and gain follow
   distance, and the proximity latch and call submission each get their own
   tone. It is written to a WAV file at shutdown.

Everything lives in `src/everest_g1/spatial_audio.py` and is shared by the
MuJoCo modes, the Isaac Lab-Arena rescue policy, and `everest-g1 dry-run`.

| Mode | Bearing use | Motion authority |
| --- | --- | --- |
| PlayStation controller | Stereo cue and audit telemetry | None; joystick remains authoritative |
| Gemini rescue | Initial Gemini context, stereo cue, final approach steering | Bearing steers; geometric range still gates |
| Gemini carry | Same as rescue until the simulated pickup | Bearing steers only before pickup |
| Gemini scan | Initial Gemini route context and continuous stereo cue | None; validated Gemini route remains authoritative |

## Authority: audio steers, range gates

This is the rule for rescue and carry, the only modes where audio can steer.

```text
microphone array --> bearing ------+
                                   |
                                   v
measured surface distance --> steering target --> bounded velocity command
        |
        +--> proximity latch --> stop --> BeaconCall
```

The bearing is a noisy estimate and is allowed to be wrong. It only ever picks
a *direction*. The distance used to build the steering target is the same
measured surface distance that gates the stop, so a bad bearing can steer the
robot badly but can never fake arrival, stop the robot early, or release a
call. `everest_g1.rescue.target_from_bearing` is the only supported way to turn
a bearing into a target, and it takes the measured range as an argument.

`BearingEstimate.range_m` is a coarse level-based figure kept for telemetry. It
is deliberately not used anywhere in the control path. Gemini receives it
under the explicit name `coarse_range_m_telemetry_only`.

## The array

Four microphones on a 0.12 m square (`DEFAULT_MIC_OFFSETS_M`), body frame,
x forward and y left. Three non-collinear microphones are the minimum that
separates a source ahead from one behind; four keeps bearing error independent
of which way the robot faces. `MicArray` rejects fewer than three microphones
and rejects collinear geometry outright.

Bearing is recovered by matching measured time differences against the
plane-wave delays of 720 candidate directions and taking the best fit. The
residual at the best fit, normalised by the array's aperture delay, becomes
`confidence`. Estimates are smoothed as unit vectors, so the value never wraps
badly at ±π.

The noise model is a configurable per-microphone arrival-time jitter
(`tdoa_jitter_s`, default 8 µs) and a multiplicative level error. The default
jitter is a plausible GCC-PHAT residual for a 48 kHz array with interpolation.
It is a simulation parameter, not a claim about any particular hardware.

## The cue

| Quantity | Mapping |
| --- | --- |
| Bearing | Equal-power pan plus up to 0.65 ms interaural delay toward the source |
| Behind the robot | Tone drops an octave, because level differences alone cannot tell front from back |
| Distance | Beep interval 0.10 s (touching) to 0.75 s (far); gain falls off with range |
| Proximity latched | One 0.6 s centred 880 Hz chime |
| Call submitted | Two centred tones, 660 Hz then 990 Hz |

The simulated array is sampled at 50 Hz even though MuJoCo physics runs at
500 Hz. Rendering is buffer accumulation and a few hundred samples of
arithmetic per beep, so it needs no audio device and works
headless and in CI. The buffer is bounded by `max_seconds` (default 300 s); a
run that exceeds it reports `"truncated": true`. The WAV is 16-bit stereo,
written once during `close()`.

Event tones are sequenced rather than stacked: the call tone is placed after
the latch chime finishes, so the two are always distinguishable.

## Running it

No GPU or MuJoCo needed:

```bash
make rescue-audio
# or
uv run everest-g1 dry-run --spatial-audio --acoustic-localization
```

The JSON summary gains a `spatial_audio` block naming the written file, its
duration, the beep count, and which event tones fired.

In MuJoCo:

```bash
make mujoco-rescue-audio
# or
uv run summit-sentinel --mode headless --seconds 12 --rescue --json \
  --spatial-audio --acoustic-localization
```

The normal macOS entry points already include those flags:

```bash
./autonomy/run_controller.sh
./autonomy/run_rescue.sh
./autonomy/run_carry.sh
./autonomy/run_scan.sh
```

Set `EVEREST_DISABLE_SPATIAL_AUDIO=1` only when you deliberately want a silent
four-mode run. The cue is written at shutdown; it is not played through the
Mac speakers during simulation.

In Isaac Lab-Arena, the policy configuration exposes `acoustic_localization`,
`spatial_audio`, and `spatial_audio_out` through Arena's generated policy CLI.

The output path has the simulation id appended to its stem, so concurrent runs
never overwrite each other:
`runtime/everest-g1-rescue.wav` becomes `runtime/everest-g1-rescue-<id>.wav`.
`runtime/` is gitignored.

## Audit events

Two redacted events join the existing JSONL stream when audio is enabled:

* `spatial_audio_started` — which halves are on.
* `spatial_audio_written` — output path, duration, beep count, event tones, and
  whether the buffer was truncated.

Like every other event in that log, they contain no credentials and no
destination phone number. With audio disabled the event stream is byte-for-byte
what it was before.

## What this is not

The cue is an after-run monitoring aid for a person watching a simulation. It is not a
localisation guarantee, not a hearing-based safety interlock, and not evidence
that a physical G1 could find a person by sound. The proximity gate, the stop,
and the one-call-per-process rule are unchanged and remain the only things that
decide whether BeaconCall is contacted.
