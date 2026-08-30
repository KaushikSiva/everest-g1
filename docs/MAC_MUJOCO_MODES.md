# macOS MuJoCo: controller + three Gemini ER 2 modes

These modes run locally on a Mac. Gemini Robotics ER 2 receives one G1 front
camera frame and a structured table for every candidate route. Each waypoint
includes measured/generated slope, temperature, wind, visibility, snow depth,
effective friction, distance, and aggregate risk. Gemini may select only a
named route. Local code rejects an unknown route or any route outside the hard
safety envelope before MuJoCo advances.

Gemini is a high-level planner here. It never writes joint targets, torques, or
raw velocity commands. The local 500 Hz MuJoCo loop and bundled locomotion
policy execute the selected path.

All four launchers also enable the simulated microphone array and stereo cue.
Gemini receives the initial bearing/confidence alongside the camera and terrain
table. Rescue/carry may use that bearing to steer the final approach; controller
and scan keep it passive so it cannot override joystick or route authority.

## 1. One-time setup

```bash
cd /path/to/everest-g1
./autonomy/setup_mac.sh
```

Create a new restricted Gemini API key. Do not put it in Git. The autonomous
launchers securely prompt for it if `GEMINI_API_KEY` is not already set.

The API key previously pasted into chat should be considered exposed and
rotated before use.

For controller mode only, calibrate the DualSense once if the profile is
missing:

```bash
uv run summit-sentinel --calibrate-joystick runtime/dualsense.json --joystick-index 0
```

## 2. Run one of four modes

```bash
./autonomy/run_controller.sh
./autonomy/run_rescue.sh
./autonomy/run_carry.sh
./autonomy/run_scan.sh
```

Equivalent Make targets are `make mode-controller`,
`make mode-gemini-rescue`, `make mode-gemini-carry`, and
`make mode-gemini-scan`.

The carry mode is deliberately labeled a MuJoCo visual carry proxy. The current
12-actuator G1 XML controls the legs only; its arm meshes are fixed. It can
demonstrate approach, proximity, attachment, and locomotion state transitions,
but not validate a physical grasp.

Each run writes a 16-bit stereo cue to
`runtime/everest-g1-rescue-<simulation-id>.wav` when the viewer closes. The cue
encodes casualty bearing, distance, proximity, and call submission. It is not
live speaker playback. The launcher prints the exact path and an `afplay`
command after shutdown. To deliberately disable it:

```bash
EVEREST_DISABLE_SPATIAL_AUDIO=1 ./autonomy/run_scan.sh
```

## 3. Change weather/terrain assumptions

All autonomous launchers accept the planner/environment flags after the script:

```bash
./autonomy/run_scan.sh \
  --temperature-c -28 \
  --wind-mps 14 \
  --visibility-m 300 \
  --snow-depth-m 0.24 \
  --terrain-friction 0.70
```

These values are validated. Friction and snow affect MuJoCo's terrain friction;
all values, including local slope samples from the checked heightfield, affect
the route table Gemini receives. The JSONL audit log records the selected route
and factor names at `runtime/everest-g1-events.jsonl` without credentials.

For a network-free deterministic smoke test, call the underlying CLI with
`--offline-plan`. The regular autonomous launchers intentionally require a
Gemini key so they cannot silently claim Gemini was used.

## 4. Optional LiveKit/Twilio call

Rescue and carry reuse the existing one-shot BeaconCall handoff. Keep it
disarmed for normal simulation. To deliberately arm it:

```bash
export BEACON_API_URL=https://YOUR-BEACON-HOST
read -rsp 'Beacon API token: ' BEACON_API_TOKEN; export BEACON_API_TOKEN; echo
export EVEREST_ARM_LIVE_CALL=ARM-LIVE-CALL
./autonomy/run_rescue.sh
```

BeaconCall—not this simulator—owns the destination phone number and the
LiveKit/Twilio credentials. Unset the variables after the run:

```bash
unset BEACON_API_TOKEN EVEREST_ARM_LIVE_CALL GEMINI_API_KEY
```

## 5. Logs and safety behavior

Planning and mission transitions append to
`runtime/everest-g1-events.jsonl`. Closing the viewer stops the process. An
invalid Gemini response, missing camera image, missing key, unsafe route, fall,
or non-finite control value fails closed; there is no automatic fallback from
Gemini to an unreported planner. Use `--offline-plan` explicitly when wanted.

See [Spatial audio](SPATIAL_AUDIO.md) for the microphone geometry, stereo
mapping, and per-mode authority boundaries.
