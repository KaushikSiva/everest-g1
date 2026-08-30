# Architecture

## Sensor-to-action mission architecture

```text
                              G1 SENSORS
                         /        |        \
                     Camera      Audio      IMU
                        |          |         |
                 front-camera   four-mic   body orientation
                    frame        array     + terrain state
                        |          |         |
                 camera capture  spatial_audio.py
                         \         |        /
                              WORLD MODEL
             frame + acoustic bearing/confidence + terrain/weather
                                   |
                    AI MISSION AGENT (Gemini Robotics-ER 2)
                                   |
          "I hear a distress call uphill. Locate and reach the person
                         using a safe route."
                                   |
                                 GR00T
                    task-conditioned latent actions
                                   |
                                 SONIC
                       50 Hz whole-body decoder
                                   |
                                   G1
```

This is a layered authority model, not one end-to-end cloud control loop:

1. Camera, simulated microphone-array output, body pose, and terrain/environment
   samples form a compact world-model packet.
2. Gemini Robotics-ER 2 runs at the mission layer. It receives one bounded
   camera frame, acoustic bearing/confidence, and scored route candidates. It
   returns one route ID plus rationale and observations.
3. The local planner rejects unknown or unsafe route IDs. Gemini cannot create
   arbitrary waypoints, write torques, satisfy proximity, or arm a call.
4. GR00T/SONIC is the promoted learned-control lane: GR00T predicts the
   task-conditioned SONIC representation and SONIC decodes it at 50 Hz.
5. Until a trained checkpoint passes the promotion gate, the runnable paths use
   the commissioned local policy and deterministic route executor instead.

The quoted mission sentence is an illustrative task instruction, not a speech
transcript. The current acoustic model estimates direction and confidence from
time differences of arrival; it does not recognize words. Its coarse range is
telemetry only.

## Control and network planes

```text
CONTROL PLANE (50 Hz, local GPU process)

Isaac/MuJoCo physics -> G1 state -> bounded approach -> proximity dwell -> zero velocity
                                        |                    |
                                        |                    +-> immutable reached latch
                                        +-> local locomotion policy action

NETWORK PLANE (non-real-time)

reached latch -> one robot-camera JPEG -> one-slot thread
                                      -> authenticated BeaconCall incident
                                      -> OpenAI observable description
                                      -> LiveKit dispatch
                                      -> Twilio SIP participant
                                      -> bounded voice acknowledgment

Bright Data MCP -> optional public context artifact (no edge to control output)
```

The policy writes the current joint pose into the 43 direct-joint action fields
and uses only Arena's final seven WBC fields for navigation, pelvis height, and
torso orientation. Commands are clamped before they enter the action tensor.

## Distance rule

The policy uses horizontal surface distance:

```text
max(0, center_distance - robot_proxy_radius - person_proxy_radius)
```

It commands zero velocity at 0.15 m or less, then requires the condition for 0.25
continuous seconds. A transient threshold crossing resets dwell. Once latched,
the process never resumes motion and never queues a second call.

## Spatial audio authority

Optional acoustic localization and the optional stereo operator cue sit beside
the control loop, never inside the gate:

```text
microphone array -> bearing ------+
                                  v
measured surface distance -> steering target -> bounded velocity command
        |
        +-> proximity dwell -> stop -> one-slot call worker
```

A bearing is a noisy estimate and only ever selects a direction. The range used
to build the steering target is the same measured surface distance from the
rule above, so a wrong bearing can steer badly and can never satisfy the
threshold, shorten the dwell, or release a call. The level-based range the array
reports is telemetry only. The cue is rendered into a bounded buffer and written
once at shutdown, so it never performs I/O inside the policy step.

That steering edge exists only during the rescue/carry final approach. In
controller and scan modes the microphone array is a passive cue/telemetry
source. Autonomous modes also give Gemini the initial bearing and confidence,
but the coarse audio range remains explicitly telemetry-only.

Full mapping and noise model: [Spatial audio](SPATIAL_AUDIO.md).

## Call behavior

At the first armed MuJoCo latch, the simulator renders one 640×480 JPEG from the
robot-relative `g1_front_camera`, then enqueues an immutable
`RescueObservation` into a one-slot worker. The disarmed path does not render or
send an evidence frame. HTTP, DNS, TLS, OpenAI, LiveKit, Twilio, speech, and
acknowledgment handling all happen outside the robot policy. BeaconCall owns:

- destination number;
- API authentication and persistent idempotency;
- one-frame OpenAI analysis before voice dispatch;
- LiveKit agent dispatch before SIP participant creation;
- deterministic base wording plus bounded observable camera context;
- acknowledgment detection, timeout, and room deletion.

## NVIDIA lanes

| Lane | Purpose | Controller | Isaac Lab |
| --- | --- | --- | --- |
| Arena commissioning | Verify scene, bounds, stop, and trigger | G1 AGILE decoupled WBC | 3.0.0 beta 2 |
| GR00T + SONIC | Learned whole-body task policy after data/checkpoint exists | GR00T N1.7 latent actions decoded by GEAR-SONIC | 2.3.2 |
| MuJoCo rescue | Mac-local scenario and call integration | vendored Unitree locomotion policy | N/A |

The lanes share scenario intent and measured acceptance criteria, not a Python
environment or an assertion of numerical equivalence.

Both simulation lanes use the dependency-free `ApproachLimits`,
`ProximityLatch`, `RescueObservation`, and asynchronous `BeaconCallWorker`.
MuJoCo reads the named `downed_person_target` site, computes the robot yaw from
its free-joint quaternion, and replaces manual velocity input only while
`--rescue` is active. Its head-mounted camera is part of the MuJoCo model and
moves with the G1. Local joystick emergency-stop handling still runs first.

## Acceptance evidence

A qualifying Isaac run should retain:

- pinned upstream SHAs;
- GPU and driver identity;
- scene/person asset digests and licenses;
- control bounds and threshold/dwell values;
- video or viewport recording;
- JSONL events showing start, proximity, and call status;
- BeaconCall incident status through acknowledgment or timeout.

That evidence measures the remaining sim-to-real gap. It does not erase it.
