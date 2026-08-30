# Architecture

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
