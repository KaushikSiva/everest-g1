# Architecture

## Control and network planes

```text
CONTROL PLANE (50 Hz, local GPU process)

Isaac physics -> G1 state -> bounded approach -> proximity dwell -> zero velocity
                                  |                    |
                                  |                    +-> immutable reached latch
                                  +-> AGILE WBC action

NETWORK PLANE (non-real-time)

reached latch -> one-slot thread -> authenticated BeaconCall incident
                                      -> LiveKit dispatch
                                      -> Twilio SIP participant
                                      -> deterministic voice acknowledgment

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

The control loop enqueues an immutable `RescueObservation` into a one-slot
worker. HTTP, DNS, TLS, LiveKit, Twilio, speech, and acknowledgment handling all
happen outside the robot policy. BeaconCall owns:

- destination number;
- API authentication and persistent idempotency;
- LiveKit agent dispatch before SIP participant creation;
- deterministic observable-facts-only wording;
- acknowledgment detection, timeout, and room deletion.

## NVIDIA lanes

| Lane | Purpose | Controller | Isaac Lab |
| --- | --- | --- | --- |
| Arena commissioning | Verify scene, bounds, stop, and trigger | G1 AGILE decoupled WBC | 3.0.0 beta 2 |
| GR00T + SONIC | Learned whole-body task policy after data/checkpoint exists | GR00T N1.7 latent actions decoded by GEAR-SONIC | 2.3.2 |
| MuJoCo fallback | Local UI/control regression | vendored Unitree policy | N/A |

The lanes share scenario intent and measured acceptance criteria, not a Python
environment or an assertion of numerical equivalence.

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
