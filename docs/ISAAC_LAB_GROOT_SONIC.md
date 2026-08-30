# Isaac Lab + GR00T + SONIC: one setup path

This is the start-to-finish runbook for the three NVIDIA lanes in this
repository. They share the Everest rescue scenario and pinned versions, but
they deliberately use separate folders and Python environments.

## What works today

- **Isaac Lab-Arena:** runnable G1 rescue commissioning controller, proximity
  dwell, built-in G1 head camera, and guarded BeaconCall handoff.
- **GR00T N1.7:** pinned data/fine-tuning/inference checkout for producing
  task-specific latent actions after an Everest dataset exists.
- **GEAR-SONIC:** pinned 50 Hz whole-body controller checkout and isolated
  MuJoCo/training environments.

The runnable Isaac rescue uses Arena's AGILE G1 WBC. GR00T plus SONIC is a
separate learned-policy promotion lane; it is not silently substituted for the
commissioning controller.

Gemini Robotics-ER 2 is a higher-level mission reasoner used by the Mac MuJoCo
autonomous modes. It fuses the initial camera observation, acoustic
bearing/confidence, and terrain/environment factors to select one locally
generated hard-safe route. It does not replace GR00T, SONIC, or the local
controller. The intended promoted layering is:

```text
G1 camera + acoustic bearing + IMU/terrain -> world model
  -> Gemini Robotics-ER 2 mission route
  -> GR00T task-conditioned latent action representation
  -> SONIC 50 Hz whole-body decoder
  -> locally bounded G1 control
```

This runbook installs the NVIDIA execution lanes. It does not claim that a
Gemini-selected simulation route is already a physical-robot policy.

```text
stacks/
  isaac_lab/   runnable simulation + G1 head camera + BeaconCall
  groot/       GR00T N1.7 checkout and Python 3.12 environment
  sonic/       GEAR-SONIC checkout and its own simulation/training environments
```

Upstream source and environments live outside the repository under
`/home/ubuntu/workspace/everest-g1-cloud`. This keeps the Git checkout small and
prevents Isaac Lab 3.0 beta, GR00T, and SONIC's Isaac Lab 2.3.2 dependencies
from overwriting each other.

## 1. Start on an NVIDIA Linux machine

Use an Ubuntu x86_64 Brev instance with an NVIDIA RTX-class GPU, persistent
storage, `git`, Git LFS, and `uv`.

```bash
git clone https://github.com/KaushikSiva/everest-g1.git
cd everest-g1
```

Do not put API keys or Hugging Face tokens in this repository.

## 2. Run Isaac Lab safely first

Setup is one command:

```bash
make isaac-setup
```

It clones the pinned Isaac Lab-Arena checkout, verifies the exact Isaac Lab and
Isaac-GR00T submodule SHAs, creates Arena's environment, installs this project
editable, installs the JPEG encoder, and verifies imports.

Run the disarmed simulation:

```bash
make isaac-run
```

The wrapper automatically supplies `--enable_cameras`. Arena's existing G1
`robot_head_cam` produces a 640×480 RGB observation under
`camera_obs.robot_head_cam_rgb`. The disarmed run renders the camera but never
sends a frame and cannot place a call.

Expected acceptance:

- the G1 starts upright and approaches the downed-person proxy;
- velocity becomes zero at the 0.15 m surface-distance threshold;
- the threshold remains true for a continuous 0.25 seconds;
- `runtime/everest-g1-events.jsonl` ends in
  `proximity_reached_call_disarmed`;
- no outbound call occurs.

## 3. Test one camera-grounded call

Do this only when the recipient expects a simulation call. BeaconCall must
already be deployed with OpenAI, LiveKit, Twilio, and the destination number
configured server-side.

```bash
export BEACON_API_URL=https://beacon-call.onrender.com
read -rsp 'Beacon API token: ' BEACON_API_TOKEN
export BEACON_API_TOKEN
echo
./stacks/isaac_lab/run.sh --arm-live-call
```

Type `ARM-LIVE-CALL` at the final gate. At the first completed proximity dwell:

1. robot velocity remains zero;
2. exactly one G1 head-camera frame is encoded as a bounded JPEG;
3. the frame, measured distance, Bearer token, and idempotency key go to
   BeaconCall on a worker thread;
4. BeaconCall asks OpenAI for an observable description before LiveKit
   dispatch;
5. the call speaks the existing simulation warning plus that visual
   description and waits for acknowledgment.

The image is never sent to LiveKit or Twilio. If camera capture or OpenAI
analysis fails, the original proximity report still proceeds and does not
invent visual facts.

Expected log sequence:

```text
simulation_started
front_camera_captured
call_queued
call_dispatched
```

`front_camera_capture_failed` followed by `call_queued` is the intended safe
fallback. The JSONL log records only byte count/status—not the image, API token,
or destination number.

## 4. Setup GR00T N1.7 separately

```bash
make groot-setup
```

This creates `/home/ubuntu/workspace/everest-g1-cloud/Isaac-GR00T`, checks out
the pinned N1.7 commit with submodules/LFS, runs its own Python 3.12 `uv sync`,
and verifies `import gr00t`.

Before model access, accept the NVIDIA/Hugging Face model terms and authenticate
inside that checkout. Store demonstrations in an external dataset directory,
not in this Git repository. The Everest dataset must contain synchronized
front-camera video, robot state, actions, task text, and the GR00T LeRobot
metadata for `UNITREE_G1_SONIC`.

The bounded Modal fine-tuning entrypoint remains available:

```bash
modal run cloud/modal_app.py::finetune_groot_n17_sonic \
  --dataset-name everest_rescue \
  --output-name everest_rescue_n17 \
  --max-steps 2000
```

This produces a candidate checkpoint; it does not automatically control the
commissioning simulation.

## 5. Setup GEAR-SONIC separately

For the upstream isolated SONIC MuJoCo environment:

```bash
make sonic-setup
```

For an Isaac Lab 2.3.2 training job where Modal supplies the environment:

```bash
./stacks/sonic/setup.sh --checkout-only
```

Then use the existing Modal training function with dataset/model volume names:

```bash
modal run cloud/modal_app.py::train_sonic_controller \
  --robot-motion-name everest_robot_motion \
  --smpl-motion-name everest_smpl_motion \
  --output-name everest_sonic \
  --iterations 100
```

SONIC decodes the 64-dimensional latent motion representation at 50 Hz. Keep
its Isaac Lab 2.3.2 environment separate from Arena's Isaac Lab 3.0 beta 2
environment.

## 6. Promotion gate

Do not replace `everest_approach` with a GR00T/SONIC checkpoint until it passes:

1. dataset schema and image/action synchronization checks;
2. open-loop GR00T evaluation;
3. headless SONIC checkpoint evaluation;
4. rendered Isaac collision and camera review;
5. the existing speed, stop-distance, dwell, one-call, and fallback tests;
6. repeated seeds and perturbations with retained logs;
7. a separate physical-robot safety review.

This repository does not claim that installing all three lanes eliminates the
simulation-to-reality gap.

## Cleanup

```bash
unset BEACON_API_TOKEN EVEREST_ARM_LIVE_CALL HF_TOKEN
```

Stop cloud jobs when finished. Keep only redacted event logs, approved videos,
dataset manifests, model digests, and evaluation results.

## Primary references

- [Isaac Lab: adding sensors to a robot](https://isaac-sim.github.io/IsaacLab/develop/source/tutorials/04_sensors/add_sensors_on_robot.html)
- [Isaac Lab-Arena](https://github.com/isaac-sim/IsaacLab-Arena)
- [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- [NVIDIA GR00T-WholeBodyControl / GEAR-SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl)
- [BeaconCall](https://github.com/KaushikSiva/beacon-call)
