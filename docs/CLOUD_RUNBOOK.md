# Cloud runbook

For the simplified three-folder entrypoints and camera-grounded call flow, use
[`ISAAC_LAB_GROOT_SONIC.md`](ISAAC_LAB_GROOT_SONIC.md). This document retains
the lower-level cloud details.

## 1. Brev commissioning

Provision Ubuntu x86_64 with an NVIDIA GPU, persistent disk, and enough space
for Isaac Sim/Arena. From a terminal on that instance:

```bash
git clone https://github.com/KaushikSiva/everest-g1.git
cd everest-g1
./cloud/brev_setup.sh
```

The setup validates platform/GPU, checks out the exact Arena SHA, resolves its
Isaac Lab and Isaac-GR00T submodules, installs Arena's wheel environment, and
imports `isaaclab`, `isaaclab_arena`, and `everest_g1`.

Run disarmed first:

```bash
./cloud/run_brev_rescue.sh
```

The wrapper enables Arena's G1 head camera automatically. The disarmed run does
not encode, upload, or call with a frame.

Acceptance:

- G1 starts upright and approaches the person proxy without exceeding bounds;
- velocity becomes zero at touching distance;
- `runtime/everest-g1-events.jsonl` ends in
  `proximity_reached_call_disarmed`;
- no outbound call occurs.

## 2. BeaconCall integration

Deploy/start BeaconCall separately and configure its LiveKit URL/key/secret, SIP
trunk ID, server-side destination number, and API token. Test BeaconCall's own
endpoint before coupling it to simulation.

On Brev:

```bash
export BEACON_API_URL=https://YOUR-BEACON-HOST
read -rsp 'Beacon API token: ' BEACON_API_TOKEN; export BEACON_API_TOKEN; echo
./cloud/run_brev_rescue.sh --arm-live-call
```

Type `ARM-LIVE-CALL` only when the recipient expects the simulation call.
Acceptance adds exactly one `call_queued` and one `call_dispatched` event, plus a
BeaconCall incident ending in `acknowledged` or `timed_out`.

## 3. Custom FBX

Convert only an asset you may legally use:

```bash
mkdir -p runtime/isaac_assets
blender --background --python scripts/convert_person_fbx.py -- \
  /absolute/input/person.fbx runtime/isaac_assets/downed_person.usd
```

Invoke Arena directly to pass the generated path:

```bash
arena=/home/ubuntu/workspace/everest-g1-cloud/IsaacLab-Arena
export PYTHONPATH="$PWD/src:$arena"
cd "$arena"
.venv/bin/python -m everest_g1.isaac.run \
  --viz kit --policy_type everest_approach --num_envs 1 --num_steps 3000 \
  --person_usd /absolute/path/to/runtime/isaac_assets/downed_person.usd \
  everest_g1_rescue
```

Keep the generated asset out of git.

## 4. Modal headless probe

Revoke any Modal token exposed in chat before logging in again. Configure Modal
with its standard CLI flow, then:

```bash
python -m pip install 'modal>=1.1,<2'
modal run cloud/modal_app.py --steps 500
```

The app requests an L40S, imports the pinned environment, and only then runs a
disarmed headless evaluation. Modal is not the interactive viewer lane. If the
renderer or container runtime is incompatible, retain the probe output and use
Brev rather than changing the simulation checks.

## 5. Cleanup

- Stop live cloud jobs and BeaconCall workers.
- Remove exported shell secrets: `unset BEACON_API_TOKEN EVEREST_ARM_LIVE_CALL`.
- Keep only redacted logs, videos, digests, and checkpoint metadata needed for evidence.
- Do not upload unlicensed FBX/USD assets.
