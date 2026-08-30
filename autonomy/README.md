# Mac MuJoCo modes

This folder is the macOS-only control/autonomy lane. It keeps Gemini Robotics
ER 2 at the route-planning layer and MuJoCo plus the local locomotion policy at
the control layer.

The four launchers are:

1. `run_controller.sh` — existing calibrated PlayStation controller.
2. `run_rescue.sh` — Gemini selects a bounded route, the G1 approaches and
   stops at the casualty, and the existing optional BeaconCall gate can run.
3. `run_carry.sh` — the same approach followed by a **simulation-only visual
   carry proxy** and a short bounded walk. The 12-actuator MuJoCo model has no
   arm joints, so this is not a grasp-policy claim.
4. `run_scan.sh` — Gemini compares slope, temperature, wind, visibility, snow,
   effective friction, and distance for complete survey routes. The G1 patrols
   the selected route in both directions.

See [`docs/MAC_MUJOCO_MODES.md`](../docs/MAC_MUJOCO_MODES.md) for setup,
BeaconCall arming, factor overrides, and exact commands.
