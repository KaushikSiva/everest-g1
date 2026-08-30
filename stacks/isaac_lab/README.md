# Isaac Lab-Arena runtime

This is the runnable rescue lane. It uses the pinned Arena G1 AGILE controller,
the built-in 640×480 `robot_head_cam`, the proximity latch, and BeaconCall.

```bash
./stacks/isaac_lab/setup.sh
./stacks/isaac_lab/run.sh
```

The first run is disarmed and cannot call. Follow the root runbook before using
`./stacks/isaac_lab/run.sh --arm-live-call`.
