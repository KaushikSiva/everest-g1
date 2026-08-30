# Isolated NVIDIA stacks

Each folder contains this repository's small, pinned integration layer. The
large NVIDIA upstream repositories and their virtual environments are fetched
outside this checkout under `/home/ubuntu/workspace/everest-g1-cloud`.

- `isaac_lab/`: runnable G1 rescue simulation and front-camera BeaconCall path.
- `groot/`: GR00T N1.7 data, fine-tuning, and inference environment.
- `sonic/`: GEAR-SONIC controller checkout and simulation environment.

Do not install these three stacks into one Python environment. See
[`docs/ISAAC_LAB_GROOT_SONIC.md`](../docs/ISAAC_LAB_GROOT_SONIC.md) for the
end-to-end sequence.
