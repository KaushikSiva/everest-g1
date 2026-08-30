# Third-party notices

## Unitree RL Gym / G1 assets and policy

This repository vendors the Unitree G1 12-DOF MuJoCo XML, the 27 meshes that XML
references, and the bundled G1 TorchScript locomotion policy from the official
[`unitreerobotics/unitree_rl_gym`](https://github.com/unitreerobotics/unitree_rl_gym)
repository.

- pinned commit: `276801e46c5d433564f24658bac64f254b7d2d4b`
- commit date: 2025-07-25
- upstream model: `resources/robots/g1_description/g1_12dof.xml`
- upstream meshes: `resources/robots/g1_description/meshes/*.STL` (referenced subset)
- upstream policy: `deploy/pre_train/g1/motion.pt`
- policy SHA-256: `cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d`
- model SHA-256: `747ede40aa726b7352bae8353e95d0d0f908cec2257a27cbd78bc6e5a2d5a314`
- license: BSD 3-Clause; copied verbatim to
  `docs/third_party/UNITREE_RL_GYM_LICENSE.txt`

Reference documentation/configuration copied unchanged from the same commit:

- `UNITREE_RL_GYM_README.md`
- `UNITREE_G1_DESCRIPTION_README.md`
- `UNITREE_G1_DEPLOY_CONFIG.yaml`
- `UNITREE_DEPLOY_MUJOCO_REFERENCE.py`
- `UNITREE_G1_TRAINING_CONFIG.py`
- `UNITREE_BASE_TRAINING_CONFIG.py`

Everest G1's `summit_scene.xml`, terrain generator, safety checks, CLI, and
control adapters are modifications/additions and are not supplied or endorsed
by Unitree Robotics.

## Local Robot Nurse casualty asset

The detailed MuJoCo snow casualty is an OBJ conversion of the user-provided
`boy.glb` previously prepared for the Robot Nurse demo. Its source and converted
mesh remain subject to the source asset's own terms; confirm those terms before
redistributing the OBJ or albedo texture. The conversion does not affect the
casualty's non-colliding safety proxy or proximity-trigger semantics.

## Other runtime libraries

MuJoCo, PyTorch, NumPy, PyYAML, and pygame-ce remain external dependencies. See
their installed distributions for their respective license texts.

## NVIDIA simulation, VLA, and whole-body-control repositories

Isaac Lab, Isaac Lab-Arena, Isaac-GR00T, Isaac Teleop, and
GR00T-WholeBodyControl are not redistributed here. Cloud setup scripts clone
the exact commits recorded in `cloud/pins.env`. Their source, model weights,
datasets, containers, and services remain subject to their respective upstream
licenses and NVIDIA terms. Running NVIDIA containers requires acceptance of the
applicable NVIDIA EULA.
