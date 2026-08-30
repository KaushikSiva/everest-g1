# GR00T N1.7 + GEAR-SONIC lane

This is the real NVIDIA whole-body path, not a rename of the Arena commissioning
controller. GR00T predicts a 40-step chunk of 64-dimensional SONIC latent motion
tokens plus hand actions; SONIC decodes body motion at 50 Hz.

## Version boundary

- `GR00T-WholeBodyControl` is pinned in `cloud/pins.env` and targets Isaac Lab
  2.3.2 for the released training stack.
- Isaac Lab-Arena is pinned separately and currently targets Isaac Lab 3.0 beta 2.

Do not install both into one virtual environment.

## Setup and model

```bash
./cloud/setup_groot_sonic.sh
cd /home/ubuntu/workspace/everest-g1-cloud/GR00T-WholeBodyControl
```

Follow the pinned upstream training installation. In its Isaac Lab 2.3.2 Python:

```bash
pip install -e 'gear_sonic[training]'
python check_environment.py --training
python download_from_hf.py --sonic-v1-1
```

Hugging Face access and NVIDIA model-license acceptance may be required. Keep
tokens in the cloud secret manager.

## Data requirement

An approach-and-assess checkpoint is not bundled. Collect and review successful
demonstrations first. The dataset must use the `UNITREE_G1_SONIC` embodiment and
the LeRobot structure documented upstream. A placeholder base model is not a
task policy and must not be used to claim a successful rescue rollout.

## N1.7 fine-tune contract

From a pinned Isaac-GR00T checkout, the task-specific form is:

```bash
uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /datasets/everest_rescue \
  --embodiment-tag UNITREE_G1_SONIC \
  --modality-config-path gr00t/configs/data/embodiment_configs.py \
  --num-gpus 1 \
  --output-dir /models/everest_rescue \
  --max-steps 2000 \
  --global-batch-size 32 \
  --dataloader-num-workers 4
```

The Modal app exposes `finetune_groot_n17_sonic`, a bounded H100 function using
named volumes and this exact embodiment contract. It separately exposes
`train_sonic_controller` for the lower-level controller. Full convergence
generally needs more data, steps, and GPUs than the smoke limits. First validate
data shape and short-run loss; then deliberately increase resources.

## Promotion gate

Do not replace `everest_approach` until a SONIC-backed policy passes all of:

1. open-loop dataset validation;
2. headless checkpoint evaluation;
3. rendered Isaac evaluation with collision review;
4. the same speed, touching-distance, dwell, and one-call acceptance tests;
5. repeated seeds and perturbations with retained metrics;
6. a separate physical-robot safety review.

XR/Isaac Teleop is the planned demonstration-collection interface. It is pinned
for reproducibility but is not required for the scripted commissioning run.
