# GEAR-SONIC

This folder owns the SONIC whole-body-controller integration. Its setup script
clones the pinned GR00T-WholeBodyControl repository, downloads its Git LFS
assets, and optionally creates the upstream isolated MuJoCo environment.

```bash
./stacks/sonic/setup.sh
```

Use `./stacks/sonic/setup.sh --checkout-only` when training through the separate
Modal Isaac Lab 2.3.2 image. This script never configures or launches physical
robot deployment.
