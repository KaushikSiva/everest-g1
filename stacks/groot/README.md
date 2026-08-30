# GR00T N1.7

This folder owns only the GR00T N1.7 integration. Its setup script clones the
pinned NVIDIA Isaac-GR00T source into a separate external checkout, initializes
submodules/LFS, creates GR00T's own Python 3.12 environment, and verifies the
import.

```bash
./stacks/groot/setup.sh
```

Dataset and fine-tuning instructions are in the root stack runbook. No GR00T
output is connected directly to the commissioning controller.
