"""Register Everest components, then delegate execution to Arena's runner."""

from isaaclab_arena.evaluation.policy_runner import main

from everest_g1.isaac import environment as _environment  # noqa: F401
from everest_g1.isaac import policy as _policy  # noqa: F401

if __name__ == "__main__":
    main()
