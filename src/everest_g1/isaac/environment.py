"""Registered Isaac Lab-Arena snow rescue environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import (
    ArenaEnvironmentCfg,
    ArenaEnvironmentFactory,
)

_ASSET_DIR = Path(__file__).resolve().parent / "assets"


@dataclass
class EverestG1RescueEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the static rescue scene and G1 embodiment."""

    scene_usd: str = str(_ASSET_DIR / "snow_rescue_scene.usda")
    person_usd: str = str(_ASSET_DIR / "downed_person_proxy.usda")
    person_x_m: float = 2.6
    person_y_m: float = 0.75
    embodiment: str = "g1_wbc_agile_joint"


@register_environment
class EverestG1RescueEnvironment(ArenaEnvironmentFactory[EverestG1RescueEnvironmentCfg]):
    """One G1, one downed-person proxy, and a collision-enabled snow field."""

    name = "everest_g1_rescue"
    _legacy_argparse_cfg_type = EverestG1RescueEnvironmentCfg

    def build(self, cfg: EverestG1RescueEnvironmentCfg):
        from isaaclab_arena.assets.background import Background
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.environments.isaaclab_arena_environment import (
            IsaacLabArenaEnvironment,
        )
        from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import (
            set_control_rate_50hz,
        )
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.no_task import NoTask
        from isaaclab_arena.utils.pose import Pose

        background = Background(
            name="everest_snow",
            usd_path=str(Path(cfg.scene_usd).expanduser().resolve()),
            object_min_z=-0.25,
        )
        person = Object(
            name="downed_person",
            usd_path=str(Path(cfg.person_usd).expanduser().resolve()),
            object_type=ObjectType.BASE,
            initial_pose=Pose(
                position_xyz=(cfg.person_x_m, cfg.person_y_m, 0.0),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
        )
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras
        )
        embodiment.set_initial_pose(
            Pose(position_xyz=(0.0, 0.0, 0.78), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        )

        def env_cfg_callback(env_cfg):
            # NVIDIA's G1 WBC is designed for a 50 Hz command rate.
            return set_control_rate_50hz(env_cfg)

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=[background, person]),
            task=NoTask(),
            env_cfg_callback=env_cfg_callback,
        )
