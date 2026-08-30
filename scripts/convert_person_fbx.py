"""Convert a legally obtained FBX to an ignored USD asset using Blender.

Run with:
    blender --background --python scripts/convert_person_fbx.py -- input.fbx output.usd
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    try:
        separator = sys.argv.index("--")
        source_arg, output_arg = sys.argv[separator + 1 : separator + 3]
    except (ValueError, IndexError) as exc:
        raise SystemExit(
            "usage: blender --background --python SCRIPT -- INPUT.fbx OUTPUT.usd"
        ) from exc

    source = Path(source_arg).expanduser().resolve()
    output = Path(output_arg).expanduser().resolve()
    if source.suffix.lower() != ".fbx" or output.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise SystemExit("input must be .fbx and output must be .usd, .usda, or .usdc")
    if not source.is_file():
        raise SystemExit(f"FBX does not exist: {source}")
    if "runtime" not in output.parts:
        raise SystemExit(
            "write converted assets beneath runtime/ so they cannot be committed accidentally"
        )

    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.usd_export(filepath=str(output), export_animation=False)
    print(f"Converted person asset: {output}")


if __name__ == "__main__":
    main()
