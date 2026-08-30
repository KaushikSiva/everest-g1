#!/usr/bin/env python3
"""Render deterministic README images from the bundled MuJoCo scene.

Run from the repository root:

    uv run python scripts/render_readme_images.py

On macOS, use ``uv run mjpython`` instead if the normal Python launcher cannot
create an OpenGL context. The script holds the robot in its configured standing
pose; it does not advance physics or modify the runtime scene.
"""

from __future__ import annotations

import argparse
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from summit_sentinel.simulation import SummitSentinelEnv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "images"
WIDTH = 1280
HEIGHT = 720


@dataclass(frozen=True)
class CameraSpec:
    filename: str
    lookat: tuple[float, float, float]
    distance: float
    azimuth: float
    elevation: float


CAMERAS = (
    CameraSpec(
        filename="summit-sentinel-hero.png",
        lookat=(0.25, 0.15, 1.05),
        distance=3.4,
        azimuth=221.0,
        elevation=-10.0,
    ),
    CameraSpec(
        filename="summit-sentinel-terrain.png",
        lookat=(0.9, 1.6, 0.95),
        distance=6.7,
        azimuth=207.0,
        elevation=-21.0,
    ),
)


def _chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))


def write_rgb_png(path: Path, pixels: np.ndarray) -> None:
    """Write a MuJoCo RGB frame as a lossless PNG using only the standard library."""

    if pixels.shape != (HEIGHT, WIDTH, 3) or pixels.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB image {(HEIGHT, WIDTH, 3)}, got {pixels.shape}")
    rows = b"".join(b"\x00" + row.tobytes() for row in pixels)
    header = struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0)
    encoded = b"\x89PNG\r\n\x1a\n"
    encoded += _chunk(b"IHDR", header)
    encoded += _chunk(b"IDAT", zlib.compress(rows, level=9))
    encoded += _chunk(b"IEND", b"")
    path.write_bytes(encoded)


def render(output_dir: Path) -> list[Path]:
    """Render every documented camera without advancing simulation time."""

    output_dir.mkdir(parents=True, exist_ok=True)
    env = SummitSentinelEnv(use_policy=False, auto_reset=False)
    renderer = mujoco.Renderer(env.model, height=HEIGHT, width=WIDTH)
    options = mujoco.MjvOption()
    options.geomgroup[4] = 0  # Hide the translucent spawn/debug marker.

    paths: list[Path] = []
    try:
        for spec in CAMERAS:
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = spec.lookat
            camera.distance = spec.distance
            camera.azimuth = spec.azimuth
            camera.elevation = spec.elevation
            renderer.update_scene(env.data, camera=camera, scene_option=options)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = True
            pixels = np.asarray(renderer.render(), dtype=np.uint8)
            path = output_dir / spec.filename
            write_rgb_png(path, pixels)
            paths.append(path)
    finally:
        renderer.close()
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"image directory (default: {DEFAULT_OUTPUT.relative_to(ROOT)})",
    )
    args = parser.parse_args()
    for path in render(args.output_dir.resolve()):
        print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == "__main__":
    main()
