"""Deterministic, compressed Everest-inspired MuJoCo heightfield generation."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

GRID_SIZE = 257
TERRAIN_SEED = 8848
HALF_EXTENT_METERS = 12.0
HEIGHT_SCALE_METERS = 3.0
BASE_DEPTH_METERS = 0.10
PLATEAU_ELEVATION = 9 / 255
PLATEAU_X_METERS = (-3.0, 4.0)
PLATEAU_Y_METERS = (-1.8, 1.8)
TERRAIN_PATH = Path(__file__).resolve().parent / "assets" / "unitree_g1" / "everest.png"


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def heightfield(seed: int = TERRAIN_SEED) -> np.ndarray:
    """Return normalized terrain with an exact flat center corridor.

    The 24 m square field is deliberately a compressed artistic environment,
    not a geospatial or literal-scale model of Mount Everest.
    """

    axis = np.linspace(-HALF_EXTENT_METERS, HALF_EXTENT_METERS, GRID_SIZE)
    x, y = np.meshgrid(axis, axis)
    rng = np.random.default_rng(seed)

    def peak(cx: float, cy: float, sx: float, sy: float, power: float) -> np.ndarray:
        radius = np.sqrt(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2)
        return np.clip(1.0 - radius, 0.0, 1.0) ** power

    main = peak(5.4, 6.6, 8.5, 8.0, 1.45)
    west_ridge = 0.55 * peak(-8.5, 7.5, 7.0, 6.0, 1.7)
    east_ridge = 0.48 * peak(10.0, -4.5, 6.0, 8.0, 1.5)
    north_wall = 0.24 * _smoothstep((y - 3.0) / 7.0)

    phase = rng.uniform(0.0, 2.0 * np.pi, size=4)
    ridges = (
        0.055 * np.sin(0.95 * x + 0.41 * y + phase[0])
        + 0.035 * np.sin(1.75 * x - 0.63 * y + phase[1])
        + 0.020 * np.sin(3.10 * x + 1.30 * y + phase[2])
        + 0.012 * np.sin(5.25 * x - 2.20 * y + phase[3])
    )
    raw = np.maximum(main + west_ridge + east_ridge + north_wall + ridges, 0.0)
    raw -= raw.min()
    raw /= raw.max()

    # Blend down to the traversal corridor, then overwrite its interior so the
    # encoded PNG is exactly planar under both feet at reset.
    dx = np.maximum(np.maximum(PLATEAU_X_METERS[0] - x, 0), x - PLATEAU_X_METERS[1])
    dy = np.maximum(np.maximum(PLATEAU_Y_METERS[0] - y, 0), y - PLATEAU_Y_METERS[1])
    distance = np.sqrt(dx * dx + dy * dy)
    blend = _smoothstep(distance / 2.2)
    terrain = PLATEAU_ELEVATION * (1.0 - blend) + raw * blend
    inside = (
        (x >= PLATEAU_X_METERS[0])
        & (x <= PLATEAU_X_METERS[1])
        & (y >= PLATEAU_Y_METERS[0])
        & (y <= PLATEAU_Y_METERS[1])
    )
    terrain[inside] = PLATEAU_ELEVATION
    terrain[0, 0] = 0.0
    terrain[-1, -1] = 1.0
    return terrain.astype(np.float32)


def encoded_heightfield(seed: int = TERRAIN_SEED) -> np.ndarray:
    """Return the exact 8-bit pixels consumed by MuJoCo."""

    return np.rint(heightfield(seed) * 255).astype(np.uint8)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))


def _grayscale_png_bytes(pixels: np.ndarray) -> bytes:
    if pixels.ndim != 2 or pixels.dtype != np.uint8:
        raise ValueError("PNG heightfield must be a two-dimensional uint8 array")
    height, width = pixels.shape
    rows = b"".join(b"\x00" + row.tobytes() for row in pixels)
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", header)
    png += _png_chunk(b"IDAT", zlib.compress(rows, level=9))
    png += _png_chunk(b"IEND", b"")
    return png


def write_grayscale_png(path: Path, pixels: np.ndarray) -> None:
    """Write an 8-bit grayscale PNG without adding an image dependency."""

    path.write_bytes(_grayscale_png_bytes(pixels))


def ensure_terrain(path: Path = TERRAIN_PATH, seed: int = TERRAIN_SEED) -> Path:
    """Materialize or validate the exact deterministic runtime heightfield."""

    expected_png = _grayscale_png_bytes(encoded_heightfield(seed))
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(expected_png)
    elif path.read_bytes() != expected_png:
        raise ValueError(f"terrain asset failed deterministic validation: {path}")
    return path


def plateau_world_height() -> float:
    """Return the MuJoCo world-space height of the encoded flat patch."""

    return float(round(PLATEAU_ELEVATION * 255) / 255 * HEIGHT_SCALE_METERS)
