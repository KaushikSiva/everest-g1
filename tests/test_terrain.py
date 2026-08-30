import numpy as np
import pytest

from summit_sentinel.terrain import (
    GRID_SIZE,
    HALF_EXTENT_METERS,
    PLATEAU_ELEVATION,
    PLATEAU_X_METERS,
    PLATEAU_Y_METERS,
    encoded_heightfield,
    ensure_terrain,
    heightfield,
)


def test_terrain_has_flat_spawn_corridor_and_scenic_far_field() -> None:
    terrain = heightfield()
    pixels = encoded_heightfield()
    axis = np.linspace(-HALF_EXTENT_METERS, HALF_EXTENT_METERS, GRID_SIZE)
    inside = (
        (axis[np.newaxis, :] >= PLATEAU_X_METERS[0])
        & (axis[np.newaxis, :] <= PLATEAU_X_METERS[1])
        & (axis[:, np.newaxis] >= PLATEAU_Y_METERS[0])
        & (axis[:, np.newaxis] <= PLATEAU_Y_METERS[1])
    )
    assert terrain.shape == (GRID_SIZE, GRID_SIZE)
    assert np.all(terrain[inside] == np.float32(PLATEAU_ELEVATION))
    assert np.unique(pixels[inside]).tolist() == [9]
    assert terrain.min() == 0.0
    assert terrain.max() == 1.0
    assert np.quantile(terrain[~inside], 0.95) > 0.35


def test_terrain_is_deterministic() -> None:
    assert np.array_equal(encoded_heightfield(), encoded_heightfield())


def test_existing_corrupt_terrain_is_rejected(tmp_path) -> None:
    terrain_path = tmp_path / "everest.png"
    terrain_path.write_bytes(b"not a png")

    with pytest.raises(ValueError, match="failed deterministic validation"):
        ensure_terrain(terrain_path)
