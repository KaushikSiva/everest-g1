import io

import pytest
import torch
from PIL import Image

from everest_g1.isaac.camera import (
    ISAAC_CAMERA_GROUP,
    ISAAC_FRONT_CAMERA_TERM,
    isaac_front_camera_jpeg,
)


def test_isaac_front_camera_encodes_single_rgb_observation() -> None:
    frame = torch.zeros((1, 48, 64, 3), dtype=torch.uint8)
    frame[:, 8:40, 16:48, 0] = 220
    jpeg = isaac_front_camera_jpeg({ISAAC_CAMERA_GROUP: {ISAAC_FRONT_CAMERA_TERM: frame}})

    assert jpeg.startswith(b"\xff\xd8")
    assert jpeg.endswith(b"\xff\xd9")
    with Image.open(io.BytesIO(jpeg)) as image:
        assert image.mode == "RGB"
        assert image.size == (64, 48)


def test_isaac_front_camera_accepts_normalized_float_rgba() -> None:
    frame = torch.ones((1, 32, 40, 4), dtype=torch.float32)
    jpeg = isaac_front_camera_jpeg({ISAAC_CAMERA_GROUP: {ISAAC_FRONT_CAMERA_TERM: frame}})

    with Image.open(io.BytesIO(jpeg)) as image:
        assert image.getpixel((0, 0)) == (255, 255, 255)


@pytest.mark.parametrize(
    "observation",
    [
        {},
        {ISAAC_CAMERA_GROUP: {}},
        {ISAAC_CAMERA_GROUP: {ISAAC_FRONT_CAMERA_TERM: torch.zeros((2, 32, 40, 3))}},
    ],
)
def test_isaac_front_camera_fails_closed_on_missing_or_multi_env_frames(
    observation,
) -> None:
    with pytest.raises(RuntimeError, match="Isaac"):
        isaac_front_camera_jpeg(observation)
