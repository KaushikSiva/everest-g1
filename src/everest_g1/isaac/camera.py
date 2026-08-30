"""Bounded conversion of the pinned Isaac Lab-Arena G1 head camera to JPEG."""

from __future__ import annotations

import io
from collections.abc import Mapping

import numpy as np
import torch
from PIL import Image

from everest_g1.models import MAX_EVIDENCE_JPEG_BYTES

ISAAC_CAMERA_GROUP = "camera_obs"
ISAAC_FRONT_CAMERA_TERM = "robot_head_cam_rgb"


def isaac_front_camera_jpeg(observation: Mapping[str, object]) -> bytes:
    """Encode the single-environment G1 head-camera observation as a JPEG."""

    camera_group = observation.get(ISAAC_CAMERA_GROUP)
    if not isinstance(camera_group, Mapping):
        raise RuntimeError(
            "Isaac camera observations are unavailable; launch with --enable_cameras"
        )
    frame = camera_group.get(ISAAC_FRONT_CAMERA_TERM)
    if not isinstance(frame, torch.Tensor):
        raise RuntimeError(f"Isaac camera term {ISAAC_FRONT_CAMERA_TERM!r} is unavailable")
    if frame.ndim != 4 or frame.shape[0] != 1 or frame.shape[-1] not in (3, 4):
        raise RuntimeError("Isaac front camera must have shape [1, height, width, 3 or 4]")

    rgb = frame[0, :, :, :3].detach().cpu()
    if rgb.is_floating_point():
        if not torch.isfinite(rgb).all():
            raise RuntimeError("Isaac front camera contains non-finite pixels")
        if float(rgb.max()) <= 1.0 and float(rgb.min()) >= 0.0:
            rgb = rgb * 255.0
        rgb = rgb.clamp(0.0, 255.0).to(torch.uint8)
    else:
        rgb = rgb.clamp(0, 255).to(torch.uint8)

    pixels = np.ascontiguousarray(rgb.numpy())
    encoded = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(
        encoded,
        format="JPEG",
        quality=85,
        optimize=False,
        progressive=False,
    )
    jpeg = encoded.getvalue()
    if (
        not jpeg.startswith(b"\xff\xd8")
        or not jpeg.endswith(b"\xff\xd9")
        or len(jpeg) > MAX_EVIDENCE_JPEG_BYTES
    ):
        raise RuntimeError("Isaac front-camera JPEG encoding failed or exceeded 2 MB")
    return jpeg
