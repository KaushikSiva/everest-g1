"""Small dependency-free models shared by the simulator and tests."""

from __future__ import annotations

import base64
from dataclasses import dataclass

MAX_EVIDENCE_JPEG_BYTES = 2_000_000


@dataclass(frozen=True)
class RescueObservation:
    """Observable simulation facts sent to BeaconCall."""

    simulation_id: str
    distance_m: float
    observed_state: str = "motionless_adult_in_snow"
    camera_name: str = "G1-FRONT-CAMERA"
    image_jpeg: bytes | None = None

    def as_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "simulation_id": self.simulation_id,
            "distance_m": self.distance_m,
            "observed_state": self.observed_state,
            "camera_name": self.camera_name,
        }
        if self.image_jpeg is not None:
            if (
                not self.image_jpeg.startswith(b"\xff\xd8")
                or not self.image_jpeg.endswith(b"\xff\xd9")
                or len(self.image_jpeg) > MAX_EVIDENCE_JPEG_BYTES
            ):
                raise ValueError("front-camera evidence must be a JPEG no larger than 2 MB")
            encoded = base64.b64encode(self.image_jpeg).decode("ascii")
            payload["image_data_url"] = f"data:image/jpeg;base64,{encoded}"
        return payload


@dataclass(frozen=True)
class NavigationCommand:
    """Bounded body-frame velocity command for the G1 WBC."""

    forward_mps: float
    lateral_mps: float
    yaw_rps: float
    surface_distance_m: float
    reached: bool
