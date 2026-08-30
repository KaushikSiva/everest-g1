"""Small dependency-free models shared by the simulator and tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RescueObservation:
    """Observable simulation facts sent to BeaconCall."""

    simulation_id: str
    distance_m: float
    observed_state: str = "motionless_adult_in_snow"

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NavigationCommand:
    """Bounded body-frame velocity command for the G1 WBC."""

    forward_mps: float
    lateral_mps: float
    yaw_rps: float
    surface_distance_m: float
    reached: bool
