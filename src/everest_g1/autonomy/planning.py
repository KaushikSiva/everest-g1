"""Gemini ER 2 route selection over deterministic, safety-checked options."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from summit_sentinel.terrain import (
    plateau_world_height,
    terrain_slope_degrees,
    terrain_world_height,
)

GEMINI_ROBOTICS_MODEL = "gemini-robotics-er-2-preview"
AutonomyMode = Literal["rescue", "carry", "scan"]


class PlannerError(RuntimeError):
    """Gemini planning failed before motion was permitted."""


@dataclass(frozen=True)
class AcousticPlanningObservation:
    """Initial simulated array reading supplied to Gemini as route context."""

    bearing_rad: float
    confidence: float
    coarse_range_m: float

    def validate(self) -> None:
        if not math.isfinite(self.bearing_rad) or not -math.pi <= self.bearing_rad <= math.pi:
            raise ValueError("acoustic bearing must be finite and wrapped to [-pi, pi]")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("acoustic confidence must be between zero and one")
        if not math.isfinite(self.coarse_range_m) or self.coarse_range_m < 0.0:
            raise ValueError("coarse acoustic range must be finite and non-negative")


@dataclass(frozen=True)
class EnvironmentProfile:
    """Environmental inputs used for both route scoring and the Gemini prompt."""

    temperature_c: float = -18.0
    wind_mps: float = 8.0
    visibility_m: float = 900.0
    snow_depth_m: float = 0.14
    friction: float = 0.82

    def validate(self) -> None:
        bounds = {
            "temperature_c": (-45.0, 10.0),
            "wind_mps": (0.0, 30.0),
            "visibility_m": (10.0, 10_000.0),
            "snow_depth_m": (0.0, 0.5),
            "friction": (0.2, 1.5),
        }
        for name, (minimum, maximum) in bounds.items():
            value = float(getattr(self, name))
            if not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}")


@dataclass(frozen=True)
class WaypointAssessment:
    x_m: float
    y_m: float
    slope_deg: float
    terrain_height_m: float
    temperature_c: float
    wind_mps: float
    visibility_m: float
    snow_depth_m: float
    effective_friction: float
    distance_from_start_m: float
    risk_score: float
    hard_safe: bool


@dataclass(frozen=True)
class RouteOption:
    route_id: str
    purpose: str
    approach_waypoints: tuple[tuple[float, float], ...]
    mission_waypoints: tuple[tuple[float, float], ...]
    assessments: tuple[WaypointAssessment, ...]
    aggregate_risk: float
    hard_safe: bool

    def prompt_payload(self) -> dict[str, object]:
        return {
            "route_id": self.route_id,
            "purpose": self.purpose,
            "approach_waypoints": self.approach_waypoints,
            "mission_waypoints": self.mission_waypoints,
            "aggregate_risk": round(self.aggregate_risk, 4),
            "hard_safe": self.hard_safe,
            "samples": [asdict(sample) for sample in self.assessments],
        }


def _assess_waypoint(
    point: tuple[float, float],
    profile: EnvironmentProfile,
    *,
    start_xy: tuple[float, float],
) -> WaypointAssessment:
    x_m, y_m = point
    height_m = terrain_world_height(x_m, y_m)
    slope_deg = terrain_slope_degrees(x_m, y_m)
    height_above_plateau = max(0.0, height_m - plateau_world_height())
    exposure = float(np.clip((y_m + 1.8) / 4.0, 0.0, 1.0))
    temperature_c = profile.temperature_c - 2.4 * height_above_plateau - 1.8 * exposure
    wind_mps = profile.wind_mps * (1.0 + 0.28 * exposure + 0.008 * slope_deg)
    snow_depth_m = min(
        0.5,
        profile.snow_depth_m + 0.045 * exposure + 0.0025 * slope_deg,
    )
    effective_friction = max(
        0.1,
        profile.friction * (1.0 - 0.60 * snow_depth_m / 0.5) - 0.003 * slope_deg,
    )
    visibility_m = max(
        10.0,
        profile.visibility_m / (1.0 + 0.018 * wind_mps + 0.9 * snow_depth_m),
    )
    distance = math.dist(start_xy, point)
    risk = (
        0.36 * min(slope_deg / 15.0, 2.0)
        + 0.18 * min(wind_mps / 25.0, 2.0)
        + 0.16 * min(snow_depth_m / 0.5, 1.0)
        + 0.14 * max(0.0, (0.45 - effective_friction) / 0.35)
        + 0.10 * max(0.0, (100.0 - visibility_m) / 90.0)
        + 0.06 * max(0.0, (-30.0 - temperature_c) / 15.0)
    )
    hard_safe = (
        slope_deg <= 14.0
        and temperature_c >= -40.0
        and wind_mps <= 25.0
        and visibility_m >= 20.0
        and snow_depth_m <= 0.45
        and effective_friction >= 0.25
    )
    return WaypointAssessment(
        x_m=round(x_m, 3),
        y_m=round(y_m, 3),
        slope_deg=round(slope_deg, 3),
        terrain_height_m=round(height_m, 3),
        temperature_c=round(temperature_c, 2),
        wind_mps=round(wind_mps, 2),
        visibility_m=round(visibility_m, 1),
        snow_depth_m=round(snow_depth_m, 3),
        effective_friction=round(effective_friction, 3),
        distance_from_start_m=round(distance, 3),
        risk_score=round(risk, 4),
        hard_safe=hard_safe,
    )


def _route(
    route_id: str,
    purpose: str,
    approach: tuple[tuple[float, float], ...],
    mission: tuple[tuple[float, float], ...],
    profile: EnvironmentProfile,
    start_xy: tuple[float, float],
) -> RouteOption:
    assessments = tuple(
        _assess_waypoint(point, profile, start_xy=start_xy) for point in approach + mission
    )
    aggregate = sum(sample.risk_score for sample in assessments) / max(1, len(assessments))
    return RouteOption(
        route_id=route_id,
        purpose=purpose,
        approach_waypoints=approach,
        mission_waypoints=mission,
        assessments=assessments,
        aggregate_risk=round(aggregate, 4),
        hard_safe=bool(assessments) and all(sample.hard_safe for sample in assessments),
    )


def build_route_options(
    mode: AutonomyMode,
    profile: EnvironmentProfile,
    *,
    start_xy: tuple[float, float] = (0.0, 0.0),
) -> tuple[RouteOption, ...]:
    """Build named paths whose complete terrain factors are visible to Gemini."""

    profile.validate()
    if mode == "rescue":
        specs = (
            ("rescue-direct", "short direct casualty approach", ((0.55, 0.10),), ()),
            (
                "rescue-south-shelter",
                "lower-exposure casualty approach",
                ((0.25, -0.45), (0.72, -0.12)),
                (),
            ),
            (
                "rescue-north-view",
                "higher-visibility casualty approach",
                ((0.25, 0.58), (0.72, 0.46)),
                (),
            ),
        )
    elif mode == "carry":
        specs = (
            (
                "carry-south-shelter",
                "approach casualty then carry through sheltered flat snow",
                ((0.45, -0.12),),
                ((0.75, -0.55), (0.05, -0.75), (-0.45, -0.15)),
            ),
            (
                "carry-flat-loop",
                "approach casualty then carry around the central plateau",
                ((0.55, 0.12),),
                ((1.10, -0.35), (0.20, -0.45), (0.05, 0.55), (0.82, 0.62)),
            ),
            (
                "carry-north-observation",
                "approach casualty then carry toward the exposed observation side",
                ((0.42, 0.42),),
                ((1.15, 1.10), (0.35, 1.48), (-0.25, 1.05)),
            ),
        )
    elif mode == "scan":
        specs = (
            (
                "scan-sheltered-low-grade",
                "survey the sheltered low-grade southern corridor",
                (),
                ((0.0, -1.15), (1.50, -1.55), (3.00, -1.72)),
            ),
            (
                "scan-central-observation",
                "survey the central route and moderate northern grade",
                (),
                ((0.0, 1.25), (1.65, 1.58), (3.00, 1.95)),
            ),
            (
                "scan-west-cold-basin",
                "survey the western basin with lower visibility exposure",
                (),
                ((-0.75, 0.40), (-1.85, 1.35), (-2.90, 1.95)),
            ),
            (
                "scan-east-ridge",
                "survey the steep exposed eastern ridge",
                (),
                ((1.25, 0.0), (3.55, 1.20), (5.00, 2.00)),
            ),
        )
    else:
        raise ValueError(f"unsupported autonomy mode: {mode}")
    return tuple(
        _route(route_id, purpose, approach, mission, profile, start_xy)
        for route_id, purpose, approach, mission in specs
    )


class GeminiRoutePlanner:
    """Ask Gemini ER 2 to rank bounded route IDs, then validate its answer."""

    def __init__(
        self,
        *,
        model: str = GEMINI_ROBOTICS_MODEL,
        api_key: str | None = None,
        offline: bool = False,
    ) -> None:
        self.model = model
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
        self.offline = offline
        self.last_reason = ""
        self.last_observations = ""
        self.last_acoustic_observation: AcousticPlanningObservation | None = None
        self.provider = "offline-deterministic" if offline else "gemini-er-2"

    def select(
        self,
        mode: AutonomyMode,
        options: tuple[RouteOption, ...],
        *,
        image_jpeg: bytes | None,
        acoustic_observation: AcousticPlanningObservation | None = None,
    ) -> RouteOption:
        if not options:
            raise PlannerError("no route options were generated")
        safe_options = tuple(option for option in options if option.hard_safe)
        if not safe_options:
            raise PlannerError("all routes violate the local hard safety envelope")
        if acoustic_observation is not None:
            acoustic_observation.validate()
        self.last_acoustic_observation = acoustic_observation
        if self.offline:
            selected = min(
                safe_options,
                key=lambda option: (option.aggregate_risk, option.route_id),
            )
            self.last_reason = "lowest locally computed aggregate risk"
            self.last_observations = "offline planning explicitly requested"
            return selected
        if not self.api_key:
            raise PlannerError(
                "GEMINI_API_KEY is required; use --offline-plan only for deterministic testing"
            )
        if image_jpeg is None or not image_jpeg.startswith(b"\xff\xd8"):
            raise PlannerError("a valid G1 front-camera JPEG is required for Gemini planning")

        prompt = self._prompt(mode, options, acoustic_observation=acoustic_observation)
        try:
            from google import genai
            from google.genai import types

            schema = {
                "type": "object",
                "properties": {
                    "route_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "observations": {"type": "string"},
                },
                "required": ["route_id", "reason", "observations"],
                "additionalProperties": False,
            }
            with genai.Client(api_key=self.api_key) as client:
                response = client.models.generate_content(
                    model=self.model,
                    contents=[
                        types.Part.from_bytes(data=image_jpeg, mime_type="image/jpeg"),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.1,
                        thinking_config=types.ThinkingConfig(thinking_level="high"),
                    ),
                )
            payload = json.loads(response.text)
        except Exception as error:
            raise PlannerError(f"Gemini ER 2 planning failed: {type(error).__name__}") from error

        route_id = payload.get("route_id")
        selected = next((option for option in options if option.route_id == route_id), None)
        if selected is None:
            raise PlannerError("Gemini returned an unknown route ID")
        if not selected.hard_safe:
            raise PlannerError("Gemini selected a route rejected by the local safety envelope")
        reason = payload.get("reason")
        observations = payload.get("observations")
        if not isinstance(reason, str) or not isinstance(observations, str):
            raise PlannerError("Gemini response omitted required planning explanations")
        self.last_reason = reason[:500]
        self.last_observations = observations[:500]
        return selected

    @staticmethod
    def _prompt(
        mode: AutonomyMode,
        options: tuple[RouteOption, ...],
        *,
        acoustic_observation: AcousticPlanningObservation | None = None,
    ) -> str:
        payload = [option.prompt_payload() for option in options]
        acoustic_context = (
            {
                "bearing_rad_body_frame": round(acoustic_observation.bearing_rad, 5),
                "confidence": round(acoustic_observation.confidence, 4),
                "coarse_range_m_telemetry_only": round(acoustic_observation.coarse_range_m, 3),
            }
            if acoustic_observation is not None
            else None
        )
        return (
            "You are the high-level embodied-reasoning planner for a Unitree G1 in MuJoCo. "
            f"Mission mode: {mode}. Inspect the attached robot-front-camera frame and compare "
            "EVERY supplied route using slope_deg, temperature_c, wind_mps, visibility_m, "
            "snow_depth_m, effective_friction, distance, and aggregate risk. Prefer a route "
            "that is consistent with the initial acoustic bearing when one is supplied, but "
            "treat its coarse acoustic range as telemetry only, never as a stop/call gate. "
            "Complete the mission conservatively. Never choose hard_safe=false. Return "
            "only the required structured response; choose exactly one existing route_id. "
            "You are not issuing joint, torque, or velocity commands. Routes:\n"
            + json.dumps(
                {"acoustic_observation": acoustic_context, "routes": payload},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
