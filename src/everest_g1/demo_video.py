"""Render the three autonomous MuJoCo modes as one exact 45-second MP4."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from everest_g1.autonomy import (
    AcousticPlanningObservation,
    AutonomousMujocoController,
    EnvironmentProfile,
    GeminiRoutePlanner,
    build_route_options,
)
from everest_g1.mujoco import yaw_from_wxyz
from everest_g1.spatial_audio import AcousticBeaconSensor, SpatialAudioSettings
from summit_sentinel.simulation import SummitSentinelEnv

WIDTH = 1280
HEIGHT = 720
FPS = 30
TOTAL_SECONDS = 45.0
TITLE_SECONDS = 1.25
DEFAULT_OUTPUT = Path("runtime/everest-g1-autonomy-demo.mp4")


@dataclass(frozen=True)
class Chapter:
    number: str
    mode: str
    title: str
    subtitle: str
    duration_seconds: float
    simulation_speed: float
    camera_distance: float


CHAPTERS = (
    Chapter(
        number="02",
        mode="rescue",
        title="RESCUE",
        subtitle="Locate · approach · verify",
        duration_seconds=15.0,
        simulation_speed=0.70,
        camera_distance=3.8,
    ),
    Chapter(
        number="03",
        mode="carry",
        title="CARRY",
        subtitle="Secure · lift · evacuate",
        duration_seconds=18.0,
        simulation_speed=1.35,
        camera_distance=3.6,
    ),
    Chapter(
        number="04",
        mode="scan",
        title="SCAN",
        subtitle="Assess · route · patrol",
        duration_seconds=12.0,
        simulation_speed=2.60,
        camera_distance=4.5,
    ),
)


@dataclass
class Mission:
    chapter: Chapter
    env: SummitSentinelEnv
    controller: AutonomousMujocoController
    planner: GeminiRoutePlanner
    route_id: str
    aggregate_risk: float
    planning_camera_bytes: int


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _safe_line(value: str, limit: int = 220) -> str:
    printable = "".join(character if character.isprintable() else " " for character in value)
    return " ".join(printable.replace("\x1b", " ").split())[:limit]


def _gradient_background() -> Image.Image:
    top = np.asarray([5, 18, 30], dtype=np.float32)
    bottom = np.asarray([12, 49, 63], dtype=np.float32)
    blend = np.linspace(0.0, 1.0, HEIGHT, dtype=np.float32)[:, None, None]
    pixels = top[None, None, :] * (1.0 - blend) + bottom[None, None, :] * blend
    pixels = np.broadcast_to(pixels, (HEIGHT, WIDTH, 3)).astype(np.uint8).copy()
    return Image.fromarray(pixels)


def title_card(chapter: Chapter, mission: Mission) -> np.ndarray:
    """Create a clean chapter card with validated plan metadata."""

    image = _gradient_background()
    draw = ImageDraw.Draw(image, "RGBA")
    accent = (89, 225, 194, 255)
    muted = (180, 203, 211, 255)

    # Subtle summit silhouette and a crisp mission accent.
    draw.polygon(
        [
            (0, 600),
            (185, 430),
            (330, 535),
            (545, 310),
            (760, 510),
            (980, 360),
            (1280, 575),
            (1280, 720),
            (0, 720),
        ],
        fill=(5, 25, 36, 150),
    )
    draw.rectangle((82, 82, 90, 638), fill=accent)
    draw.text((122, 92), "EVEREST G1  /  AUTONOMOUS MISSION", font=_font(24, bold=True), fill=muted)
    draw.text((112, 174), chapter.number, font=_font(76, bold=True), fill=accent)
    draw.text((112, 255), chapter.title, font=_font(118, bold=True), fill=(244, 249, 250, 255))
    draw.text((120, 398), chapter.subtitle, font=_font(34), fill=muted)
    draw.line((120, 475, 700, 475), fill=(89, 225, 194, 150), width=2)
    provider = (
        "GEMINI ROBOTICS ER 2" if mission.planner.provider == "gemini-er-2" else "OFFLINE PLAN"
    )
    draw.text(
        (120, 505),
        f"{provider}  ·  {mission.route_id}  ·  risk {mission.aggregate_risk:.3f}",
        font=_font(22, bold=True),
        fill=(213, 229, 233, 255),
    )
    return np.asarray(image, dtype=np.uint8)


def _wrap_reason(value: str, width: int = 92) -> list[str]:
    return textwrap.wrap(_safe_line(value), width=width, max_lines=2, placeholder="…") or ["—"]


def action_overlay(frame: np.ndarray, mission: Mission, elapsed_s: float) -> np.ndarray:
    """Add mission identity and Gemini rationale without obscuring motion."""

    image = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    accent = (89, 225, 194, 255)

    draw.rounded_rectangle((34, 30, 515, 158), radius=18, fill=(4, 13, 20, 190))
    draw.text(
        (58, 48),
        f"{mission.chapter.number}  {mission.chapter.title}",
        font=_font(32, bold=True),
        fill=(247, 250, 250, 255),
    )
    provider = "GEMINI ER 2" if mission.planner.provider == "gemini-er-2" else "OFFLINE"
    draw.text(
        (59, 96),
        f"{provider}  ·  {mission.route_id}",
        font=_font(18, bold=True),
        fill=accent,
    )
    stage = mission.controller.stage.replace("-", " ").upper()
    draw.text((59, 126), f"STAGE  {stage}", font=_font(16), fill=(201, 219, 224, 255))

    reason_lines = _wrap_reason(mission.planner.last_reason)
    panel_top = HEIGHT - 114
    draw.rounded_rectangle(
        (34, panel_top, WIDTH - 34, HEIGHT - 28), radius=16, fill=(4, 13, 20, 178)
    )
    draw.text((58, panel_top + 14), "PLAN", font=_font(15, bold=True), fill=accent)
    for index, line in enumerate(reason_lines):
        draw.text(
            (122, panel_top + 11 + 25 * index),
            line,
            font=_font(18),
            fill=(237, 244, 245, 255),
        )

    progress = float(np.clip(elapsed_s / mission.chapter.duration_seconds, 0.0, 1.0))
    draw.rounded_rectangle(
        (34, HEIGHT - 12, WIDTH - 34, HEIGHT - 7), radius=2, fill=(255, 255, 255, 55)
    )
    draw.rounded_rectangle(
        (34, HEIGHT - 12, 34 + (WIDTH - 68) * progress, HEIGHT - 7),
        radius=2,
        fill=accent,
    )
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"), dtype=np.uint8)


def _prepare_mission(chapter: Chapter, *, offline: bool, audit_log: Path) -> Mission:
    env = SummitSentinelEnv(use_policy=True, auto_reset=True)
    profile = EnvironmentProfile()
    env.apply_scenario_conditions(
        {
            "friction": profile.friction,
            "wind_mps": profile.wind_mps,
            "visibility_m": profile.visibility_m,
            "snow_depth_m": profile.snow_depth_m,
        }
    )
    root = env.data.joint("floating_base_joint").qpos
    options = build_route_options(
        chapter.mode,
        profile,
        start_xy=(float(root[0]), float(root[1])),
    )
    planner = GeminiRoutePlanner(offline=offline)
    yaw = yaw_from_wxyz(root[3:7])
    estimate = AcousticBeaconSensor().sense(
        robot_xy=(float(root[0]), float(root[1])),
        robot_yaw_rad=yaw,
        source_xy=env.rescue_target_xy,
    )
    acoustic = AcousticPlanningObservation(
        bearing_rad=estimate.bearing_rad,
        confidence=estimate.confidence,
        coarse_range_m=estimate.range_m,
    )
    planning_frame = None if offline else env.front_camera_jpeg()
    route = planner.select(
        chapter.mode,
        options,
        image_jpeg=planning_frame,
        acoustic_observation=acoustic,
    )
    camera_bytes = len(planning_frame) if planning_frame is not None else 0
    print(
        f"VIDEO PLAN {chapter.title}: {route.route_id} risk={route.aggregate_risk:.4f}",
        file=sys.stderr,
        flush=True,
    )
    print(f"  reason={_safe_line(planner.last_reason)}", file=sys.stderr, flush=True)
    controller = AutonomousMujocoController(
        env=env,
        mode=chapter.mode,
        route=route,
        planner=planner,
        audit_log=audit_log,
        planning_frame_bytes=camera_bytes,
        spatial_audio=SpatialAudioSettings(acoustic_localization=True),
        simulation_id=f"video-{chapter.mode}",
    )
    return Mission(
        chapter=chapter,
        env=env,
        controller=controller,
        planner=planner,
        route_id=route.route_id,
        aggregate_risk=route.aggregate_risk,
        planning_camera_bytes=camera_bytes,
    )


def _ffmpeg_command(ffmpeg: str, output: Path, *, fps: int) -> list[str]:
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{WIDTH}x{HEIGHT}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]


def _render_action_frame(renderer: mujoco.Renderer, mission: Mission) -> np.ndarray:
    root = mission.env.data.joint("floating_base_joint").qpos
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (float(root[0]), float(root[1]), float(root[2]) + 0.05)
    camera.distance = mission.chapter.camera_distance
    camera.azimuth = 135.0
    camera.elevation = -12.0
    options = mujoco.MjvOption()
    options.geomgroup[4] = 0
    renderer.update_scene(mission.env.data, camera=camera, scene_option=options)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = True
    return np.asarray(renderer.render(), dtype=np.uint8)


def render_video(
    output: Path,
    *,
    offline: bool = False,
    force: bool = False,
    fps: int = FPS,
) -> Path:
    if fps <= 0 or fps > 60:
        raise ValueError("fps must be between 1 and 60")
    if output.exists() and not force:
        raise FileExistsError(f"output already exists: {output}; pass --force to replace it")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required; install it with: brew install ffmpeg")
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_log = output.parent / "everest-g1-video-events.jsonl"
    missions: list[Mission] = []
    try:
        for chapter in CHAPTERS:
            missions.append(_prepare_mission(chapter, offline=offline, audit_log=audit_log))
    except Exception:
        for mission in missions:
            mission.controller.close()
        raise

    with tempfile.NamedTemporaryFile(
        prefix=f".{output.stem}-",
        suffix=".mp4",
        dir=output.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            _ffmpeg_command(ffmpeg, temporary_path, fps=fps),
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None
        title_frames = round(TITLE_SECONDS * fps)
        total_frames = sum(round(mission.chapter.duration_seconds * fps) for mission in missions)
        written = 0

        for mission in missions:
            chapter_frames = round(mission.chapter.duration_seconds * fps)
            action_frames = chapter_frames - title_frames
            card = np.ascontiguousarray(title_card(mission.chapter, mission))
            for _ in range(title_frames):
                process.stdin.write(card.tobytes())
                written += 1

            renderer = mujoco.Renderer(mission.env.model, height=HEIGHT, width=WIDTH)
            step_accumulator = 0.0
            try:
                for frame_index in range(action_frames):
                    step_accumulator += (
                        mission.chapter.simulation_speed
                        / fps
                        / mission.env.config.simulation.timestep
                    )
                    steps = math.floor(step_accumulator)
                    step_accumulator -= steps
                    for _ in range(steps):
                        command = mission.controller.update(mission.env)
                        mission.env.step(command)
                    frame = _render_action_frame(renderer, mission)
                    elapsed_s = TITLE_SECONDS + (frame_index + 1) / fps
                    composed = np.ascontiguousarray(action_overlay(frame, mission, elapsed_s))
                    process.stdin.write(composed.tobytes())
                    written += 1
                    if frame_index and frame_index % (5 * fps) == 0:
                        print(
                            f"  {mission.chapter.title}: {frame_index / fps:.0f}s action rendered",
                            file=sys.stderr,
                            flush=True,
                        )
            finally:
                renderer.close()

        if written != total_frames:
            raise RuntimeError(
                f"video frame contract failed: wrote {written}, expected {total_frames}"
            )
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed ({return_code}): {stderr[-1000:]}")
        temporary_path.replace(output)
    except Exception:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        for mission in missions:
            mission.controller.close()
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offline-plan", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args(argv)
    try:
        output = render_video(
            args.output.resolve(),
            offline=args.offline_plan,
            force=args.force,
            fps=args.fps,
        )
    except (FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
