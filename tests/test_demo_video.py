import numpy as np

from everest_g1.autonomy import EnvironmentProfile, GeminiRoutePlanner, build_route_options
from everest_g1.demo_video import (
    CHAPTERS,
    FPS,
    HEIGHT,
    TITLE_SECONDS,
    TOTAL_SECONDS,
    WIDTH,
    Mission,
    title_card,
)


def test_storyboard_is_exactly_45_seconds_with_expected_chapters() -> None:
    assert len(CHAPTERS) == 3
    assert [chapter.title for chapter in CHAPTERS] == ["RESCUE", "CARRY", "SCAN"]
    assert [chapter.number for chapter in CHAPTERS] == ["02", "03", "04"]
    assert [chapter.duration_seconds for chapter in CHAPTERS] == [15.0, 18.0, 12.0]
    assert sum(chapter.duration_seconds for chapter in CHAPTERS) == TOTAL_SECONDS == 45.0
    assert sum(round(chapter.duration_seconds * FPS) for chapter in CHAPTERS) == 1350
    assert all(0.0 < TITLE_SECONDS < chapter.duration_seconds for chapter in CHAPTERS)


def test_title_card_has_video_contract() -> None:
    chapter = CHAPTERS[0]
    planner = GeminiRoutePlanner(offline=True)
    route = planner.select(
        chapter.mode,
        build_route_options(chapter.mode, EnvironmentProfile()),
        image_jpeg=None,
    )
    mission = Mission(
        chapter=chapter,
        env=None,  # type: ignore[arg-type]
        controller=None,  # type: ignore[arg-type]
        planner=planner,
        route_id=route.route_id,
        aggregate_risk=route.aggregate_risk,
        planning_camera_bytes=0,
    )

    frame = title_card(chapter, mission)

    assert frame.shape == (HEIGHT, WIDTH, 3)
    assert frame.dtype == np.uint8
    assert np.unique(frame.reshape(-1, 3), axis=0).shape[0] > 20
