import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cloud_scripts_parse_as_bash() -> None:
    scripts = [
        *sorted((ROOT / "cloud").glob("*.sh")),
        *sorted((ROOT / "stacks").glob("*/*.sh")),
    ]
    assert scripts
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_three_isolated_stack_entrypoints_and_isaac_camera_flag_exist() -> None:
    assert {path.name for path in (ROOT / "stacks").iterdir() if path.is_dir()} == {
        "groot",
        "isaac_lab",
        "sonic",
    }
    isaac_wrapper = (ROOT / "cloud/run_brev_rescue.sh").read_text()
    policy = (ROOT / "src/everest_g1/isaac/policy.py").read_text()

    assert "--enable_cameras" in isaac_wrapper
    assert "isaac_front_camera_jpeg(observation)" in policy
    assert 'simulator="isaac_lab"' in policy


def test_upstream_pins_are_full_commit_shas() -> None:
    pins = {}
    for line in (ROOT / "cloud/pins.env").read_text().splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            pins[key] = value

    assert set(pins) == {
        "ISAACLAB_ARENA_SHA",
        "ISAACLAB_SHA",
        "ISAAC_GROOT_SHA",
        "ISAAC_TELEOP_SHA",
        "GROOT_WBC_SHA",
    }
    assert all(re.fullmatch(r"[0-9a-f]{40}", value) for value in pins.values())


def test_modal_contract_uses_real_sonic_embodiment_and_no_literal_credentials() -> None:
    source = (ROOT / "cloud/modal_app.py").read_text()

    assert "UNITREE_G1_SONIC" in source
    assert "nvcr.io/nvidia/isaac-lab:3.0.0-beta2" in source
    assert "nvcr.io/nvidia/isaac-lab:2.3.2" in source
    assert "modal.Secret.from_name" in source
    assert "token-secret" not in source
    assert not re.search(r"\b(?:ak|as)-[A-Za-z0-9]{16,}\b", source)


def test_removed_sponsor_branding_does_not_return() -> None:
    checked_files = [
        ROOT / "README.md",
        ROOT / "SECURITY.md",
        *sorted((ROOT / "docs").glob("*.md")),
    ]
    text = "\n".join(path.read_text() for path in checked_files).lower()

    assert "trueforge" not in text
    assert "truefoundry" not in text
    assert "qodo" not in text
