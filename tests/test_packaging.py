import subprocess
import sys
import zipfile
from pathlib import Path


def test_built_wheel_contains_project_and_unitree_license_notices(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "hatchling",
            "build",
            "-t",
            "wheel",
            "-d",
            str(tmp_path),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(tmp_path.glob("everest_g1-*.whl"))

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()

    assert "everest_g1/__init__.py" in names
    assert "everest_g1/isaac/assets/downed_person_proxy.usda" in names
    assert "summit_sentinel/__init__.py" in names
    assert any(name.endswith(".dist-info/licenses/LICENSE") for name in names)
    assert any(name.endswith(".dist-info/licenses/THIRD_PARTY_NOTICES.md") for name in names)
    assert any(
        name.endswith(".dist-info/licenses/docs/third_party/UNITREE_RL_GYM_LICENSE.txt")
        for name in names
    )
