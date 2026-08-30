.PHONY: setup sim sim-macos sim-linux sim-headless mujoco-rescue mujoco-rescue-headless mcp rescue-dry brev-setup brev-run isaac-setup isaac-run groot-setup sonic-setup verify

SIM_LAUNCHER := $(if $(filter Darwin,$(shell uname -s)),mjpython,python)

setup:
	uv sync --extra dev

sim:
	uv run $(SIM_LAUNCHER) -m summit_sentinel --mode viewer --seconds 600 --joystick --joystick-index 0 --joystick-calibration runtime/dualsense.json --bridge-db runtime/summit.db --telemetry-hz 15

sim-macos:
	uv run mjpython -m summit_sentinel --mode viewer --seconds 600 --joystick --joystick-index 0 --joystick-calibration runtime/dualsense.json --bridge-db runtime/summit.db --telemetry-hz 15

sim-linux:
	uv run python -m summit_sentinel --mode viewer --seconds 600 --joystick --joystick-index 0 --joystick-calibration runtime/dualsense.json --bridge-db runtime/summit.db --telemetry-hz 15

sim-headless:
	uv run summit-sentinel --mode headless --seconds 600 --bridge-db runtime/summit.db --telemetry-hz 15 --json

mujoco-rescue:
	uv run $(SIM_LAUNCHER) -m summit_sentinel --mode viewer --seconds 60 --rescue

mujoco-rescue-headless:
	uv run summit-sentinel --mode headless --seconds 12 --rescue --json

mcp:
	uv run summit-sentinel-mcp --bridge-db runtime/summit.db --port 8000

rescue-dry:
	uv run everest-g1 dry-run

brev-setup:
	./cloud/brev_setup.sh

brev-run:
	./cloud/run_brev_rescue.sh

isaac-setup:
	./stacks/isaac_lab/setup.sh

isaac-run:
	./stacks/isaac_lab/run.sh

groot-setup:
	./stacks/groot/setup.sh

sonic-setup:
	./stacks/sonic/setup.sh

verify:
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest
