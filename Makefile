.PHONY: setup sim sim-beacon sim-macos sim-linux sim-headless mujoco-rescue mujoco-rescue-headless mujoco-rescue-audio mode-controller mode-gemini-rescue mode-gemini-carry mode-gemini-scan demo-video mcp rescue-dry rescue-audio brev-setup brev-run isaac-setup isaac-run groot-setup sonic-setup verify

SIM_LAUNCHER := $(if $(filter Darwin,$(shell uname -s)),mjpython,python)

setup:
	uv sync --extra dev

sim:
	uv run $(SIM_LAUNCHER) -m summit_sentinel --mode viewer --seconds 600 --joystick --joystick-index 0 --joystick-calibration runtime/dualsense.json --bridge-db runtime/summit.db --telemetry-hz 15

sim-beacon:
	@env_file=".env"; \
	if [ ! -f "$$env_file" ]; then env_file="../sim-g1-everest/.env"; fi; \
	test -f "$$env_file" || { echo "Missing .env with BEACON_API_URL and BEACON_API_TOKEN" >&2; exit 1; }; \
	echo "Loading Beacon configuration from $$env_file"; \
	set -a; . "$$env_file"; set +a; \
	EVEREST_ARM_LIVE_CALL=ARM-LIVE-CALL ./autonomy/run_controller.sh

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

mode-controller:
	./autonomy/run_controller.sh

mode-gemini-rescue:
	./autonomy/run_rescue.sh

mode-gemini-carry:
	./autonomy/run_carry.sh

mode-gemini-scan:
	./autonomy/run_scan.sh

demo-video:
	./scripts/render_autonomy_demo.sh

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

rescue-audio:
	uv run everest-g1 dry-run --spatial-audio --acoustic-localization

mujoco-rescue-audio:
	uv run $(SIM_LAUNCHER) -m summit_sentinel --mode viewer --seconds 60 --rescue --spatial-audio --acoustic-localization
