import json
from pathlib import Path

import pytest

from everest_g1.beacon import (
    ARM_VALUE,
    BeaconCallWorker,
    BeaconConfigurationError,
    BeaconSettings,
    JsonlAuditLog,
)
from everest_g1.models import RescueObservation


def test_live_call_requires_flag_and_exact_environment_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BEACON_API_TOKEN", "test-token")
    monkeypatch.setenv("EVEREST_ARM_LIVE_CALL", ARM_VALUE)
    BeaconSettings.from_env(arm_requested=True).validate()

    with pytest.raises(BeaconConfigurationError):
        BeaconSettings.from_env(arm_requested=False).validate()

    monkeypatch.setenv("EVEREST_ARM_LIVE_CALL", "yes")
    with pytest.raises(BeaconConfigurationError):
        BeaconSettings.from_env(arm_requested=True).validate()


def test_token_is_required_when_armed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVEREST_ARM_LIVE_CALL", ARM_VALUE)
    monkeypatch.delenv("BEACON_API_TOKEN", raising=False)

    with pytest.raises(BeaconConfigurationError, match="BEACON_API_TOKEN"):
        BeaconSettings.from_env(arm_requested=True).validate()


def test_worker_submits_only_once_and_log_has_no_secret_or_phone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import everest_g1.beacon as beacon_module

    settings = BeaconSettings("https://beacon.test", "very-secret", armed=True)
    log_path = tmp_path / "events.jsonl"
    calls: list[RescueObservation] = []

    def fake_post(_settings: BeaconSettings, observation: RescueObservation) -> dict[str, object]:
        calls.append(observation)
        return {"incident_id": "inc-1", "status": "queued"}

    monkeypatch.setattr(beacon_module, "post_incident", fake_post)
    worker = BeaconCallWorker(settings, JsonlAuditLog(log_path))
    observation = RescueObservation("sim-1", 0.12)
    assert observation.observed_state == "motionless_adult_in_snow"
    assert worker.submit_once(observation)
    assert not worker.submit_once(observation)
    worker.close(timeout_s=2)

    assert calls == [observation]
    text = log_path.read_text()
    assert "very-secret" not in text
    assert "+15550101234" not in text
    records = [json.loads(line) for line in text.splitlines()]
    assert [record["event"] for record in records] == ["call_queued", "call_dispatched"]
