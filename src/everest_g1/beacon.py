"""Fail-closed, one-shot BeaconCall client isolated from the control loop."""

from __future__ import annotations

import json
import os
import queue
import threading
import urllib.error
import urllib.request
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from everest_g1.models import RescueObservation

ARM_VALUE = "ARM-LIVE-CALL"


class BeaconConfigurationError(RuntimeError):
    """Raised when a live call was requested without complete guarded configuration."""


@dataclass(frozen=True)
class BeaconSettings:
    api_url: str
    api_token: str
    armed: bool
    timeout_s: float = 45.0

    @classmethod
    def from_env(cls, *, arm_requested: bool) -> BeaconSettings:
        armed = arm_requested and os.getenv("EVEREST_ARM_LIVE_CALL") == ARM_VALUE
        return cls(
            api_url=os.getenv("BEACON_API_URL", "http://127.0.0.1:8000").rstrip("/"),
            api_token=os.getenv("BEACON_API_TOKEN", ""),
            armed=armed,
        )

    def validate(self) -> None:
        if not self.armed:
            raise BeaconConfigurationError(
                "live call is not armed; pass --arm-live-call and set "
                f"EVEREST_ARM_LIVE_CALL={ARM_VALUE}"
            )
        if not self.api_token:
            raise BeaconConfigurationError(
                "BEACON_API_TOKEN is required when live calling is armed"
            )
        if not self.api_url.startswith(("http://", "https://")):
            raise BeaconConfigurationError("BEACON_API_URL must use http:// or https://")


class JsonlAuditLog:
    """Append redacted status events without ever recording credentials or a phone number."""

    def __init__(self, path: Path = Path("runtime/everest-g1-events.jsonl")) -> None:
        self.path = path
        self._lock = threading.Lock()

    def write(self, event: str, **fields: object) -> None:
        record = {"timestamp": datetime.now(UTC).isoformat(), "event": event, **fields}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            _write_json_line(stream, record)


def _write_json_line(stream: TextIO, record: dict[str, object]) -> None:
    stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def post_incident(settings: BeaconSettings, observation: RescueObservation) -> dict[str, object]:
    """Submit one authenticated, idempotent incident to BeaconCall."""

    settings.validate()
    payload = json.dumps(observation.as_payload()).encode("utf-8")
    request = urllib.request.Request(
        f"{settings.api_url}/api/incidents/outbound-call",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {settings.api_token}",
            "Content-Type": "application/json",
            "Idempotency-Key": f"everest-{observation.simulation_id}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=settings.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"BeaconCall returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"BeaconCall is unreachable: {exc.reason}") from exc


class BeaconCallWorker:
    """One-slot worker: submission never blocks the Isaac control loop."""

    def __init__(self, settings: BeaconSettings, audit_log: JsonlAuditLog | None = None) -> None:
        self.settings = settings
        self.audit_log = audit_log or JsonlAuditLog()
        self._queue: queue.Queue[RescueObservation | None] = queue.Queue(maxsize=1)
        self._submitted = False
        self._thread = threading.Thread(target=self._run, name="beacon-call", daemon=True)
        self._thread.start()

    @property
    def submitted(self) -> bool:
        return self._submitted

    def submit_once(self, observation: RescueObservation) -> bool:
        if self._submitted:
            return False
        self.settings.validate()
        self._submitted = True
        self.audit_log.write(
            "call_queued",
            simulation_id=observation.simulation_id,
            distance_m=round(observation.distance_m, 3),
        )
        self._queue.put_nowait(observation)
        return True

    def _run(self) -> None:
        observation = self._queue.get()
        if observation is None:
            return
        try:
            result = post_incident(self.settings, observation)
            incident = result.get("incident")
            incident_fields = incident if isinstance(incident, dict) else {}
            self.audit_log.write(
                "call_dispatched",
                simulation_id=observation.simulation_id,
                incident_id=incident_fields.get("id", result.get("incident_id")),
                status=incident_fields.get("status", result.get("status")),
            )
        except Exception as exc:  # The control loop must survive network/service failures.
            self.audit_log.write(
                "call_failed",
                simulation_id=observation.simulation_id,
                error_type=type(exc).__name__,
                error=str(exc)[:500],
            )

    def close(self, timeout_s: float = 0.25) -> None:
        if not self._submitted:
            with suppress(queue.Full):
                self._queue.put_nowait(None)
        self._thread.join(timeout=timeout_s)


def new_simulation_id() -> str:
    return uuid.uuid4().hex
