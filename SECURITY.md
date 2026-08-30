# Security and deployment boundaries

## Authority model

The active simulator process owns physics, joint targets, motion bounds, and the
proximity latch. External services are weaker and asynchronous:

1. simulator stop/latch
2. bounded local policy
3. one-shot BeaconCall enqueue
4. Bright Data public context

No network response can clear the latch or add velocity. A BeaconCall failure is
logged while the G1 remains stopped.

## Live-call gate

A call requires all of the following:

- the launch command includes `--arm-live-call` (wrapper or MuJoCo) or
  `--arm_live_call` (Arena policy option);
- `EVEREST_ARM_LIVE_CALL` equals exactly `ARM-LIVE-CALL`;
- `BEACON_API_URL` uses HTTP(S);
- `BEACON_API_TOKEN` is non-empty;
- proximity remains within 0.15 m for 0.25 continuous seconds.

The Isaac wrapper prompts immediately before launch. MuJoCo requires its CLI
flag plus the same exact environment value. One process submits at most one
incident. BeaconCall independently authenticates and idempotently records it.

The destination number, LiveKit keys, SIP trunk ID, and Twilio credentials
belong only on the BeaconCall server. Do not add them to this repository.

## Secret handling

- `.env`, `runtime/`, checkpoints, datasets, `.fbx`, and generated `.usd` files
  are ignored.
- Do not put tokens in CLI flags, URLs printed to a console, issues, video, or
  shell history.
- Enter secrets with a hidden prompt or a cloud secret manager.
- The Bright Data remote MCP uses a credential-bearing URL internally. Errors
  intentionally discard the provider message so that URL cannot reach logs.
- JSONL audit records contain only simulation IDs, measured distance, camera
  capture byte count/status, incident ID/status, and bounded error text.
- Rotate any token pasted into chat or another durable transcript before use.

## Cloud isolation

Brev is the primary interactive lane. Modal is a headless probe/batch lane.
Both use pinned upstream checkouts. A failed import, GPU, renderer, or dependency
check must fail closed.

Arena and SONIC intentionally use separate environments because their released
Isaac Lab versions differ. Combining them into one interpreter is unsupported
and can silently invalidate evaluation.

## Model and asset provenance

Do not deserialize an untrusted model. Keep checkpoint source, digest, license,
upstream commit, and evaluation record together. The included MuJoCo TorchScript
policy is checked against the digest documented in `THIRD_PARTY_NOTICES.md`.

The Isaac human is a primitive proxy. The committed MuJoCo casualty is a
converted local Robot Nurse asset documented in `THIRD_PARTY_NOTICES.md`; verify
its redistribution terms before publishing a fork. Only convert and use another
external FBX when you have the necessary rights. Generated human assets stay
under ignored `runtime/` paths.

The armed MuJoCo path sends exactly one proximity-triggered JPEG to BeaconCall,
which forwards it to OpenAI for an observable scene description. It does not
send the image to LiveKit or Twilio. The vision output is untrusted quoted data,
cannot issue robot commands, and must not be treated as a medical assessment.

## Physical robot boundary

This release is simulation-only. It does not authorize low-level commands to a
physical G1, developer mode, a gantry, or an onboard computer. A physical
deployment needs its own facility review, watchdog, damping shutdown, joint
limits, network-loss behavior, and supervised commissioning evidence.

## Reporting

Do not publish active credentials or an exploitable endpoint. Revoke exposed
credentials first, preserve redacted evidence, then report the issue privately
to the repository owner and affected service.
