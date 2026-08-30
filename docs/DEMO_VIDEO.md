# Autonomous demo video

Render one exact 45-second, 1280×720 H.264 video containing the three
autonomous MuJoCo chapters:

1. `02 RESCUE` — 15 seconds
2. `03 CARRY` — 18 seconds
3. `04 SCAN` — 12 seconds, one concise pass

Each chapter starts with a title card. The action view includes the validated
route, current mission stage, and Gemini's returned rationale. The renderer
uses mode-specific playback rates so each chapter reaches its key mission
transition within its 15-second allocation. Scan is limited to one concise pass.
The renderer does not arm BeaconCall.

```bash
cd /Users/kaushiksivakumar/workspace/everest-g1
./scripts/render_autonomy_demo.sh
```

Output:

```text
runtime/everest-g1-autonomy-demo.mp4
```

For a deterministic render without Gemini API calls:

```bash
./scripts/render_autonomy_demo.sh --offline-plan
```

Use `--force` to deliberately replace an existing output. The result is silent
so a separately recorded controller/BeaconCall chapter can retain its original
voice-call audio when the four chapters are merged. Inspect that source clip
before merging; its dimensions, duration, frame rate, and audio tracks determine
the final normalization command.
