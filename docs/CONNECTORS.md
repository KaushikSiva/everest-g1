# External connectors

## BeaconCall / LiveKit

Everest G1 calls a single authenticated BeaconCall endpoint. The request carries
only an opaque simulation ID, observed-state string, and measured distance. The
phone number and telephony credentials remain server-side.

Environment:

```dotenv
BEACON_API_URL=https://YOUR-BEACON-HOST
BEACON_API_TOKEN=YOUR_DEDICATED_BEARER_TOKEN
EVEREST_ARM_LIVE_CALL=ARM-LIVE-CALL
```

Do not put these values in a committed `.env`. The live-call wrapper also
requires an explicit launch flag and typed confirmation.

## Bright Data

`everest_g1.bright_data` connects to Bright Data's hosted Streamable HTTP MCP
using `BRIGHT_DATA_API_TOKEN`. It hard-restricts the remote session to:

```text
search_engine,scrape_as_markdown
```

The adapter exposes public research only. It does not parse web text into
velocity, thresholds, joint targets, live-call arming, or emergency claims.
Provider errors are wrapped without their original messages because the MCP URL
contains the token.

## Cloud secrets

Use Brev's environment secret support or a hidden shell prompt. Modal training
expects a named `everest-huggingface` secret with `HF_TOKEN`. Modal account
credentials are configured through the Modal CLI and are never read by the app.
