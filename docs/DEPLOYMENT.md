# Deployment guide

> New to this? Read [EXPLAINED.md](../EXPLAINED.md) first. It covers the
> same ground in plain language, without assuming any background.

How this was deployed, reproducibly, from an empty account. Roughly 45 minutes
of work plus about 30 minutes of waiting.

## Prerequisites

- A Runpod account with credit
- A Hugging Face account
- A GitHub account
- Python 3.11+ locally (only for the test scripts and running the UI)

Docker is **not** required. Runpod builds the image from GitHub. A local
Docker build is useful for verification, and the commands are at the end.

## 1. Accept the FLUX.1-dev licence

FLUX.1-dev is a gated repository. The acceptance is recorded against your
**Hugging Face account**, not against a token - a valid token belonging to an
account that has not accepted the licence returns 403, and Runpod surfaces
that as a generic worker failure.

1. Log in at huggingface.co with the account that owns your token
2. Open <https://huggingface.co/black-forest-labs/FLUX.1-dev>
3. Click **Agree and access repository**

Verify it programmatically rather than trusting the page:

```bash
python scripts/verify_hf_access.py
```

Expect `PASS - Runpod will be able to download FLUX.1-dev with this token.`

## 2. Create the Hugging Face token

Settings → Access Tokens → **Create new token**, type **Read**. This token is
what Runpod uses to fetch the gated weights.

A second **Write** token is only needed if you deploy the UI to a Hugging Face
Space (see `scripts/deploy_space.py`, and note the caveat there).

## 3. Create the Runpod API key

Runpod console → **Settings → API Keys → Create API Key**, permissions
**All**. It is displayed once and never again.

## 4. Connect Runpod to GitHub

Runpod console → **Settings → Connections → GitHub → Connect**. Grant access
to this repository only.

Runpod then clones the repo, builds the image on its own infrastructure, and
stores it in its private registry - so no large image is ever pushed from a
local machine.

## 5. Create the endpoint

**Serverless → New Endpoint → Import Git Repository**, select the repo,
branch `main`, Dockerfile path `/Dockerfile`.

A pre-flight warning reading *"Could not find runpod.serverless.start() in
your repo"* is a false positive: the scanner looks in the repository root,
while the entry point is `worker/handler.py`, copied to `/app/` by the
Dockerfile and launched by its `CMD`. Continue.

### Configuration

| Setting | Value | Why |
|---|---|---|
| Endpoint type | **Queue** | The worker is a handler function, not an HTTP server |
| GPU priority | 48 GB PRO → 48 GB → 80 GB PRO | 34 GB of bf16 weights needs 48 GB; three tiers avoid throttling |
| Enabled GPU types | untick **PRO 6000 MIG 48GB** | Blackwell; the pinned torch 2.4.0 has no kernels for it |
| Active workers | **0** | One always-on 48 GB worker costs ~$42/day |
| Max workers | **1** (2 for a demo) | Concurrency cap and cost ceiling |
| Idle timeout | **5s** | Billed while idle; raise only while recording |
| Execution timeout | **900s** | Default 600 is tight for 50 steps on a cold worker |
| FlashBoot | **on** | Free; speeds up waking a stopped worker |
| Container disk | **60 GB** | Headroom for the runtime-download fallback |
| CUDA versions | **12.4 and above** | The image is built on CUDA 12.4; older hosts cannot run it |
| Data centers | **all** | Restricting shrinks the GPU pool |
| Network volume | **none** | Would pin the endpoint to one datacentre; model caching is faster |

### The setting that matters most

**Cached model**: `https://huggingface.co/black-forest-labs/FLUX.1-dev`, with
the read token in the **Hugging Face access token** field that appears beneath
it.

Store the token as a Runpod **Secret** rather than a plain environment
variable - the console warns that environment variables are not encrypted.

### Environment variables

| Key | Value | Purpose |
|---|---|---|
| `HF_TOKEN` | secret reference | Feeds the runtime-download fallback in `pipeline.py` |
| `MODEL_ID` | `black-forest-labs/FLUX.1-dev` | Explicit rather than relying on the default |
| `RUNPOD_INIT_TIMEOUT` | `800` | Raises the 7-minute cold-start ceiling |

Then **Deploy Endpoint**.

## 6. Wait for the build and the staging

The **Builds** tab moves through Pending → Building → Uploading → Testing →
Completed, in roughly 8-20 minutes.

The endpoint then shows **Initializing** while Runpod stages the model onto a
host. This took **27.7 minutes** and is **not billed**. Watch it with:

```bash
python scripts/wait_ready.py
```

## 7. Verify from outside the console

```bash
python scripts/test_endpoint.py "a red fox in deep snow" --steps 28 --seed 42
```

This submits via `/run`, polls `/status`, prints each progress phase, and
writes the PNG to `outputs/`. Check that the output reports
`weights_source: runpod_cache` - if it says `runtime_download`, the cached
model is not configured correctly and each cold start will be billed for a
34 GB download.

Or with curl:

```bash
curl -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"prompt":"a red fox in deep snow","num_inference_steps":28}}'

curl https://api.runpod.ai/v2/$ENDPOINT_ID/status/$JOB_ID \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

A Postman collection is in [`postman/`](../postman).

## 8. Deploy the UI

Render dashboard → **New + → Blueprint** → select this repository. Render
reads [`render.yaml`](../render.yaml) and prompts for the three values marked
`sync: false`:

| Key | Value |
|---|---|
| `RUNPOD_API_KEY` | your Runpod key |
| `RUNPOD_ENDPOINT_ID` | your endpoint id |
| `APP_PASSWORD` | the demo password |

First build takes 3-6 minutes.

> Hugging Face Spaces was the original target and is no longer viable on a free
> account: creating a Gradio Space on `cpu-basic` returns 402 (*"requires a PRO
> subscription"*), the free-account ZeroGPU allowance also returns 402, and a
> private Space returns 404 to everyone but its owner so a reviewer could not
> open it. `scripts/deploy_space.py` is kept for anyone who does have PRO.

## Running locally

```bash
python -m venv .venv
.venv/Scripts/pip install -r app/requirements.txt   # Linux/macOS: .venv/bin/pip
cp app/.env.example .env      # then fill it in
python app/app.py
```

Without credentials, mock mode needs nothing at all:

```bash
RUNPOD_MOCK=1 python app/app.py
```

## Verifying the worker without a GPU

```bash
python scripts/local_handler_test.py     # 44 assertions, stub pipeline, no GPU
```

And the container itself, on any Linux machine with Docker:

```bash
docker build --platform linux/amd64 -t flux-worker:test .
docker run --rm -e FLUX_STUB=1 flux-worker:test
```

The build runs an import smoke test, so a version incompatibility fails the
build rather than the worker. `FLUX_STUB=1` exercises the handler end to end
without weights; without it the container exits with a clear
*"No CUDA device available"* rather than silently downloading 34 GB.

## Operational notes

- **Every push to `main` rebuilds the image**, including documentation-only
  commits. Batch commits before a demo.
- **Runpod scales idle endpoints down**: max workers drops to 2 after 3 days
  without requests and to 0 after 7. Raise it in the console to revive.
- **Render free instances sleep** after 15 minutes, adding ~50s to the first
  request.
- **Rotate credentials** when the deployment is retired: revoke the Runpod API
  key, the Hugging Face tokens, and delete the endpoint.
