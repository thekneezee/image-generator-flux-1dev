"""Create (or update) a Hugging Face Space for the chat UI.

NOT the deployment path this project uses - kept because it works for anyone
with a Hugging Face PRO subscription, and because the reason it was abandoned
is worth recording.

As of 2026 Hugging Face requires a paid plan to create Gradio Spaces: cpu-basic
returns 402 ("hosting Gradio and Docker Spaces on free cpu-basic requires a PRO
subscription") and the free-account ZeroGPU allowance returns 402 as well ("you
must be subscribed to PRO to host Spaces with ZeroGPU"). Private Spaces are
also unusable for a demo regardless of plan: they return 404 to everyone except
the owner, so a reviewer could not open one.

The live demo is therefore deployed to Render instead - see render.yaml.
The original docstring follows.

Create (or update) the Hugging Face Space that hosts the chat UI.

Hardware note: Hugging Face now requires a paid plan to create Gradio Spaces
on cpu-basic. Free personal accounts may host up to two Gradio Spaces on
ZeroGPU, so that is the default here. This app never allocates a GPU - all
inference happens on the Runpod endpoint - so ZeroGPU's quota is not consumed.

Visibility note: private Spaces return 404 to everyone except the owner, so
the app would be unreachable by a reviewer. Public is the default; the app
itself is password-gated and credentials live in encrypted Space secrets.

The Space runs on free CPU hardware and contains no model - it is a thin
client. All GPU work happens on the Runpod endpoint, so hosting the interface
costs nothing.

Credentials are set as Space *secrets*, which are encrypted at rest and
injected as environment variables at runtime. They never reach the browser,
which is the whole reason the UI is server-rendered Python rather than
JavaScript calling Runpod directly.

Usage:  python scripts/deploy_space.py [--public]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = REPO_ROOT / "app"
SPACE_NAME = "flux-serverless-demo"

# (local path, path inside the Space). SPACE_README.md becomes the Space's
# README.md, whose YAML frontmatter tells Hugging Face how to build it.
FILES = [
    (APP_DIR / "app.py", "app.py"),
    (APP_DIR / "runpod_client.py", "runpod_client.py"),
    (APP_DIR / "theme.py", "theme.py"),
    (APP_DIR / "requirements.txt", "requirements.txt"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private", action="store_true",
                        help="Create the Space private. Note: private Spaces return "
                             "404 to everyone but the owner, so the app is not "
                             "reachable by a reviewer.")
    parser.add_argument("--hardware", default="zero-a10g",
                        help="Space hardware. Free personal accounts can host Gradio "
                             "Spaces on ZeroGPU; cpu-basic now requires PRO.")
    args = parser.parse_args()

    env = dotenv_values(REPO_ROOT / ".env")
    token = env.get("HF_TOKEN_WRITE", "").strip()
    if not token:
        print("FATAL: HF_TOKEN_WRITE is not set in .env")
        return 1

    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{SPACE_NAME}"
    print(f"account : {user}")
    print(f"space   : {repo_id}  "
          f"({'private' if args.private else 'public'}, {args.hardware})")

    for local, _ in FILES:
        if not local.exists():
            print(f"FATAL: missing {local}")
            return 1

    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=args.private,
        space_hardware=args.hardware,
        exist_ok=True,
    )
    print("repo    : ready")

    for local, remote in FILES:
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=repo_id,
            repo_type="space",
            commit_message=f"Deploy {remote}",
        )
        print(f"uploaded: {remote}")

    secrets = {
        "RUNPOD_API_KEY": env.get("RUNPOD_API_KEY", ""),
        "RUNPOD_ENDPOINT_ID": env.get("RUNPOD_ENDPOINT_ID", ""),
        "APP_PASSWORD": env.get("APP_PASSWORD", ""),
    }
    missing = [k for k, v in secrets.items() if not v]
    if missing:
        print(f"FATAL: these are empty in .env: {', '.join(missing)}")
        return 1

    for key, value in secrets.items():
        api.add_space_secret(repo_id=repo_id, key=key, value=value)
        print(f"secret  : {key} set ({len(value)} chars, value not shown)")

    # Not a secret - a tuning knob, useful to see and change in the UI.
    api.add_space_variable(repo_id=repo_id, key="MAX_GENERATIONS", value="25")
    api.add_space_variable(repo_id=repo_id, key="RUNPOD_MOCK", value="0")
    print("vars    : MAX_GENERATIONS=25, RUNPOD_MOCK=0")

    print()
    print(f"Space URL: https://huggingface.co/spaces/{repo_id}")
    print("It will build for a minute or two, then ask for the password.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HfHubHTTPError as exc:
        print(f"Hugging Face rejected the request: {exc}")
        raise SystemExit(1) from exc
