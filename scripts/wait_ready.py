"""Poll the endpoint until a worker is ready (or something goes wrong).

Runpod stages cached model weights onto a host before the first worker can
serve. That is not billed, but it can take tens of minutes for a 58 GB repo,
so this watches the /health endpoint rather than making anyone refresh a page.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests
from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parent.parent
env = dotenv_values(REPO_ROOT / ".env")
EID, KEY = env["RUNPOD_ENDPOINT_ID"], env["RUNPOD_API_KEY"]
HEADERS = {"Authorization": f"Bearer {KEY}"}

POLL_SECONDS = 60
MAX_MINUTES = 60


def main() -> int:
    started = time.time()
    last = None
    while time.time() - started < MAX_MINUTES * 60:
        try:
            r = requests.get(
                f"https://api.runpod.ai/v2/{EID}/health", headers=HEADERS, timeout=30
            )
            w = r.json().get("workers", {})
        except Exception as exc:  # noqa: BLE001
            print(f"[{time.strftime('%H:%M:%S')}] poll failed: {exc}", flush=True)
            time.sleep(POLL_SECONDS)
            continue

        summary = ", ".join(f"{k}={v}" for k, v in sorted(w.items()) if v)
        if summary != last:
            print(f"[{time.strftime('%H:%M:%S')}] workers: {summary or 'all zero'}", flush=True)
            last = summary

        if w.get("unhealthy"):
            print("UNHEALTHY worker detected - stopping so we can read the logs.", flush=True)
            return 2
        if w.get("ready") or w.get("idle") or w.get("running"):
            mins = (time.time() - started) / 60
            print(f"WORKER READY after {mins:.1f} min of staging.", flush=True)
            return 0

        time.sleep(POLL_SECONDS)

    print(f"Still not ready after {MAX_MINUTES} min.", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
