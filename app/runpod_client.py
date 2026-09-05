"""Thin client for a Runpod queue-based Serverless endpoint.

Deliberately uses /run + /status polling rather than /runsync:

* /runsync waits inline with a 90 second default ceiling. A cold worker
  loading 34 GB of weights can exceed that, and the request would fail even
  though generation was progressing fine.
* Polling /status surfaces the progress updates our handler emits on every
  denoising step. That is what lets the UI narrate generation instead of
  showing a spinner.
* /run results are retained for 30 minutes, so a slow client cannot lose a
  finished image.

A MockClient with the same interface backs RUNPOD_MOCK=1, so the UI can be
developed and demonstrated without spending GPU credit.
"""

from __future__ import annotations

import base64
import io
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Iterator

import requests

BASE_URL = "https://api.runpod.ai/v2"

# Runpod's console quotes per-hour rates by tier. Converted to per-second and
# keyed by the GPU name torch reports, so the UI prices a real request rather
# than guessing.
GPU_HOURLY_RATES = {
    # 48 GB PRO tier - $1.75/hr
    "l40s": 1.75,
    "l40": 1.75,
    "rtx 6000 ada": 1.75,
    "6000 ada": 1.75,
    # 48 GB tier - $1.22/hr
    "a6000": 1.22,
    "a40": 1.22,
    # 80 GB PRO tier - $4.79/hr
    "h100": 4.79,
}
DEFAULT_HOURLY_RATE = 1.75

TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


def hourly_rate_for(gpu_name: str | None) -> float:
    if not gpu_name:
        return DEFAULT_HOURLY_RATE
    lowered = gpu_name.lower()
    # Check longer keys first so "rtx 6000 ada" wins over a bare substring.
    for token in sorted(GPU_HOURLY_RATES, key=len, reverse=True):
        if token in lowered:
            return GPU_HOURLY_RATES[token]
    return DEFAULT_HOURLY_RATE


def cost_for(gpu_name: str | None, seconds: float) -> float:
    return hourly_rate_for(gpu_name) / 3600.0 * seconds


class RunPodError(RuntimeError):
    """Carries a message already phrased for a human."""


@dataclass
class JobUpdate:
    """One poll of a job's state."""

    status: str
    progress: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    delay_ms: int | None = None
    execution_ms: int | None = None
    worker_id: str | None = None


class RunPodClient:
    def __init__(self, api_key: str, endpoint_id: str, poll_interval: float = 1.0):
        if not api_key or not endpoint_id:
            raise RunPodError(
                "RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID must both be set. "
                "Copy .env.example to .env and fill them in."
            )
        self.endpoint_id = endpoint_id
        self.poll_interval = poll_interval
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )

    # -- plumbing ---------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{BASE_URL}/{self.endpoint_id}/{path}"

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """One request, with exponential backoff on the failures worth retrying."""
        delay = 1.0
        for attempt in range(4):
            try:
                response = self._session.request(
                    method, self._url(path), timeout=60, **kwargs
                )
            except requests.RequestException as exc:
                if attempt == 3:
                    raise RunPodError(f"Could not reach Runpod: {exc}") from exc
                time.sleep(delay)
                delay *= 2
                continue

            if response.status_code == 200:
                return response.json()

            # 429 and 5xx are transient. Everything else is our fault and
            # retrying would only waste time.
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 3:
                    raise RunPodError(
                        self._explain(response.status_code, response.text)
                    )
                time.sleep(delay)
                delay *= 2
                continue

            raise RunPodError(self._explain(response.status_code, response.text))

        raise RunPodError("Runpod request failed after several attempts.")

    @staticmethod
    def _explain(code: int, body: str) -> str:
        return {
            401: "Runpod rejected the API key (401). Check RUNPOD_API_KEY.",
            403: "Runpod denied access (403). The key may lack permission for this endpoint.",
            404: "Endpoint not found (404). Check RUNPOD_ENDPOINT_ID.",
            429: "Rate limited by Runpod (429) - too many requests in a short window.",
        }.get(code, f"Runpod returned HTTP {code}: {body[:300]}")

    # -- operations -------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._request("GET", "health")

    def submit(self, payload: dict[str, Any]) -> str:
        data = self._request("POST", "run", json={"input": payload})
        job_id = data.get("id")
        if not job_id:
            raise RunPodError(f"Runpod did not return a job id: {data}")
        return job_id

    def status(self, job_id: str) -> JobUpdate:
        data = self._request("GET", f"status/{job_id}")
        status = data.get("status", "UNKNOWN")
        raw_output = data.get("output")

        # Runpod does not expose progress under its own key. Verified against
        # the live API: progress_update() payloads arrive in `output` while the
        # job is still running, and `output` only becomes the handler's return
        # value once the job reaches a terminal state. Distinguish them by the
        # "phase" marker our handler sets.
        progress = data.get("progress") or data.get("progressUpdate")
        output = raw_output
        if (
            status not in TERMINAL_STATES
            and isinstance(raw_output, dict)
            and "phase" in raw_output
        ):
            progress = raw_output
            output = None

        return JobUpdate(
            status=status,
            progress=progress,
            output=output,
            delay_ms=data.get("delayTime"),
            execution_ms=data.get("executionTime"),
            worker_id=data.get("workerId"),
        )

    def cancel(self, job_id: str) -> dict[str, Any]:
        return self._request("POST", f"cancel/{job_id}")

    def poll(self, job_id: str, timeout: float = 900.0) -> Iterator[JobUpdate]:
        """Yield a JobUpdate about once a second until the job settles."""
        started = time.time()
        while True:
            update = self.status(job_id)
            yield update
            if update.status in TERMINAL_STATES:
                return
            if time.time() - started > timeout:
                raise RunPodError(
                    f"Gave up waiting after {timeout:.0f}s "
                    f"(job {job_id} still {update.status})."
                )
            time.sleep(self.poll_interval)


class MockClient:
    """Same interface as RunPodClient, but no network and no cost.

    Exists so the UI can be built without spending credit, and so a reviewer
    can explore the interface without Runpod credentials. It simulates a cold
    start on the first request, then per-step denoising progress.
    """

    def __init__(self, cold_start_s: float = 9.0):
        self.endpoint_id = "mock-endpoint"
        self.poll_interval = 0.25
        self._cold_start_s = cold_start_s
        self._jobs: dict[str, dict[str, Any]] = {}
        self._warm = False

    def health(self) -> dict[str, Any]:
        running = sum(1 for j in self._jobs.values() if j["status"] == "IN_PROGRESS")
        completed = sum(1 for j in self._jobs.values() if j["status"] == "COMPLETED")
        return {
            "jobs": {
                "completed": completed,
                "failed": 0,
                "inProgress": running,
                "inQueue": 0,
                "retried": 0,
            },
            "workers": {
                "idle": 1 if self._warm and not running else 0,
                "initializing": 0,
                "ready": 1 if self._warm else 0,
                "running": running,
                "throttled": 0,
                "unhealthy": 0,
            },
        }

    def submit(self, payload: dict[str, Any]) -> str:
        job_id = f"mock-{random.randint(10**8, 10**9)}"
        self._jobs[job_id] = {
            "status": "IN_QUEUE",
            "payload": payload,
            "started": time.time(),
            "cold": not self._warm,
        }
        return job_id

    def status(self, job_id: str) -> JobUpdate:
        job = self._jobs.get(job_id)
        if not job:
            raise RunPodError(f"Unknown mock job {job_id}")
        return JobUpdate(status=job["status"])

    def cancel(self, job_id: str) -> dict[str, Any]:
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = "CANCELLED"
        return {"id": job_id, "status": "CANCELLED"}

    def poll(self, job_id: str, timeout: float = 900.0) -> Iterator[JobUpdate]:
        job = self._jobs[job_id]
        payload = job["payload"]
        steps = int(payload.get("num_inference_steps", 28))
        queue_seconds = self._cold_start_s if job["cold"] else 0.6
        per_step = 0.09

        started = time.time()
        while time.time() - started < queue_seconds:
            if job["status"] == "CANCELLED":
                yield JobUpdate(status="CANCELLED")
                return
            yield JobUpdate(status="IN_QUEUE")
            time.sleep(self.poll_interval)

        job["status"] = "IN_PROGRESS"
        self._warm = True

        # Mirror the real handler's phase sequence, so the UI narration is
        # exercised in mock mode exactly as it will be in production.
        yield JobUpdate(
            status="IN_PROGRESS",
            progress={"phase": "encoding_prompt", "step": 0,
                      "total_steps": steps, "percent": 0},
        )
        time.sleep(0.5)

        for step in range(1, steps + 1):
            if job["status"] == "CANCELLED":
                yield JobUpdate(status="CANCELLED")
                return
            yield JobUpdate(
                status="IN_PROGRESS",
                progress={
                    "phase": "denoising",
                    "step": step,
                    "total_steps": steps,
                    "percent": round(100 * step / steps),
                },
            )
            time.sleep(per_step)

        yield JobUpdate(
            status="IN_PROGRESS",
            progress={"phase": "decoding", "step": steps,
                      "total_steps": steps, "percent": 100},
        )
        time.sleep(0.4)

        job["status"] = "COMPLETED"
        execution_ms = int((time.time() - started - queue_seconds) * 1000)
        yield JobUpdate(
            status="COMPLETED",
            output=self._fake_output(payload, execution_ms),
            delay_ms=int(queue_seconds * 1000),
            execution_ms=execution_ms,
            worker_id="mock-worker",
        )

    @staticmethod
    def _fake_output(payload: dict[str, Any], execution_ms: int) -> dict[str, Any]:
        from PIL import Image, ImageDraw

        width = int(payload.get("width", 1024))
        height = int(payload.get("height", 1024))
        seed = payload.get("seed", -1)
        seed = random.randint(0, 2**32 - 1) if seed is None or seed < 0 else int(seed)

        rng = random.Random(seed)
        base = (rng.randint(20, 90), rng.randint(20, 90), rng.randint(60, 140))
        image = Image.new("RGB", (width, height), base)
        draw = ImageDraw.Draw(image)

        # Deterministic from the seed, so "same seed gives the same image"
        # still demonstrates correctly in mock mode.
        for i in range(48):
            angle = i / 48 * math.tau + seed % 360
            radius = min(width, height) * (0.12 + 0.34 * ((i * (seed % 7 + 3)) % 11) / 11)
            cx = width / 2 + math.cos(angle) * radius
            cy = height / 2 + math.sin(angle) * radius
            dot = min(width, height) * 0.035
            shade = tuple(min(255, c + 70 + (i * 3) % 90) for c in base)
            draw.ellipse([cx - dot, cy - dot, cx + dot, cy + dot], fill=shade)

        # This placeholder must never be mistaken for a real generation.
        # A small caption was not enough - it fooled a first-time user - so
        # the label is now unmissable.
        try:
            from PIL import ImageFont

            big = ImageFont.load_default(size=max(28, width // 22))
            small = ImageFont.load_default(size=max(15, width // 46))
        except Exception:  # noqa: BLE001 - very old Pillow
            big = small = None

        band = height // 5
        overlay = Image.new("RGBA", (width, band), (10, 10, 14, 205))
        image.paste(overlay, (0, (height - band) // 2), overlay)
        cy = height // 2
        draw.text((width // 2, cy - band // 5), "MOCK MODE",
                  fill=(250, 204, 21), font=big, anchor="mm")
        draw.text((width // 2, cy + band // 6),
                  "placeholder pattern - no GPU, no model, not your prompt",
                  fill=(230, 230, 240), font=small, anchor="mm")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return {
            "images": [
                {
                    "image": base64.b64encode(buffer.getvalue()).decode("ascii"),
                    "format": "png",
                    "width": width,
                    "height": height,
                }
            ],
            "seed": seed,
            "generation_time": round(execution_ms / 1000, 2),
            "parameters": payload,
            "notes": [],
            "gpu": "NVIDIA L40S (simulated)",
            "vram_gb": 48.0,
            "weights_source": "mock",
            "model": "black-forest-labs/FLUX.1-dev",
            "worker_id": "mock-worker",
            "offload": False,
        }
