"""Runpod Serverless handler for FLUX.1-dev text-to-image.

Request  ->  {"input": {"prompt": "...", "num_inference_steps": 28, ...}}
Response ->  {"images": [{"image": "<base64>", ...}], "seed": ..., ...}

Two design points worth knowing:

* The model is loaded at MODULE level, not inside handler(). Runpod keeps a
  worker alive between jobs, so this pays the 34 GB load once per worker
  rather than once per request.

* Every denoising step fires progress_update(). Polling /status then shows
  live progress, which is what lets the UI narrate the generation instead of
  showing a spinner. Updates are throttled so a 50-step run does not make 50
  API calls.
"""

from __future__ import annotations

import base64
import io
import os
import time
import traceback
from typing import Any

import runpod

from pipeline import load_pipeline
from schema import ValidationError, parse_request

# torch is always present in the container, but absent from the lightweight
# local venv used for stub-mode tests. Import defensively so the same file
# runs in both places.
try:
    import torch
except ImportError:  # pragma: no cover - only hit in local stub testing
    torch = None  # type: ignore[assignment]

# Minimum seconds between progress updates. 0.4s keeps a 28-step run under
# ~20 updates while still feeling continuous in the UI.
PROGRESS_INTERVAL_S = 0.4

JPEG_QUALITY = 92

WORKER_STARTED_AT = time.time()

# ---------------------------------------------------------------------------
# Module-level initialisation: runs once when the worker boots, so the 34 GB
# of weights are read once per worker rather than once per request.
#
# A failure here must NOT kill the process. Runpod's build pipeline starts the
# container during its testing stage, and a container that exits on import
# looks like a broken worker - which Runpod then retries for up to 7 days.
# So we record the failure, keep serving, and retry the load on first use.
# ---------------------------------------------------------------------------
PIPE = None
MODEL_INFO: dict[str, Any] = {}
LOAD_ERROR: str | None = None


def _try_load() -> None:
    global PIPE, MODEL_INFO, LOAD_ERROR
    try:
        PIPE, MODEL_INFO = load_pipeline()
        LOAD_ERROR = None
    except Exception as exc:  # noqa: BLE001
        PIPE, MODEL_INFO = None, {}
        LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        print(f"[handler] model load failed at startup: {LOAD_ERROR}")
        print("[handler] worker will stay up and retry on the first request")


_try_load()


def _progress(job: dict[str, Any], payload: dict[str, Any]) -> None:
    """Send a progress update, never letting a reporting failure kill a job."""
    try:
        runpod.serverless.progress_update(job, payload)
    except Exception as exc:  # noqa: BLE001 - telemetry must not be fatal
        print(f"[handler] progress_update failed (ignored): {exc}")


def _make_step_callback(job: dict[str, Any], total_steps: int):
    """Build a diffusers callback that reports denoising progress."""
    state = {"last_sent": 0.0}

    def callback(_pipe: Any, step: int, _timestep: Any, kwargs: dict[str, Any]):
        completed = step + 1
        now = time.time()
        is_final = completed >= total_steps
        if is_final or now - state["last_sent"] >= PROGRESS_INTERVAL_S:
            state["last_sent"] = now
            _progress(
                job,
                {
                    "phase": "denoising",
                    "step": completed,
                    "total_steps": total_steps,
                    "percent": round(100 * completed / total_steps),
                },
            )
        # diffusers requires the callback to return the (possibly modified)
        # tensor dict.
        return kwargs

    return callback


def _encode_image(image: Any, image_format: str) -> str:
    buffer = io.BytesIO()
    if image_format == "jpeg":
        image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    else:
        image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _base_metadata() -> dict[str, Any]:
    return {
        **MODEL_INFO,
        "worker_uptime_seconds": round(time.time() - WORKER_STARTED_AT, 1),
        "worker_id": os.environ.get("RUNPOD_POD_ID", "unknown"),
    }


def handler(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get("input") or {}

    # --- validate before touching the GPU ---------------------------------
    try:
        request = parse_request(job_input)
    except ValidationError as exc:
        return {"error": str(exc), "error_type": "ValidationError"}

    # --- warmup: the worker is already loaded, so just say so -------------
    if request.warmup:
        if PIPE is None:
            _try_load()
        return {
            "warmup": True,
            "ready": PIPE is not None,
            "message": (
                "Worker is loaded and ready."
                if PIPE is not None
                else f"Worker is up but the model failed to load: {LOAD_ERROR}"
            ),
            **_base_metadata(),
        }

    # Startup load failed (no GPU during a build test, a transient cache miss,
    # a bad token). Retry once here, where we can report a real error.
    if PIPE is None:
        _try_load()
    if PIPE is None:
        return {
            "error": (
                "The model is not loaded on this worker. "
                f"Startup failed with: {LOAD_ERROR}"
            ),
            "error_type": "ModelNotLoaded",
            "worker_id": os.environ.get("RUNPOD_POD_ID", "unknown"),
        }

    seed = request.resolved_seed()
    started = time.time()

    _progress(
        job,
        {
            "phase": "encoding_prompt",
            "step": 0,
            "total_steps": request.num_inference_steps,
            "percent": 0,
        },
    )

    try:
        # A CPU generator seeded explicitly is what the FLUX model card uses,
        # and it keeps results reproducible across GPU types.
        generator = torch.Generator("cpu").manual_seed(seed) if torch else None

        call_kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance,
            "num_images_per_prompt": request.num_images,
            "max_sequence_length": 512,
            "generator": generator,
            "callback_on_step_end": _make_step_callback(
                job, request.num_inference_steps
            ),
        }

        # FLUX.1-dev only honours a negative prompt when true CFG is active.
        # schema.parse_request has already turned it on if one was supplied.
        if request.uses_true_cfg:
            call_kwargs["true_cfg_scale"] = request.true_cfg_scale
            call_kwargs["negative_prompt"] = request.negative_prompt or None

        result = PIPE(**call_kwargs)

        _progress(
            job,
            {
                "phase": "decoding",
                "step": request.num_inference_steps,
                "total_steps": request.num_inference_steps,
                "percent": 100,
            },
        )

        images = [
            {
                "image": _encode_image(img, request.image_format),
                "format": request.image_format,
                "width": request.width,
                "height": request.height,
            }
            for img in result.images
        ]

    except Exception as exc:  # noqa: BLE001 - report, do not crash the worker
        traceback.print_exc()
        # OOM deserves actionable advice rather than a raw CUDA message.
        if type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower():
            return {
                "error": (
                    "The GPU ran out of memory. Try a smaller resolution, fewer "
                    "images, or remove the negative prompt (it doubles memory use)."
                ),
                "error_type": "OutOfMemoryError",
                "detail": str(exc)[:500],
                **_base_metadata(),
            }
        return {
            "error": str(exc)[:1000],
            "error_type": type(exc).__name__,
            **_base_metadata(),
        }

    generation_time = round(time.time() - started, 2)
    print(
        f"[handler] {request.width}x{request.height} "
        f"{request.num_inference_steps} steps seed={seed} "
        f"in {generation_time}s"
    )

    return {
        "images": images,
        "seed": seed,
        "generation_time": generation_time,
        "parameters": request.as_echo(),
        "notes": request.notes,
        **_base_metadata(),
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
