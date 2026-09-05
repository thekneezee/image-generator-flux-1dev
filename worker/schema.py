"""Input contract for the FLUX.1-dev serverless worker.

The field names deliberately mirror Runpod's own Public Endpoint schema for
Flux Dev (prompt / negative_prompt / width / height / num_inference_steps /
guidance / seed / image_format), so this worker is a drop-in replacement for
their hosted endpoint. `num_images`, `true_cfg_scale` and `warmup` are our
additions.

Validation happens before the GPU is touched: a bad request should cost
milliseconds, not a diffusion run.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

# Payload ceilings. /run allows 10 MB and base64 inflates by ~33%, so a
# 1024x1024 PNG (~1.5-2.5 MB encoded) is comfortable but a large batch is not.
MAX_IMAGES = 2
MAX_PIXELS = 1536 * 1536

SIZE_MIN, SIZE_MAX, SIZE_MULTIPLE = 256, 1536, 64
STEPS_MIN, STEPS_MAX = 1, 50
GUIDANCE_MIN, GUIDANCE_MAX = 0.0, 10.0
PROMPT_MAX_CHARS = 2000

# FLUX.1-dev is guidance-distilled: `guidance` is an embedded conditioning
# signal, not classifier-free guidance, so a negative prompt does nothing
# unless true CFG is switched on. Turning it on runs the transformer twice
# per step, roughly doubling cost. We enable it automatically but surface
# the fact in the response so the UI can warn the user.
DEFAULT_TRUE_CFG_WITH_NEGATIVE = 4.0

IMAGE_FORMATS = ("png", "jpeg")


class ValidationError(ValueError):
    """Raised for a malformed request. Reported to the caller, never a crash."""


@dataclass
class GenerationRequest:
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 28
    guidance: float = 3.5
    true_cfg_scale: float = 1.0
    seed: int = -1
    image_format: str = "png"
    num_images: int = 1
    warmup: bool = False
    # Filled during validation, echoed back to the caller.
    notes: list[str] = field(default_factory=list)

    @property
    def uses_true_cfg(self) -> bool:
        return self.true_cfg_scale > 1.0

    def resolved_seed(self) -> int:
        """A concrete seed, so every response is reproducible."""
        if self.seed is None or self.seed < 0:
            return random.randint(0, 2**32 - 1)
        return int(self.seed) % (2**32)

    def as_echo(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "width": self.width,
            "height": self.height,
            "num_inference_steps": self.num_inference_steps,
            "guidance": self.guidance,
            "true_cfg_scale": self.true_cfg_scale,
            "image_format": self.image_format,
            "num_images": self.num_images,
        }


def _as_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValidationError(f"'{name}' must be an integer, got {value!r}") from None


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"'{name}' must be a number, got {value!r}") from None


def _check_range(value: float, lo: float, hi: float, name: str) -> None:
    if not lo <= value <= hi:
        raise ValidationError(f"'{name}' must be between {lo} and {hi}, got {value}")


def _snap_dimension(value: int, name: str, notes: list[str]) -> int:
    _check_range(value, SIZE_MIN, SIZE_MAX, name)
    if value % SIZE_MULTIPLE:
        snapped = max(
            SIZE_MIN, min(SIZE_MAX, round(value / SIZE_MULTIPLE) * SIZE_MULTIPLE)
        )
        notes.append(
            f"{name} {value} is not a multiple of {SIZE_MULTIPLE}; using {snapped}"
        )
        return snapped
    return value


def parse_request(job_input: Any) -> GenerationRequest:
    """Validate and normalise `job["input"]`.

    Raises ValidationError with a human-readable message. The handler turns
    that into a structured error response rather than a failed job, so the UI
    can tell the user what to fix.
    """
    if not isinstance(job_input, dict):
        raise ValidationError("'input' must be a JSON object")

    notes: list[str] = []

    # Warmup jobs exist only to pay the cold start ahead of a real request,
    # so they skip prompt validation entirely.
    warmup = bool(job_input.get("warmup", False))

    prompt = job_input.get("prompt")
    if warmup:
        prompt = prompt if isinstance(prompt, str) and prompt.strip() else "warmup"
    else:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValidationError("'prompt' is required and must be a non-empty string")
        if len(prompt) > PROMPT_MAX_CHARS:
            raise ValidationError(
                f"'prompt' is {len(prompt)} characters; maximum is {PROMPT_MAX_CHARS}"
            )
        prompt = prompt.strip()

    negative_prompt = job_input.get("negative_prompt") or ""
    if not isinstance(negative_prompt, str):
        raise ValidationError("'negative_prompt' must be a string")
    negative_prompt = negative_prompt.strip()

    width = _snap_dimension(_as_int(job_input.get("width", 1024), "width"), "width", notes)
    height = _snap_dimension(
        _as_int(job_input.get("height", 1024), "height"), "height", notes
    )
    if width * height > MAX_PIXELS:
        raise ValidationError(
            f"{width}x{height} exceeds the {MAX_PIXELS:,}-pixel limit"
        )

    steps = _as_int(job_input.get("num_inference_steps", 28), "num_inference_steps")
    _check_range(steps, STEPS_MIN, STEPS_MAX, "num_inference_steps")

    # Accept diffusers' own parameter name as an alias for convenience.
    raw_guidance = job_input.get("guidance", job_input.get("guidance_scale", 3.5))
    guidance = _as_float(raw_guidance, "guidance")
    _check_range(guidance, GUIDANCE_MIN, GUIDANCE_MAX, "guidance")

    true_cfg_scale = _as_float(job_input.get("true_cfg_scale", 1.0), "true_cfg_scale")
    _check_range(true_cfg_scale, 1.0, 10.0, "true_cfg_scale")
    if negative_prompt and true_cfg_scale <= 1.0:
        true_cfg_scale = DEFAULT_TRUE_CFG_WITH_NEGATIVE
        notes.append(
            "negative_prompt requires true CFG; true_cfg_scale set to "
            f"{DEFAULT_TRUE_CFG_WITH_NEGATIVE}. This roughly doubles generation time."
        )

    seed = _as_int(job_input.get("seed", -1), "seed")

    image_format = str(job_input.get("image_format", "png")).lower()
    if image_format == "jpg":
        image_format = "jpeg"
    if image_format not in IMAGE_FORMATS:
        raise ValidationError(
            f"'image_format' must be one of {IMAGE_FORMATS}, got {image_format!r}"
        )

    num_images = _as_int(job_input.get("num_images", 1), "num_images")
    _check_range(num_images, 1, MAX_IMAGES, "num_images")

    return GenerationRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance=guidance,
        true_cfg_scale=true_cfg_scale,
        seed=seed,
        image_format=image_format,
        num_images=num_images,
        warmup=warmup,
        notes=notes,
    )
