"""Locate the FLUX.1-dev weights and load them onto the GPU.

Weight resolution is tiered, cheapest first:

  1. Runpod cached model  - /runpod-volume/huggingface-cache/hub/...
     Runpod pre-stages the repo on the host machine. Not billed while it
     downloads, shared between workers, and by far the fastest cold start.
     This is the path we configure the endpoint to use.

  2. A pre-populated local HF cache (HF_HOME on a network volume, or weights
     baked into the image).

  3. Runtime download from Hugging Face with HF_TOKEN. Billed, so it is a
     fallback rather than a plan, but it means the worker still functions if
     model caching is unavailable. `allow_patterns` restricts it to the
     diffusers-format subset (~34 GB) and skips the repo's top-level
     flux1-dev.safetensors (23.8 GB in Black Forest Labs' original format,
     which diffusers never reads).

The resolved tier is reported back in every response as `weights_source`, so
a glance at the API output tells you whether caching is actually working.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

MODEL_ID = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.1-dev")
RUNPOD_CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")

# Everything FluxPipeline.from_pretrained actually opens.
DIFFUSERS_ALLOW_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "text_encoder/*",
    "text_encoder_2/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "transformer/*",
    "vae/*",
]

# Below this, the 34 GB of weights will not fit alongside activations and we
# fall back to CPU offloading instead of a hard failure.
#
# Set to 40, not 44: a nominally 48 GB L40S reports 44.4 GB usable, so a 44.0
# threshold cleared by only 0.4 GB. Another 48 GB card reporting slightly less
# would have silently dropped to CPU offload and run several times slower for
# no reason. 40 keeps every 48 GB card on the fast path while still catching
# 24 GB and 32 GB cards, which genuinely cannot hold the model.
FULL_GPU_VRAM_THRESHOLD_GB = 40.0

STUB = os.environ.get("FLUX_STUB") == "1"


class _StubImage:
    """Stands in for a PIL image in stub mode."""

    def __init__(self, width: int, height: int) -> None:
        self.width, self.height = width, height

    def save(self, fp: Any, format: str = "PNG", **_: Any) -> None:  # noqa: A002
        from PIL import Image

        Image.new("RGB", (self.width, self.height), (35, 35, 48)).save(fp, format=format)


class _StubResult:
    def __init__(self, images: list[Any]) -> None:
        self.images = images


class StubPipeline:
    """A fake FluxPipeline for CPU-only local testing.

    Exercises the real call signature and fires the step callback so the
    progress-reporting path is covered without a GPU or 34 GB of weights.
    """

    def __call__(
        self,
        prompt: str,
        num_inference_steps: int = 28,
        width: int = 1024,
        height: int = 1024,
        callback_on_step_end: Any = None,
        num_images_per_prompt: int = 1,
        **_: Any,
    ) -> _StubResult:
        for step in range(num_inference_steps):
            if callback_on_step_end is not None:
                callback_on_step_end(self, step, 0, {"latents": None})
            time.sleep(0.005)
        return _StubResult([_StubImage(width, height) for _ in range(num_images_per_prompt)])


def resolve_cached_snapshot(model_id: str) -> Path | None:
    """Find the Runpod-cached snapshot directory for `model_id`, if present.

    Runpod mirrors Hugging Face's on-disk layout: forward slashes in the repo
    id become double dashes, and the real files live under a commit-hash
    directory inside `snapshots/`.
    """
    if "/" not in model_id:
        return None

    org, name = model_id.split("/", 1)
    model_root = RUNPOD_CACHE_ROOT / f"models--{org}--{name}"
    snapshots = model_root / "snapshots"

    # refs/main names the commit the main branch points at. This is what
    # from_pretrained would resolve to with network access, so prefer it.
    refs_main = model_root / "refs" / "main"
    if refs_main.is_file():
        candidate = snapshots / refs_main.read_text(encoding="utf-8").strip()
        if candidate.is_dir():
            return candidate

    # Older cache layouts have no refs/main; take the only snapshot present.
    if snapshots.is_dir():
        versions = sorted(d for d in snapshots.iterdir() if d.is_dir())
        if versions:
            return versions[0]

    return None


def _looks_complete(path: Path) -> bool:
    """Guard against a half-written cache directory."""
    return (path / "model_index.json").is_file() and (path / "transformer").is_dir()


def resolve_model_path() -> tuple[str, str]:
    """Return (path_or_repo_id, weights_source)."""
    cached = resolve_cached_snapshot(MODEL_ID)
    if cached and _looks_complete(cached):
        return str(cached), "runpod_cache"
    if cached:
        print(f"[pipeline] cache dir {cached} exists but looks incomplete; ignoring")

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        local = Path(hf_home) / "hub"
        org, name = MODEL_ID.split("/", 1)
        candidate = local / f"models--{org}--{name}"
        if (candidate / "snapshots").is_dir():
            return MODEL_ID, "local_hf_cache"

    return MODEL_ID, "runtime_download"


def _download(repo_id: str) -> str:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "FLUX.1-dev is a gated repository and no cached copy was found. "
            "Set the HF_TOKEN environment variable on the endpoint, or "
            "configure the Model field so Runpod caches the weights."
        )
    print(f"[pipeline] downloading {repo_id} (~34 GB, diffusers subset only)")
    started = time.time()
    path = snapshot_download(
        repo_id,
        token=token,
        allow_patterns=DIFFUSERS_ALLOW_PATTERNS,
        max_workers=8,
    )
    print(f"[pipeline] download finished in {time.time() - started:.0f}s -> {path}")
    return path


def load_pipeline() -> tuple[Any, dict[str, Any]]:
    """Load FluxPipeline. Returns (pipeline, info)."""
    started = time.time()

    if STUB:
        print("[pipeline] FLUX_STUB=1 - using StubPipeline, no weights loaded")
        return StubPipeline(), {
            "weights_source": "stub",
            "model_load_seconds": 0.0,
            "gpu": "stub",
            "vram_gb": 0.0,
            "offload": False,
            "model": MODEL_ID,
        }

    import torch

    # Fail fast and legibly. Without this, a CPU-only machine would silently
    # start downloading 34 GB before discovering it has no GPU - which is
    # exactly what happens when you smoke-test the image locally.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device available. This worker requires a GPU with at least "
            "24 GB of VRAM (48 GB recommended). For a local smoke test without a "
            "GPU, set FLUX_STUB=1."
        )

    from diffusers import FluxPipeline

    model_path, weights_source = resolve_model_path()
    print(f"[pipeline] weights_source={weights_source} path={model_path}")

    if weights_source == "runtime_download":
        model_path = _download(model_path)

    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=weights_source == "runpod_cache",
    )

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3

    # 34 GB of weights plus activations fits comfortably on a 48 GB card, so
    # keep everything resident. On anything smaller, offload layers to system
    # RAM: several times slower, but it runs rather than OOMing.
    offload = vram_gb < FULL_GPU_VRAM_THRESHOLD_GB
    if offload:
        print(f"[pipeline] {vram_gb:.0f} GB VRAM - enabling model CPU offload")
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    else:
        print(f"[pipeline] {vram_gb:.0f} GB VRAM - loading fully onto the GPU")
        pipe.to("cuda")

    elapsed = time.time() - started
    print(f"[pipeline] ready in {elapsed:.1f}s on {props.name}")

    return pipe, {
        "weights_source": weights_source,
        "model_load_seconds": round(elapsed, 1),
        "gpu": props.name,
        "vram_gb": round(vram_gb, 1),
        "offload": offload,
        "model": MODEL_ID,
    }
