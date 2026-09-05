"""S10: exercise the handler on CPU with a stub pipeline. No GPU, no cost.

Covers the code paths that are expensive to debug once deployed: input
validation, defaulting, dimension snapping, the negative-prompt/true-CFG
interaction, seed reproducibility, progress reporting and base64 encoding.

Usage:  .venv/Scripts/python scripts/local_handler_test.py
"""

from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "worker"))

# Must be set before importing handler: it decides whether the module-level
# load pulls 34 GB of weights or instantiates a stub.
os.environ["FLUX_STUB"] = "1"

import handler as handler_mod  # noqa: E402
from schema import ValidationError, parse_request  # noqa: E402

PASSED = 0
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if condition:
        PASSED += 1
        print(f"  [pass] {name}")
    else:
        FAILED.append(f"{name}{' - ' + detail if detail else ''}")
        print(f"  [FAIL] {name} {detail}")


def expect_rejected(name: str, payload: dict) -> None:
    try:
        parse_request(payload)
    except ValidationError as exc:
        check(name, True)
        print(f"         -> {exc}")
    else:
        check(name, False, "should have been rejected but was accepted")


def test_validation() -> None:
    print("\nValidation - bad input must be rejected cheaply")
    expect_rejected("missing prompt", {})
    expect_rejected("empty prompt", {"prompt": "   "})
    expect_rejected("prompt too long", {"prompt": "x" * 2001})
    expect_rejected("steps above max", {"prompt": "a", "num_inference_steps": 51})
    expect_rejected("steps below min", {"prompt": "a", "num_inference_steps": 0})
    expect_rejected("guidance out of range", {"prompt": "a", "guidance": 99})
    expect_rejected("width too large", {"prompt": "a", "width": 4096})
    expect_rejected("too many images", {"prompt": "a", "num_images": 9})
    expect_rejected("bad image format", {"prompt": "a", "image_format": "webp"})
    expect_rejected("non-numeric steps", {"prompt": "a", "num_inference_steps": "many"})
    expect_rejected("input not an object", "just a string")


def test_defaults_and_normalisation() -> None:
    print("\nDefaults and normalisation")
    req = parse_request({"prompt": "  a fox  "})
    check("prompt is trimmed", req.prompt == "a fox")
    check("default size is 1024x1024", (req.width, req.height) == (1024, 1024))
    check("default steps is 28", req.num_inference_steps == 28)
    check("default guidance is 3.5", req.guidance == 3.5)
    check("default format is png", req.image_format == "png")

    req = parse_request({"prompt": "a", "image_format": "JPG"})
    check("jpg is normalised to jpeg", req.image_format == "jpeg")

    req = parse_request({"prompt": "a", "guidance_scale": 7.0})
    check("guidance_scale alias accepted", req.guidance == 7.0)

    req = parse_request({"prompt": "a", "width": 1000, "height": 1000})
    check("odd dimensions snap to a multiple of 64", (req.width, req.height) == (1024, 1024))
    check("snapping is reported in notes", any("multiple of 64" in n for n in req.notes))


def test_true_cfg_interaction() -> None:
    print("\nFLUX true-CFG behaviour")
    req = parse_request({"prompt": "a"})
    check("true CFG off by default", not req.uses_true_cfg)

    req = parse_request({"prompt": "a", "negative_prompt": "blurry"})
    check("negative prompt enables true CFG", req.uses_true_cfg)
    check("cost warning surfaced", any("doubles" in n for n in req.notes))


def test_seed_reproducibility() -> None:
    print("\nSeeds")
    req = parse_request({"prompt": "a", "seed": 42})
    check("explicit seed is preserved", req.resolved_seed() == 42)

    req = parse_request({"prompt": "a", "seed": -1})
    seeds = {parse_request({"prompt": "a", "seed": -1}).resolved_seed() for _ in range(20)}
    check("seed -1 produces random seeds", len(seeds) > 1)
    check("random seed is in uint32 range", all(0 <= s < 2**32 for s in seeds))


def test_handler_end_to_end() -> None:
    print("\nHandler end to end (stub pipeline)")
    progress_events: list[dict] = []
    handler_mod._progress = lambda _job, payload: progress_events.append(payload)

    job = {"id": "local_test", "input": {"prompt": "a red fox", "num_inference_steps": 8}}
    result = handler_mod.handler(job)

    check("no error returned", "error" not in result, str(result.get("error", "")))
    check("one image returned", len(result.get("images", [])) == 1)
    check("seed reported", isinstance(result.get("seed"), int))
    check("generation_time reported", isinstance(result.get("generation_time"), float))
    check("weights_source is stub", result.get("weights_source") == "stub")
    check("parameters echoed back", result.get("parameters", {}).get("num_inference_steps") == 8)

    b64 = result["images"][0]["image"]
    raw = base64.b64decode(b64)
    check("output decodes as a real PNG", raw[:8] == b"\x89PNG\r\n\x1a\n")

    from PIL import Image

    img = Image.open(io.BytesIO(raw))
    check("image is 1024x1024", img.size == (1024, 1024), str(img.size))

    phases = [e["phase"] for e in progress_events]
    check("prompt-encoding phase reported", "encoding_prompt" in phases)
    check("denoising phase reported", "denoising" in phases)
    check("decoding phase reported", "decoding" in phases)
    check("final progress reaches 100%", any(e.get("percent") == 100 for e in progress_events))
    check(
        "progress is throttled, not one event per step",
        len([p for p in phases if p == "denoising"]) <= 8,
        f"{len([p for p in phases if p == 'denoising'])} denoising events",
    )


def test_handler_error_paths() -> None:
    print("\nHandler error paths")
    result = handler_mod.handler({"id": "t", "input": {}})
    check("missing prompt returns structured error", result.get("error_type") == "ValidationError")
    check("error message is human readable", "prompt" in result.get("error", "").lower())

    result = handler_mod.handler({"id": "t", "input": {"warmup": True}})
    check("warmup succeeds without a prompt", result.get("warmup") is True)
    check("warmup reports worker metadata", "worker_uptime_seconds" in result)

    result = handler_mod.handler({"id": "t", "input": {"prompt": "a", "image_format": "jpeg"}})
    raw = base64.b64decode(result["images"][0]["image"])
    check("jpeg output has a JPEG header", raw[:3] == b"\xff\xd8\xff")


def main() -> int:
    test_validation()
    test_defaults_and_normalisation()
    test_true_cfg_interaction()
    test_seed_reproducibility()
    test_handler_end_to_end()
    test_handler_error_paths()

    print(f"\n{PASSED} passed, {len(FAILED)} failed")
    for f in FAILED:
        print(f"  - {f}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
