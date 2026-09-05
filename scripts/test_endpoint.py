"""Submit a prompt to the live endpoint, watch it work, save the PNG.

Proves the endpoint end-to-end from outside the Runpod console: submit via
/run, poll /status, decode the base64 image, and report what it actually cost.

Usage:
    python scripts/test_endpoint.py "a red fox in deep snow"
    python scripts/test_endpoint.py "..." --steps 12 --warmup
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "app"))

from dotenv import dotenv_values  # noqa: E402

from runpod_client import RunPodClient, RunPodError, cost_for  # noqa: E402

OUT_DIR = REPO_ROOT / "outputs"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?",
                        default="a red fox curled asleep in deep snow, dawn light "
                                "through pine branches, photorealistic")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", action="store_true",
                        help="Send a warmup job instead of generating.")
    args = parser.parse_args()

    env = dotenv_values(REPO_ROOT / ".env")
    client = RunPodClient(
        api_key=env.get("RUNPOD_API_KEY", ""),
        endpoint_id=env.get("RUNPOD_ENDPOINT_ID", ""),
    )

    health = client.health()
    print(f"endpoint : {client.endpoint_id}")
    print(f"workers  : {health['workers']}")

    payload = (
        {"warmup": True, "prompt": "warmup"}
        if args.warmup
        else {
            "prompt": args.prompt,
            "width": args.width,
            "height": args.height,
            "num_inference_steps": args.steps,
            "guidance": args.guidance,
            "seed": args.seed,
            "image_format": "png",
        }
    )

    if not args.warmup:
        print(f"prompt   : {args.prompt}")
        print(f"settings : {args.width}x{args.height}, {args.steps} steps, "
              f"guidance {args.guidance}, seed {args.seed}")
    print()

    started = time.time()
    job_id = client.submit(payload)
    print(f"submitted: {job_id}")

    last_line = ""
    final = None
    try:
        for update in client.poll(job_id, timeout=900):
            final = update
            elapsed = time.time() - started
            progress = update.progress or {}
            phase = progress.get("phase", "")
            if phase == "denoising":
                line = (f"  {elapsed:6.1f}s  denoising "
                        f"{progress.get('step')}/{progress.get('total_steps')} "
                        f"({progress.get('percent')}%)")
            elif phase:
                line = f"  {elapsed:6.1f}s  {phase}"
            else:
                line = f"  {elapsed:6.1f}s  {update.status}"
            if line != last_line:
                print(line, flush=True)
                last_line = line
    except RunPodError as exc:
        print(f"\nFAILED: {exc}")
        return 1

    if final is None or final.status != "COMPLETED":
        print(f"\nJob ended as {final.status if final else 'UNKNOWN'}")
        if final and final.output:
            print(json.dumps(final.output, indent=2)[:2000])
        return 1

    output = final.output or {}
    if "error" in output:
        print(f"\nWorker returned an error: {output['error_type']}: {output['error']}")
        return 1

    exec_s = (final.execution_ms or 0) / 1000
    delay_s = (final.delay_ms or 0) / 1000
    gpu = output.get("gpu")
    cost = cost_for(gpu, exec_s)

    print()
    print(f"status        : {final.status}")
    print(f"gpu           : {gpu} ({output.get('vram_gb')} GB, offload={output.get('offload')})")
    print(f"weights_source: {output.get('weights_source')}")
    print(f"delay time    : {delay_s:.1f}s  (queue + cold start, partly unbilled)")
    print(f"execution time: {exec_s:.1f}s  (billed)")
    print(f"model load    : {output.get('model_load_seconds')}s")
    print(f"worker uptime : {output.get('worker_uptime_seconds')}s")
    print(f"estimated cost: ${cost:.4f}")

    if args.warmup:
        print(f"warmup ready  : {output.get('ready')}")
        return 0

    print(f"seed          : {output.get('seed')}")
    print(f"generation    : {output.get('generation_time')}s")
    for note in output.get("notes") or []:
        print(f"note          : {note}")

    OUT_DIR.mkdir(exist_ok=True)
    for index, image in enumerate(output.get("images", [])):
        suffix = "jpg" if image.get("format") == "jpeg" else "png"
        path = OUT_DIR / f"{output.get('seed')}-{index}.{suffix}"
        path.write_bytes(base64.b64decode(image["image"]))
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"saved         : {path}  ({size_mb:.2f} MB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
