# Measured performance and cost

Every figure below was measured against the live endpoint by
[`scripts/benchmark.py`](../scripts/benchmark.py). Nothing here is an estimate.

- **GPU served:** NVIDIA RTX 6000 Ada Generation
- **Note:** the endpoint lists three GPU tiers in priority order, so the silicon varies between runs. An L40S completed 28 steps in 16.4s where this RTX 6000 Ada took 22.7s - both in the same 48 GB PRO tier, both billed at the same rate, roughly 38% apart in speed.
- **Runpod rate for this tier:** $1.75/hr ($0.000486/s)
- **Weights source:** `runpod_cache`
- **Model load into VRAM:** 7.6s
- **Prompt:** _a red fox curled asleep in deep snow, dawn light through pine branches, photorealistic_ at 1024x1024, guidance 3.5, seed 42

## Warm worker

| Steps | Execution time | Cost | PNG size |
|------:|---------------:|-----:|---------:|
| 4 | 4.3s | $0.0021 | 0.99 MB |
| 12 | 10.0s | $0.0049 | 1.11 MB |
| 28 | 22.7s | $0.0110 | 1.28 MB |
| 50 | 40.6s | $0.0197 | 1.23 MB |

Marginal cost of a denoising step: **0.76s** (**$0.00037**).

## Cold start

| Phase | Time | Billed? |
|---|---:|---|
| Staging cached weights onto the host (one-off, ~58 GB) | ~27.7 min | **No** |
| Worker wake from scaled-to-zero (measured separately) | ~14.5s | Partly |
| Loading 34 GB into VRAM | 7.6s | Yes |
| Delay time on this run's first request | 1.0s | Partly |

The last row is small because a worker was already warm when the benchmark
started; it is not a true cold start. The 14.5s figure comes from a request
made after the endpoint had fully scaled to zero.

Runpod does not bill while a worker is `Initializing`, which is the whole
argument for the cached-model design: the 58 GB download cost nothing, and
loading from the host's local disk into VRAM takes seconds rather than the
minutes a fresh Hugging Face download would.

## Payload headroom

A 1024x1024 PNG returns as ~1.23 MB of base64. Runpod caps
`/run` at 10 MB and `/runsync` at 20 MB, so a single image has ample room;
batches are capped at 2 in `worker/schema.py` to stay inside the limit.

## What this project actually cost

This benchmark run: **$0.0397** across 5 generations.

The dominant cost control is **Active workers = 0**: one always-on 48 GB
worker would cost $42.00/day, which would consume a $30 budget in
17 hours. Scale-to-zero plus a 5 second idle timeout
means the endpoint costs nothing between requests.
