# Runpod's GitHub builder uses the repository root as the build context and
# looks for the Dockerfile here by default, so this file stays at the root.
#
# Base image choice: the CUDA *runtime* variant, not *devel*. We need the CUDA
# runtime libraries, not the compiler toolchain, and devel would add ~15 GB of
# dead weight that Runpod would have to pull onto every fresh host.
# 3.93 GB compressed, ships Python 3.11 + torch 2.4.0 + CUDA 12.4.
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Where the runtime-download fallback puts weights if Runpod's model
    # cache is unavailable. Overridden by the endpoint if a network volume
    # is attached.
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Dependencies first, in their own layer: application edits then rebuild in
# seconds instead of reinstalling every package.
COPY worker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Fail the BUILD, not the worker, on an import-time incompatibility.
#
# pip resolves versions from declared metadata, which does not capture
# everything: diffusers 0.39 installs cleanly against torch 2.4 and then
# raises RuntimeError on `from diffusers import FluxPipeline`, because it
# registers a custom op with a type annotation torch 2.4 cannot parse.
#
# Without this line that failure surfaces as a worker that crash-loops on
# Runpod - which is retried for up to 7 days and is far more expensive to
# diagnose than a red build. Importing FluxPipeline needs no GPU.
RUN python -c "\
import torch, diffusers, transformers, accelerate, huggingface_hub; \
from diffusers import FluxPipeline; \
print('import smoke test OK'); \
print('  torch       ', torch.__version__); \
print('  diffusers   ', diffusers.__version__); \
print('  transformers', transformers.__version__); \
print('  hub         ', huggingface_hub.__version__)"

COPY worker/ /app/

# -u disables output buffering so print() appears live in Runpod's Logs tab
# instead of arriving in a batch when the worker exits.
CMD ["python", "-u", "/app/handler.py"]
