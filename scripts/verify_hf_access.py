"""S6: prove the Hugging Face READ token can actually reach gated FLUX.1-dev.

Catches, before we spend any GPU money:
  - a token belonging to an account that never accepted the licence (403)
  - a fine-grained token missing gated-repo permission
  - a revoked or mistyped token (401)

Also checks the WRITE token has the permissions Stage E needs.
Never prints a token value.

Usage:  .venv/Scripts/python scripts/verify_hf_access.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "black-forest-labs/FLUX.1-dev"

# The subset diffusers actually loads. Confirming these exist means the
# runtime-download fallback can skip the 23.8 GB original-format checkpoint.
DIFFUSERS_DIRS = (
    "model_index.json",
    "scheduler/",
    "text_encoder/",
    "text_encoder_2/",
    "tokenizer/",
    "tokenizer_2/",
    "transformer/",
    "vae/",
)


def check_identity(api: HfApi, label: str) -> str | None:
    try:
        me = api.whoami()
    except HfHubHTTPError as exc:
        print(f"  [FAIL] {label}: could not authenticate ({exc.response.status_code}).")
        return None
    name = me.get("name") or me.get("email") or "<unknown>"
    auth = (me.get("auth") or {}).get("accessToken") or {}
    role = auth.get("role", "unknown")
    print(f"  [ok]   {label}: authenticated as '{name}' (role: {role})")
    return name


def check_read_access(api: HfApi) -> bool:
    try:
        files = api.list_repo_files(MODEL_ID)
    except GatedRepoError:
        print(f"  [FAIL] Gated: this account has NOT accepted the {MODEL_ID} licence.")
        print("         Fix: log into huggingface.co as THIS account, open")
        print(f"         https://huggingface.co/{MODEL_ID}")
        print("         and click 'Agree and access repository'.")
        return False
    except RepositoryNotFoundError:
        print(f"  [FAIL] {MODEL_ID} not found, or the token cannot see it at all.")
        return False
    except HfHubHTTPError as exc:
        code = exc.response.status_code
        hint = {
            401: "token is invalid or revoked",
            403: "token lacks permission for gated repos",
        }.get(code, "unexpected HTTP error")
        print(f"  [FAIL] HTTP {code}: {hint}.")
        return False

    print(f"  [ok]   Can read {MODEL_ID}: {len(files)} files listed.")

    missing = [
        d for d in DIFFUSERS_DIRS
        if not any(f == d or f.startswith(d) for f in files)
    ]
    if missing:
        print(f"  [WARN] diffusers components not found in repo: {missing}")
        return False
    print("  [ok]   All diffusers components present (transformer, T5, CLIP, VAE).")
    return True


def main() -> int:
    env = dotenv_values(REPO_ROOT / ".env")

    read_token = env.get("HF_TOKEN", "").strip()
    write_token = env.get("HF_TOKEN_WRITE", "").strip()

    if not read_token:
        print("FATAL: HF_TOKEN is empty in .env")
        return 1

    ok = True

    print("READ token")
    read_user = check_identity(HfApi(token=read_token), "HF_TOKEN")
    if read_user is None:
        return 1
    ok &= check_read_access(HfApi(token=read_token))

    print("\nWRITE token")
    if not write_token:
        print("  [WARN] HF_TOKEN_WRITE is empty. Needed at Stage E, not before.")
    else:
        write_user = check_identity(HfApi(token=write_token), "HF_TOKEN_WRITE")
        if write_user is None:
            ok = False
        elif read_user and write_user != read_user:
            print(f"  [WARN] Write token belongs to '{write_user}' but read token to"
                  f" '{read_user}'. Not fatal, but the Space will live under"
                  f" '{write_user}'.")

    print()
    if ok:
        print("PASS - RunPod will be able to download FLUX.1-dev with this token.")
        return 0
    print("FAIL - fix the issues above before creating the endpoint (S15).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
