"""Validate .env without ever printing secret values.

Prints one line per variable: presence, expected prefix, and length.
Never echoes the value itself, so it is safe to run with the output shared.

Usage:  python scripts/check_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"

# name -> (required_now, expected_prefix or None, human description)
EXPECTED = {
    "RUNPOD_API_KEY": (True, "rpa_", "RunPod API key (S3)"),
    "HF_TOKEN": (True, "hf_", "Hugging Face READ token"),
    "HF_TOKEN_WRITE": (True, "hf_", "Hugging Face WRITE token (S5)"),
    "GITHUB_REPO_URL": (True, "https://github.com/", "GitHub repo URL (S7)"),
    "APP_PASSWORD": (True, None, "Demo password"),
    "RUNPOD_ENDPOINT_ID": (False, None, "Endpoint ID (filled at S18)"),
    "RUNPOD_MOCK": (False, None, "Mock mode flag"),
}


def load_env(path: Path) -> dict[str, str]:
    if not path.exists():
        print(f"FATAL: {path} does not exist.")
        sys.exit(1)
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def main() -> int:
    env = load_env(ENV_PATH)
    problems: list[str] = []

    print(f"Checking {ENV_PATH}\n")
    for name, (required, prefix, desc) in EXPECTED.items():
        value = env.get(name, "")

        if not value:
            if required:
                print(f"  [MISSING]  {name:22s}  <- {desc}")
                problems.append(f"{name} is empty")
            else:
                print(f"  [blank ok] {name:22s}  <- {desc}")
            continue

        # Catch the classic paste mistakes without revealing the value.
        if value != value.strip():
            problems.append(f"{name} has surrounding whitespace")
        if value.startswith(("'", '"')) or value.endswith(("'", '"')):
            problems.append(f"{name} is wrapped in quotes - remove them")
        if prefix and not value.startswith(prefix):
            problems.append(f"{name} should start with '{prefix}'")
            print(f"  [BAD FMT]  {name:22s}  len={len(value)}  expected prefix '{prefix}'")
            continue

        shown = f"{prefix}..." if prefix else "set"
        print(f"  [ok]       {name:22s}  {shown}  len={len(value)}")

    # Cross-check: the two HF tokens must not be identical.
    if env.get("HF_TOKEN") and env.get("HF_TOKEN") == env.get("HF_TOKEN_WRITE"):
        problems.append("HF_TOKEN and HF_TOKEN_WRITE are the same value")

    print()
    if problems:
        print("PROBLEMS FOUND:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("All required variables look correctly formatted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
