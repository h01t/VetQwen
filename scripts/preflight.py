"""
preflight.py — lightweight environment checks for VetQwen.

Usage:
    python scripts/preflight.py --profile remote
    python scripts/preflight.py --profile demo --skip-ollama
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import sys

from judge_utils import DEFAULT_OLLAMA_BASE_URL, check_ollama

RECOMMENDED_PYTHON = (3, 11)
REMOTE_SUPPORTED_PYTHON_MIN = (3, 10)
REMOTE_SUPPORTED_PYTHON_MAX = (3, 12)
DEMO_SUPPORTED_PYTHON_MIN = (3, 10)

REMOTE_PACKAGE_CHECKS = [
    ("yaml", "PyYAML", True),
    ("requests", "requests", True),
    ("numpy", "numpy", True),
    ("datasets", "datasets", True),
    ("transformers", "transformers", True),
    ("peft", "peft", True),
    ("trl", "trl", True),
    ("accelerate", "accelerate", True),
    ("bitsandbytes", "bitsandbytes", True),
    ("sentence_transformers", "sentence-transformers", False),
    ("rouge_score", "rouge-score", False),
    ("bert_score", "bert-score", False),
    ("tensorboard", "tensorboard", False),
]

DEMO_PACKAGE_CHECKS = [
    ("requests", "requests", True),
    ("numpy", "numpy", True),
    ("transformers", "transformers", True),
    ("peft", "peft", True),
    ("gradio", "gradio", True),
    ("accelerate", "accelerate", False),
]


def status_line(state: str, message: str) -> str:
    return f"[{state}] {message}"


def python_status(profile: str) -> tuple[str, str]:
    version = sys.version_info[:3]
    version_str = ".".join(str(part) for part in version)

    if version[:2] == RECOMMENDED_PYTHON:
        return "PASS", f"Python {version_str} matches the recommended 3.11 runtime."

    if profile == "remote":
        if REMOTE_SUPPORTED_PYTHON_MIN <= version[:2] <= REMOTE_SUPPORTED_PYTHON_MAX:
            return (
                "WARN",
                f"Python {version_str} is usable, but 3.11 is the recommended runtime for the pinned GPU stack.",
            )
        return (
            "FAIL",
            f"Python {version_str} is outside the supported 3.10-3.12 range for the pinned torch 2.4.1/cu121 setup.",
        )

    if version[:2] >= DEMO_SUPPORTED_PYTHON_MIN:
        return (
            "WARN",
            f"Python {version_str} is acceptable for the local demo path, but 3.11 remains the recommended runtime.",
        )
    return (
        "FAIL",
        f"Python {version_str} is too old for the supported demo/runtime path.",
    )


def package_status(module_name: str, package_name: str, required: bool) -> tuple[str, str]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        state = "FAIL" if required else "WARN"
        return state, f"{package_name} is not installed."

    try:
        version = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        version = "installed"
    return "PASS", f"{package_name} {version}"


def torch_status(profile: str) -> list[tuple[str, str]]:
    statuses: list[tuple[str, str]] = []
    try:
        import torch
    except ImportError:
        return [("FAIL", "torch is not installed.")]

    statuses.append(("PASS", f"torch {getattr(torch, '__version__', 'unknown')}"))

    try:
        cuda_available = torch.cuda.is_available()
    except Exception as exc:  # pragma: no cover - defensive
        statuses.append(("WARN", f"CUDA check failed: {exc}"))
        return statuses

    if cuda_available:
        statuses.append(("PASS", f"CUDA is available ({torch.cuda.device_count()} device(s))."))
        return statuses

    if profile == "remote":
        statuses.append(
            (
                "FAIL",
                "CUDA is not available. Remote training/evaluation is expected to run on the NVIDIA workstation.",
            )
        )
    else:
        statuses.append(
            (
                "WARN",
                "CUDA is not available. That is fine for the local demo path; inference will fall back to CPU.",
            )
        )
    return statuses


def ollama_status(base_url: str, skip: bool, profile: str) -> tuple[str, str]:
    if skip:
        return "WARN", "Skipped Ollama reachability check."

    ok, detail = check_ollama(base_url)
    if ok:
        models = detail or "no models reported"
        return "PASS", f"Ollama reachable at {base_url}. Models: {models}"

    if profile == "remote":
        return "FAIL", f"Ollama not reachable at {base_url}: {detail}"
    return "WARN", f"Ollama not reachable at {base_url}: {detail}"


def package_checks_for_profile(profile: str) -> list[tuple[str, str, bool]]:
    if profile == "demo":
        return DEMO_PACKAGE_CHECKS
    return REMOTE_PACKAGE_CHECKS


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VetQwen environment preflight checks")
    parser.add_argument(
        "--profile",
        choices=["remote", "demo"],
        default="remote",
        help="Check the remote research stack or the local demo stack (default: remote)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=DEFAULT_OLLAMA_BASE_URL,
        help=f"Ollama API URL (default: {DEFAULT_OLLAMA_BASE_URL})",
    )
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Skip the Ollama reachability check",
    )
    args = parser.parse_args()

    statuses: list[tuple[str, str]] = [python_status(args.profile)]
    statuses.extend(package_status(*package_check) for package_check in package_checks_for_profile(args.profile))
    statuses.extend(torch_status(args.profile))
    statuses.append(ollama_status(args.ollama_url, args.skip_ollama, args.profile))

    print(f"VetQwen preflight ({args.profile} profile)")
    print("=" * (23 + len(args.profile)))
    for state, message in statuses:
        print(status_line(state, message))

    has_failures = any(state == "FAIL" for state, _ in statuses)
    return 1 if has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
