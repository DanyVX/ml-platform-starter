"""Runtime discovery and reproducibility metadata. Never inspect user data."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _command_output(*args: str) -> str | None:
    if shutil.which(args[0]) is None:
        return None
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=3, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _ram_bytes() -> int | None:
    if not hasattr(os, "sysconf"):
        return None
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError):
        return None


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_report() -> dict[str, str | int | bool | None]:
    """Capture local runner properties; an absent probe means unknown, not unsupported."""
    cuda_probe = _command_output("nvcc", "--version")
    gpu_probe = _command_output("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
    return {
        "os": platform.platform(),
        "shell": os.environ.get("SHELL") or os.environ.get("COMSPEC"),
        "python": platform.python_version(),
        "python_executable": Path(sys.executable).name,
        "uv": shutil.which("uv") is not None,
        "pip": _version("pip"),
        "gpu": gpu_probe.splitlines()[0] if gpu_probe else None,
        "cuda": cuda_probe.splitlines()[-1] if cuda_probe else None,
        "ram_bytes": _ram_bytes(),
        "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
        "docker": shutil.which("docker") is not None,
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "git_name_configured": bool(_command_output("git", "config", "user.name")),
        "git_email_configured": bool(_command_output("git", "config", "user.email")),
        "git_commit": _command_output("git", "rev-parse", "HEAD"),
        "pydantic_settings_version": _version("pydantic-settings"),
        "internet": None,
    }
