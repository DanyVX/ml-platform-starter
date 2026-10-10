"""Cross-platform task runner for environments without make."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def call(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def unsupported(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(2)


def clean() -> None:
    for directory in (".pytest_cache", ".mypy_cache", ".ruff_cache", ".coverage"):
        candidate = ROOT / directory
        if candidate.is_dir():
            shutil.rmtree(candidate)
        elif candidate.is_file():
            candidate.unlink()
    for candidate in ROOT.rglob("__pycache__"):
        if candidate.is_dir():
            shutil.rmtree(candidate)


def main() -> None:
    tasks = {
        "setup",
        "lint",
        "format",
        "typecheck",
        "test",
        "test-fast",
        "bench",
        "run",
        "docker-build",
        "clean",
        "doctor",
        "train",
        "up",
        "down",
        "demo",
        "rollback",
    }
    task = sys.argv[1] if len(sys.argv) == 2 else ""
    if task not in tasks:
        unsupported(f"Specify one task: {', '.join(sorted(tasks))}")

    if task == "setup":
        if shutil.which("uv"):
            call("uv", "sync", "--extra", "dev")
        else:
            call(sys.executable, "-m", "pip", "install", "-e", ".[dev]")
    elif task == "lint":
        call("ruff", "check", ".")
        call("ruff", "format", "--check", ".")
    elif task == "format":
        call("ruff", "check", "--fix", ".")
        call("ruff", "format", ".")
    elif task == "typecheck":
        call("mypy", "src/mlp")
    elif task in {"test", "test-fast"}:
        call(sys.executable, "-m", "pytest", "-m", "not slow and not gpu")
    elif task == "train":
        call(sys.executable, "-m", "mlp", "train")
    elif task == "run":
        call(
            sys.executable,
            "-m",
            "uvicorn",
            "mlp.serve.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        )
    elif task == "doctor":
        call(sys.executable, "-m", "mlp", "doctor")
    elif task == "clean":
        clean()
    elif task == "bench":
        call(sys.executable, "eval/drift_injection.py")
    elif task == "docker-build":
        call("docker", "build", "-t", "ml-platform-starter:local", ".")
    elif task == "up":
        call("docker", "compose", "-f", "deploy/docker-compose.yml", "up", "-d", "--build")
    elif task == "down":
        call("docker", "compose", "-f", "deploy/docker-compose.yml", "down")
    elif task == "demo":
        call(sys.executable, "scripts/demo.py")
    elif task == "rollback":
        if not os.environ.get("MLP_ROLLBACK_VERSION"):
            unsupported("Set MLP_ROLLBACK_VERSION to an explicit previously healthy champion")
        call(sys.executable, "scripts/rollback.py", os.environ["MLP_ROLLBACK_VERSION"])


if __name__ == "__main__":
    main()
