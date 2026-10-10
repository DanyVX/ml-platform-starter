"""Cross-platform task runner for environments without make."""

from __future__ import annotations

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
        "setup", "lint", "format", "typecheck", "test", "test-fast", "bench",
        "run", "docker-build", "clean", "doctor",
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
    elif task in {"run", "doctor"}:
        call(sys.executable, "-m", "mlp", "doctor")
    elif task == "clean":
        clean()
    elif task == "bench":
        unsupported("Benchmark runner arrives in M2; no measured results are available yet.")
    elif task == "docker-build":
        unsupported("Dockerfile and container build arrive in M3/M7, not M0.")


if __name__ == "__main__":
    main()
