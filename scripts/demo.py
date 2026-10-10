"""One-command local platform demo, no cloud services and no real data."""

from __future__ import annotations

import subprocess
import time

import httpx

COMPOSE = ["docker", "compose", "-f", "deploy/docker-compose.yml"]


def command(*args: str) -> None:
    subprocess.run([*COMPOSE, *args], check=True)


def main() -> None:
    command("up", "-d", "--build")
    command("exec", "-T", "api", "python", "-m", "mlp", "train")
    for _ in range(60):
        try:
            response = httpx.get("http://127.0.0.1:8000/readyz", timeout=2)
            if response.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(2)
    else:
        raise SystemExit("Serving never became ready after training")
    record = {**{f"f_{i}": 0.1 for i in range(6)}, "channel": "web"}
    for _ in range(25):
        result = httpx.post(
            "http://127.0.0.1:8000/v1/predict", json={"records": record}, timeout=10
        )
        result.raise_for_status()
    command("exec", "-T", "api", "python", "eval/drift_injection.py")
    print(
        "Demo running. MLflow localhost:5000, API localhost:8000, "
        "Prometheus localhost:9090, Grafana localhost:3000"
    )


if __name__ == "__main__":
    main()
