"""Static Compose completeness and required service/healthcheck checks."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_four_service_platform_with_local_bindings() -> None:
    manifest = yaml.safe_load(Path("deploy/docker-compose.yml").read_text(encoding="utf-8"))
    services = manifest["services"]
    assert {"mlflow", "api", "prometheus", "grafana"} <= set(services)
    assert services["mlflow"]["healthcheck"]
    assert services["api"]["healthcheck"]
    for svc in services.values():
        for port in svc.get("ports", []):
            assert port.startswith("127.0.0.1:")


def test_dashboard_has_real_queries() -> None:
    import json

    dashboard = json.loads(
        Path("deploy/grafana/dashboards/platform.json").read_text(encoding="utf-8")
    )
    assert len(dashboard["panels"]) >= 4
    assert all(panel["targets"][0]["expr"] for panel in dashboard["panels"])
