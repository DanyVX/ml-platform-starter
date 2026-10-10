from __future__ import annotations

from mlp.common import environment


def test_missing_probes_do_not_crash(monkeypatch) -> None:
    monkeypatch.setattr(environment, "_command_output", lambda *args: None)
    report = environment.environment_report()
    assert report["gpu"] is None
    assert report["cuda"] is None
    assert report["internet"] is None
    assert report["git_commit"] is None
    assert isinstance(report["python"], str)
    assert isinstance(report["disk_free_bytes"], int)


def test_invalid_command_is_unknown(monkeypatch) -> None:
    monkeypatch.setattr(environment.shutil, "which", lambda command: None)
    assert environment._command_output("command-that-is-not-installed") is None


def test_missing_package_returns_none() -> None:
    assert environment._version("this-package-is-not-installed-abc123") is None


def test_probe_keys_have_no_private_identifiers(monkeypatch) -> None:
    monkeypatch.setattr(environment, "_command_output", lambda *args: None)
    output = environment.environment_report()
    assert "email" not in output
    assert "username" not in output
    assert output["git_email_configured"] is False
