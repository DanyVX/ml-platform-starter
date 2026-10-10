from __future__ import annotations

import json

import pytest

from mlp import __main__


def test_doctor_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(__main__, "environment_report", lambda: {"python": "3.11.0"})
    assert __main__.main(["doctor", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"python": "3.11.0"}


def test_doctor_text(monkeypatch, capsys) -> None:
    monkeypatch.setattr(__main__, "environment_report", lambda: {"gpu": None})
    assert __main__.main(["doctor"]) == 0
    assert "gpu: unknown" in capsys.readouterr().out


def test_config_is_valid_json(capsys) -> None:
    assert __main__.main(["config"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["environment"] in {"development", "test", "production"}
    assert "password" not in output


def test_missing_command_exits_with_error() -> None:
    with pytest.raises(SystemExit) as caught:
        __main__.main([])
    assert caught.value.code == 2
