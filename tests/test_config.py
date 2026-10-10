from __future__ import annotations

import pytest
from pydantic import ValidationError

from mlp.common.config import AppSettings


def test_defaults_are_cpu_friendly(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "MLP_ENVIRONMENT",
        "MLP_LOG_LEVEL",
        "MLP_LOG_JSON",
        "MLP_RANDOM_SEED",
        "MLP_TINY_ROWS",
        "MLP_VALIDATION_FRACTION",
        "MLP_TEST_FRACTION",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = AppSettings(_env_file=None)
    assert settings.tiny_rows == 200
    assert settings.random_seed == 42
    assert settings.validation_fraction + settings.test_fraction < 0.5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("random_seed", -1),
        ("tiny_rows", 0),
        ("tiny_rows", 10_001),
        ("environment", "invalid"),
        ("log_level", "TRACE"),
        ("validation_fraction", 0),
        ("test_fraction", 0.5),
    ],
)
def test_invalid_configuration_fails(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, **{field: value})


def test_invalid_split_sum_fails() -> None:
    with pytest.raises(ValidationError, match="must be below 0.5"):
        AppSettings(_env_file=None, validation_fraction=0.30, test_fraction=0.25)


def test_env_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLP_TINY_ROWS", "128")
    monkeypatch.setenv("MLP_LOG_JSON", "true")
    settings = AppSettings(_env_file=None)
    assert settings.tiny_rows == 128
    assert settings.log_json is True


def test_public_configuration_has_explicit_allowlist() -> None:
    result = AppSettings(_env_file=None).public_config()
    assert set(result) == {
        "environment",
        "log_level",
        "log_json",
        "random_seed",
        "tiny_rows",
        "validation_fraction",
        "test_fraction",
    }
