"""Validated non-secret runtime configuration."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Environment-based settings; invalid values fail during initialization."""

    model_config = SettingsConfigDict(env_prefix="MLP_", env_file=".env", extra="ignore")

    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False
    random_seed: int = Field(default=42, ge=0, le=2**32 - 1)
    tiny_rows: int = Field(default=200, ge=40, le=10_000)
    validation_fraction: float = Field(default=0.2, gt=0, lt=0.5)
    test_fraction: float = Field(default=0.2, gt=0, lt=0.5)

    @model_validator(mode="after")
    def validate_fractions(self) -> Self:
        """Keep a useful training fraction even in tiny smoke configurations."""
        if self.validation_fraction + self.test_fraction >= 0.5:
            raise ValueError("MLP_VALIDATION_FRACTION + MLP_TEST_FRACTION must be below 0.5")
        return self

    def public_config(self) -> dict[str, str | int | float | bool]:
        """Only explicitly approved non-sensitive settings may be displayed."""
        return {
            "environment": self.environment,
            "log_level": self.log_level,
            "log_json": self.log_json,
            "random_seed": self.random_seed,
            "tiny_rows": self.tiny_rows,
            "validation_fraction": self.validation_fraction,
            "test_fraction": self.test_fraction,
        }
