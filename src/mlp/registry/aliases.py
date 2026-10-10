"""Safe alias administration; never mutate aliases without explicit requested action."""

from __future__ import annotations

from mlflow import MlflowClient

from mlp.train.pipeline import MODEL_NAME


def alias_version(alias: str, client: MlflowClient | None = None) -> str:
    registry = client or MlflowClient()
    return str(registry.get_model_version_by_alias(MODEL_NAME, alias).version)


def promote_version(version: str, client: MlflowClient | None = None) -> None:
    registry = client or MlflowClient()
    registry.get_model_version(MODEL_NAME, version)
    registry.set_registered_model_alias(MODEL_NAME, "champion", version)


def rollback_to(version: str, client: MlflowClient | None = None) -> None:
    """Restore a previously recorded champion, not an arbitrary newest version."""
    promote_version(version, client)
