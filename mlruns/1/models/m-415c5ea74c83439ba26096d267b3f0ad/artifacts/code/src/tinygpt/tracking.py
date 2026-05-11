from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def mlflow_enabled(*, default: bool = True) -> bool:
    """
    Enable/disable MLflow via environment:
      - MLFLOW_ENABLE=0 disables
      - MLFLOW_ENABLE=1 enables (default)
    """
    return _truthy_env("MLFLOW_ENABLE", default=default)


def _import_mlflow():
    try:
        import mlflow  # type: ignore
    except Exception:
        return None
    return mlflow


@contextmanager
def mlflow_run(
    *,
    experiment: str = "tinygpt",
    run_name: str | None = None,
    tracking_uri: str | None = None,
    tags: dict[str, str] | None = None,
) -> Iterator[Any]:
    """
    Best-effort MLflow run context. If MLflow isn't available, yields None.
    """
    mlflow = _import_mlflow()
    if mlflow is None or not mlflow_enabled():
        yield None
        return

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags=tags):
        yield mlflow


def log_params(mlflow: Any, params: dict[str, Any]) -> None:
    if mlflow is None:
        return
    # MLflow requires simple scalar-ish values; stringify anything else.
    clean: dict[str, Any] = {}
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    if clean:
        mlflow.log_params(clean)


def log_metric(mlflow: Any, key: str, value: float, *, step: int | None = None) -> None:
    if mlflow is None:
        return
    mlflow.log_metric(key, float(value), step=step)


def log_artifact(mlflow: Any, path: str | Path, *, artifact_path: str | None = None) -> None:
    if mlflow is None:
        return
    p = Path(path)
    if not p.exists():
        return
    mlflow.log_artifact(str(p), artifact_path=artifact_path)

