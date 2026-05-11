from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any


def _truthy_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def registry_enabled() -> bool:
    """
    Controls whether we attempt to register a model in the MLflow Model Registry.

    - MLFLOW_REGISTER_MODEL=0 disables
    - MLFLOW_REGISTER_MODEL=1 enables (default)
    """
    return _truthy_env("MLFLOW_REGISTER_MODEL", default=True)


def model_name(default: str = "tinygpt-mlx") -> str:
    return os.getenv("MLFLOW_MODEL_NAME", default).strip() or default


def _maybe(path: Path) -> str | None:
    return str(path) if path.exists() else None


def log_and_register_mlx_tinygpt(
    mlflow: Any,
    *,
    repo_root: str | Path,
    checkpoint_path: str | Path,
    tokenizer_model_path: str | Path,
    gpt_config: Any,
    vocab_size: int,
    max_seq_length: int,
    artifact_path: str = "model",
) -> str | None:
    """
    Log a trained TinyGPT-MLX checkpoint as an MLflow Model (pyfunc).

    If MLFLOW_REGISTER_MODEL is enabled, also register it in the Model Registry
    using MLFLOW_MODEL_NAME (default: "tinygpt-mlx").

    Returns the logged model URI (e.g. "runs:/<run_id>/model") when possible.
    """
    if mlflow is None:
        return None

    repo_root = Path(repo_root)
    checkpoint_path = Path(checkpoint_path)
    tokenizer_model_path = Path(tokenizer_model_path)

    # Delay importing mlflow.pyfunc until runtime (keeps import light for non-mlflow runs).
    try:
        import mlflow.pyfunc  # type: ignore
    except Exception:
        return None

    # Persist a small config blob as an artifact.
    meta_dir = repo_root / "artifacts" / "mlflow_model"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "config.json"

    config_dict: dict[str, Any]
    try:
        # GPTConfig is a dataclass in mlx implementation; if it changes, fallback to repr().
        config_dict = asdict(gpt_config)  # type: ignore[arg-type]
    except Exception:
        config_dict = {"repr": repr(gpt_config)}

    meta_path.write_text(
        json.dumps(
            {
                "vocab_size": vocab_size,
                "max_seq_length": max_seq_length,
                "gpt_config": config_dict,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    tokenizer_vocab_path = tokenizer_model_path.with_suffix(".vocab")

    artifacts: dict[str, str] = {
        "checkpoint": str(checkpoint_path),
        "tokenizer_model": str(tokenizer_model_path),
        "config": str(meta_path),
    }
    if tokenizer_vocab_path.exists():
        artifacts["tokenizer_vocab"] = str(tokenizer_vocab_path)

    class TinyGPTMLXModel(mlflow.pyfunc.PythonModel):  # type: ignore[misc]
        def load_context(self, context):  # type: ignore[no-untyped-def]
            import json as _json
            from pathlib import Path as _Path

            import mlx.core as mx  # type: ignore

            from tinygpt.mlx_gpt import GPT, GPTConfig  # type: ignore
            from tinygpt.tokenizer import BPETokenizer, RegexPatterns  # type: ignore

            ckpt = _Path(context.artifacts["checkpoint"])
            tok_model = _Path(context.artifacts["tokenizer_model"])
            cfg = _Path(context.artifacts["config"])

            cfg_obj = _json.loads(cfg.read_text(encoding="utf-8"))
            self._max_seq_length = int(cfg_obj["max_seq_length"])
            self._vocab_size = int(cfg_obj["vocab_size"])
            gpt_cfg = cfg_obj.get("gpt_config", {})

            # Recreate config.
            try:
                self._config = GPTConfig(**gpt_cfg)
            except Exception:
                # Fallback to the minimal required fields used by this script.
                self._config = GPTConfig(
                    vocab_size=self._vocab_size,
                    max_seq_length=self._max_seq_length,
                    num_layers=int(gpt_cfg.get("num_layers", 6)),
                    num_heads=int(gpt_cfg.get("num_heads", 4)),
                    embedding_dim=int(gpt_cfg.get("embedding_dim", 128)),
                )

            self._tokenizer = BPETokenizer(regex_pattern=RegexPatterns.GPT4)
            self._tokenizer.load(str(tok_model))

            self._model = GPT(self._config)
            if hasattr(self._model, "astype"):
                try:
                    self._model = self._model.astype(mx.float32)
                except Exception:
                    pass
            if ckpt.exists():
                self._model.load_weights(str(ckpt), strict=False)

        def predict(self, context, model_input, params=None):  # type: ignore[no-untyped-def]
            import mlx.core as mx  # type: ignore

            # Accept either a DataFrame with a `prompt` column or a list[str].
            prompts: list[str]
            max_new_tokens = 128
            temperature = 0.8

            if params:
                if "max_new_tokens" in params:
                    max_new_tokens = int(params["max_new_tokens"])
                if "temperature" in params:
                    temperature = float(params["temperature"])

            try:
                # pandas DataFrame-like
                prompts = list(model_input["prompt"].astype(str).tolist())
                if "max_new_tokens" in model_input.columns:
                    max_new_tokens = int(model_input["max_new_tokens"].iloc[0])
                if "temperature" in model_input.columns:
                    temperature = float(model_input["temperature"].iloc[0])
            except Exception:
                prompts = [str(p) for p in model_input]

            outputs: list[str] = []
            for prompt in prompts:
                input_ids = self._tokenizer.encode(prompt, allowed_special="all")
                x = mx.array([input_ids], dtype=mx.int32)
                out = self._model.generate_sample(x, max_new_tokens=max_new_tokens, temperature=temperature)
                mx.eval(out)
                outputs.append(self._tokenizer.decode(out.tolist()[0]))
            return outputs

    registered_name = model_name() if registry_enabled() else None

    # Keep requirements minimal and MLX-focused.
    pip_requirements = [
        "mlflow",
        "mlx",
        "kernels",
        "numpy",
        "pandas",
        "requests",
        "rustbpe",
        "tiktoken",
    ]

    # Log as an MLflow Model. If a registry is available, this registers a new version.
    mlflow.pyfunc.log_model(  # type: ignore[attr-defined]
        artifact_path=artifact_path,
        python_model=TinyGPTMLXModel(),
        artifacts=artifacts,
        code_paths=[str(repo_root / "src")],
        pip_requirements=pip_requirements,
        registered_model_name=registered_name,
    )

    try:
        run_id = mlflow.active_run().info.run_id  # type: ignore[union-attr]
        return f"runs:/{run_id}/{artifact_path}"
    except Exception:
        return None

