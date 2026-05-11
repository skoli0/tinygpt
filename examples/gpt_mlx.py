"""
Train a GPT model on Shakespeare using Apple's MLX backend (macOS / Apple Silicon).

This mirrors `examples/gpt.py` but uses `tinygpt.mlx_gpt` for the model + optimizer.

### What this script does (step-by-step)

1) **Load dataset files** (`data/shakespeare/{input,train,val}.txt`). If they are missing, run
   `data/shakespeare/prepare.py` first.
2) **Load or train a tokenizer** (TinyGPT's BPE tokenizer).
3) **Build a GPT model** backed by MLX and optionally load a checkpoint.
4) **Train** for a small number of epochs:
   - forward pass → loss
   - compute gradients w.r.t. model parameters
   - update parameters with Adam
   - periodically log training loss + step time
5) **Validate** after each epoch and save a checkpoint.
6) Optionally run **inference** (greedy + sampling) using `--inference`.

### Usage (from repo root)

```bash
uv sync
uv run python data/shakespeare/prepare.py
uv run python examples/gpt_mlx.py
```
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from tinygpt.dataset import DatasetHandler, TextDataset
from tinygpt.mlx_gpt import GPT, GPTConfig, cross_entropy_loss
from tinygpt.mlflow_registry import log_and_register_mlx_tinygpt
from tinygpt.tracking import log_artifact, log_metric, log_params, mlflow_run
from tinygpt.tokenizer import BPETokenizer, RegexPatterns

# MLX autodiff can be sensitive to mixed precision depending on the installed
# version/build. Force float32 by default for stable training.
if hasattr(mx, "set_default_dtype"):
    try:
        mx.set_default_dtype(mx.float32)
    except Exception:
        pass


# Config (kept close to examples/gpt.py defaults)
vocab_size = 1024
max_seq_length = 64
batch_size = 16
num_epochs = 2
learning_rate = 3e-4
sampling_temperature = 0.8

# Resolve paths relative to the repo root so this script works no matter the CWD.
_REPO_ROOT = Path(__file__).resolve().parents[1]

data_path = _REPO_ROOT / "data/shakespeare/input.txt"
train_path = _REPO_ROOT / "data/shakespeare/train.txt"
val_path = _REPO_ROOT / "data/shakespeare/val.txt"
tokenizer_path = _REPO_ROOT / "examples/tokenizer.model"  # set to None to train tokenizer


def _as_int_batch(batch: list[list[int]]) -> mx.array:
    # DatasetHandler returns python lists; convert to int32 arrays for MLX.
    return mx.array(batch, dtype=mx.int32)


def _configure_logging(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger("gpt_mlx")


def _format_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def validation(val_handler: DatasetHandler, model: GPT, *, log: logging.Logger) -> float:
    losses = []
    for it, (input_ids, target_ids) in enumerate(val_handler):
        x = _as_int_batch(input_ids)
        y = _as_int_batch(target_ids)
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        mx.eval(loss)
        losses.append(float(loss.item()))
        if it == 0 or (it + 1) % 25 == 0 or (it + 1) == len(val_handler):
            log.info("[val] it=%d/%d loss=%.4f", it + 1, len(val_handler), losses[-1])
    mean = sum(losses) / max(1, len(losses))
    log.info("[val] mean_loss=%.4f", mean)
    return mean


def inference(model: GPT, tokenizer: BPETokenizer) -> None:
    raise RuntimeError("inference() now requires an explicit prompt; use --inference/--prompt.")


def run_inference(
    model: GPT,
    tokenizer: BPETokenizer,
    *,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    log: logging.Logger,
) -> None:
    input_ids = tokenizer.encode(prompt, allowed_special="all")
    x = mx.array([input_ids], dtype=mx.int32)

    log.info("[infer] prompt_len=%d max_new_tokens=%d temperature=%.3f", len(input_ids), max_new_tokens, temperature)

    out = model.generate_greedy(x, max_new_tokens=max_new_tokens)
    mx.eval(out)
    log.info("[infer] greedy")
    print("-----")
    print(tokenizer.decode(out.tolist()[0]))

    log.info("[infer] samples=3")
    for _ in range(3):
        out = model.generate_sample(x, max_new_tokens=max_new_tokens, temperature=temperature)
        mx.eval(out)
        print("-----")
        print(tokenizer.decode(out.tolist()[0]))

def _mlflow_env() -> tuple[str, str | None, str | None]:
    experiment = os.getenv("MLFLOW_EXPERIMENT", "tinygpt")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    run_name = os.getenv("MLFLOW_RUN_NAME")
    return experiment, tracking_uri, run_name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train/infer TinyGPT on MLX (Shakespeare).")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(_REPO_ROOT / "examples/gpt_mlx_weights.npz"),
        help="Path to save/load MLX checkpoint weights.",
    )
    parser.add_argument("--inference", action="store_true", help="Run inference instead of training.")
    parser.add_argument("--prompt", type=str, default="First Citizen:\n", help="Inference prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Tokens to generate during inference.")
    parser.add_argument("--temperature", type=float, default=sampling_temperature, help="Sampling temperature.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    args = parser.parse_args(argv)

    log = _configure_logging(args.log_level)

    log.info("Repo root: %s", _REPO_ROOT)
    log.info("Dataset: train=%s val=%s", train_path, val_path)
    log.info("Tokenizer: %s", tokenizer_path)

    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10000)
    if old_limit < 10000:
        log.info("Increased recursion limit: %d -> %d", old_limit, sys.getrecursionlimit())

    # Tokenizer (reuse TinyGPT's BPE)
    tokenizer = BPETokenizer(regex_pattern=RegexPatterns.GPT4)
    try:
        text_corpus = data_path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Run `python data/shakespeare/prepare.py` (or `uv run ...`) first."
        ) from e

    if tokenizer_path is None and not args.inference:
        log.info("Training tokenizer (vocab_size=%d)...", vocab_size)
        tokenizer.train(text_corpus=text_corpus, vocab_size=vocab_size, verbose=True)
        out_prefix = _REPO_ROOT / "examples/tokenizer"
        tokenizer.save(str(out_prefix))
        log.info("Saved tokenizer to %s.(model|vocab)", out_prefix)
    else:
        tokenizer.load(str(tokenizer_path))
        log.info("Loaded tokenizer model: %s", tokenizer_path)

    # Model
    config = GPTConfig(
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        num_layers=6,
        num_heads=4,
        embedding_dim=128,
    )
    log.info(
        "Model config: vocab=%d seq_len=%d layers=%d heads=%d embed=%d lr=%.4g batch=%d epochs=%d",
        config.vocab_size,
        config.max_seq_length,
        config.num_layers,
        config.num_heads,
        config.embedding_dim,
        learning_rate,
        batch_size,
        num_epochs,
    )

    model = GPT(config)

    # Force parameters to float32 to avoid MLX autodiff dtype/cotangent mismatches.
    if hasattr(model, "astype"):
        try:
            model = model.astype(mx.float32)
        except Exception:
            log.debug("Model astype(mx.float32) failed; continuing.", exc_info=True)

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        model.load_weights(str(ckpt_path), strict=False)
        log.info("Loaded checkpoint: %s", ckpt_path)

    if args.inference:
        experiment, tracking_uri, run_name = _mlflow_env()
        with mlflow_run(
            experiment=experiment,
            run_name=run_name or "infer-mlx",
            tracking_uri=tracking_uri,
            tags={"stage": "inference", "backend": "mlx", "dataset": "shakespeare"},
        ) as mlf:
            log_params(
                mlf,
                {
                    "checkpoint": str(ckpt_path),
                    "prompt_len": len(args.prompt),
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                },
            )

            # Capture output to an artifact for reproducibility.
            out_dir = _REPO_ROOT / "artifacts" / "inference"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "generated.txt"
            with out_path.open("w", encoding="utf-8") as f:
                f.write(args.prompt)
                f.write("\n\n")

                input_ids = tokenizer.encode(args.prompt, allowed_special="all")
                x = mx.array([input_ids], dtype=mx.int32)

                greedy = model.generate_greedy(x, max_new_tokens=args.max_new_tokens)
                mx.eval(greedy)
                greedy_text = tokenizer.decode(greedy.tolist()[0])
                f.write("-----\n[greedy]\n")
                f.write(greedy_text)
                f.write("\n")

                samples: list[str] = []
                for _ in range(3):
                    out = model.generate_sample(
                        x, max_new_tokens=args.max_new_tokens, temperature=args.temperature
                    )
                    mx.eval(out)
                    samples.append(tokenizer.decode(out.tolist()[0]))
                for i, t in enumerate(samples, start=1):
                    f.write(f"-----\n[sample {i}]\n")
                    f.write(t)
                    f.write("\n")

            log_artifact(mlf, out_path, artifact_path="inference")

            # Keep existing terminal behavior.
            run_inference(
                model,
                tokenizer,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                log=log,
            )
        return 0

    # Data
    train_dataset = TextDataset(data_file_path=train_path, tokenizer=tokenizer, max_seq_length=max_seq_length)
    train_handler = DatasetHandler(dataset=train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    val_dataset = TextDataset(data_file_path=val_path, tokenizer=tokenizer, max_seq_length=max_seq_length)
    val_handler = DatasetHandler(dataset=val_dataset, batch_size=batch_size, drop_last=True, shuffle=False)

    log.info("Train dataset: %d sequences | batches/epoch=%d", len(train_dataset), len(train_handler))
    log.info("Val dataset: %d sequences | batches/epoch=%d", len(val_dataset), len(val_handler))

    optimizer = optim.Adam(learning_rate=learning_rate)

    def loss_fn(x: mx.array, y: mx.array) -> mx.array:
        # For `nn.value_and_grad(model, ...)`, MLX captures model parameters separately,
        # so the loss function should only take the data inputs.
        return cross_entropy_loss(model(x), y)

    # MLX: compute grads w.r.t. model parameters
    if hasattr(nn, "value_and_grad"):
        loss_and_grad = nn.value_and_grad(model, loss_fn)

        def _loss_and_grad_step(x: mx.array, y: mx.array):
            return loss_and_grad(x, y)

        log.info("Gradients: using mlx.nn.value_and_grad(model, loss_fn)")
    else:
        loss_and_grad = mx.value_and_grad(loss_fn)

        def _loss_and_grad_step(x: mx.array, y: mx.array):
            return loss_and_grad(x, y)

        log.info("Gradients: using mlx.core.value_and_grad(loss_fn)")

    log.info("Beginning training (MLX)...")
    experiment, tracking_uri, run_name = _mlflow_env()
    with mlflow_run(
        experiment=experiment,
        run_name=run_name or "train-mlx",
        tracking_uri=tracking_uri,
        tags={"stage": "training", "backend": "mlx", "dataset": "shakespeare"},
    ) as mlf:
        log_params(
            mlf,
            {
                "vocab_size": vocab_size,
                "max_seq_length": max_seq_length,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "num_layers": config.num_layers,
                "num_heads": config.num_heads,
                "embedding_dim": config.embedding_dim,
                "checkpoint": str(ckpt_path),
                "tokenizer_model": str(tokenizer_path) if tokenizer_path is not None else None,
            },
        )
        log_artifact(mlf, tokenizer_path, artifact_path="tokenizer")

        global_step = 0
        total_start = time.time()
        for epoch in range(num_epochs):
            epoch_start = time.time()
            last_loss: float | None = None

            for it, (input_ids, target_ids) in enumerate(train_handler):
                x = _as_int_batch(input_ids)
                y = _as_int_batch(target_ids)

                step_start = time.time()
                loss, grads = _loss_and_grad_step(x, y)
                optimizer.update(model, grads)
                mx.eval(loss)
                step_s = time.time() - step_start

                last_loss = float(loss.item())
                if it == 0 or (it + 1) % 10 == 0 or (it + 1) == len(train_handler):
                    log.info(
                        "[train] epoch=%d/%d it=%d/%d loss=%.4f step=%s",
                        epoch + 1,
                        num_epochs,
                        it + 1,
                        len(train_handler),
                        last_loss,
                        _format_seconds(step_s),
                    )
                    log_metric(mlf, "train/loss", last_loss, step=global_step)
                    log_metric(mlf, "train/step_seconds", step_s, step=global_step)

                global_step += 1

            val_mean = validation(val_handler, model, log=log)
            log_metric(mlf, "val/mean_loss", val_mean, step=epoch)
            log_metric(mlf, "train/epoch_seconds", time.time() - epoch_start, step=epoch)
            log.info(
                "[epoch] %d/%d done: train_last_loss=%s val_mean_loss=%.4f epoch_time=%s",
                epoch + 1,
                num_epochs,
                f"{last_loss:.4f}" if last_loss is not None else "n/a",
                val_mean,
                _format_seconds(time.time() - epoch_start),
            )

            # Save checkpoint each epoch
            try:
                model.save_weights(str(ckpt_path))
                log.info("Saved checkpoint: %s", ckpt_path)
                log_artifact(mlf, ckpt_path, artifact_path="checkpoints")
            except Exception:
                log.warning("Failed to save checkpoint: %s", ckpt_path, exc_info=True)

        log_metric(mlf, "run/total_seconds", time.time() - total_start)
        log.info("Training complete. Total time: %s", _format_seconds(time.time() - total_start))

        # Log and (optionally) register the final model in the MLflow Model Registry.
        try:
            model_uri = log_and_register_mlx_tinygpt(
                mlf,
                repo_root=_REPO_ROOT,
                checkpoint_path=ckpt_path,
                tokenizer_model_path=tokenizer_path,
                gpt_config=config,
                vocab_size=vocab_size,
                max_seq_length=max_seq_length,
            )
            if model_uri:
                log.info("MLflow model logged: %s", model_uri)
        except Exception:
            log.warning("MLflow model log/register failed.", exc_info=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
