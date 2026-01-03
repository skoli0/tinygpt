"""
Train a GPT model on Shakespeare using Apple's MLX backend (macOS / Apple Silicon).

This mirrors `examples/gpt.py` but uses `tinygpt.mlx_gpt` for the model + optimizer.

Usage (from repo root):
  python -m pip install -e .
  python examples/gpt_mlx.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from tinygpt.dataset import DatasetHandler, TextDataset
from tinygpt.mlx_gpt import GPT, GPTConfig, cross_entropy_loss
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


def validation(val_handler: DatasetHandler, model: GPT) -> None:
    losses = []
    for it, (input_ids, target_ids) in enumerate(val_handler):
        x = _as_int_batch(input_ids)
        y = _as_int_batch(target_ids)
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        mx.eval(loss)
        losses.append(float(loss.item()))
        print(f"[VAL][It. {it + 1:>5d}/{len(val_handler)}] Loss {losses[-1]:01.4f}")
    print(f"Mean val loss: {sum(losses) / max(1, len(losses)):.4f}")


def inference(model: GPT, tokenizer: BPETokenizer) -> None:
    raise RuntimeError("inference() now requires an explicit prompt; use --inference/--prompt.")


def run_inference(
    model: GPT,
    tokenizer: BPETokenizer,
    *,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> None:
    input_ids = tokenizer.encode(prompt, allowed_special="all")
    x = mx.array([input_ids], dtype=mx.int32)

    # print("Greedy")
    out = model.generate_greedy(x, max_new_tokens=max_new_tokens)
    mx.eval(out)
    print("-----")
    print(tokenizer.decode(out.tolist()[0]))

    print("Sample")
    for _ in range(3):
        out = model.generate_sample(x, max_new_tokens=max_new_tokens, temperature=temperature)
        mx.eval(out)
        print("-----")
        print(tokenizer.decode(out.tolist()[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/infer TinyGPT on MLX (Shakespeare).")
    parser.add_argument("--checkpoint", type=str, default=str(_REPO_ROOT / "examples/gpt_mlx_weights.npz"))
    parser.add_argument("--inference", action="store_true", help="Run inference instead of training.")
    parser.add_argument("--prompt", type=str, default="First Citizen:\n")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=sampling_temperature)
    args = parser.parse_args()

    print("Current recursion limit:", sys.getrecursionlimit())
    sys.setrecursionlimit(10000)

    # Tokenizer (reuse TinyGPT's BPE)
    tokenizer = BPETokenizer(regex_pattern=RegexPatterns.GPT4)
    try:
        text_corpus = data_path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "If you cloned the repo without data, run data/shakespeare/prepare.py first."
        ) from e

    if tokenizer_path is None and not args.inference:
        print("Training tokenizer...")
        tokenizer.train(text_corpus=text_corpus, vocab_size=vocab_size, verbose=True)
        tokenizer.save(str(_REPO_ROOT / "examples/tokenizer"))
    else:
        tokenizer.load(str(tokenizer_path))

    # Model
    config = GPTConfig(
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        num_layers=6,
        num_heads=4,
        embedding_dim=128,
    )
    model = GPT(config)
    # Force parameters to float32 to avoid MLX autodiff dtype/cotangent mismatches.
    if hasattr(model, "astype"):
        try:
            model = model.astype(mx.float32)
        except Exception:
            pass

    # Load checkpoint if present
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        model.load_weights(str(ckpt_path), strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")

    if args.inference:
        run_inference(
            model,
            tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        raise SystemExit(0)

    # Data
    train_dataset = TextDataset(data_file_path=train_path, tokenizer=tokenizer, max_seq_length=max_seq_length)
    train_handler = DatasetHandler(dataset=train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    val_dataset = TextDataset(data_file_path=val_path, tokenizer=tokenizer, max_seq_length=max_seq_length)
    val_handler = DatasetHandler(dataset=val_dataset, batch_size=batch_size, drop_last=True, shuffle=False)

    print(f"Train dataset with {len(train_dataset)} sequences")
    print(f"Val dataset with {len(val_dataset)} sequences")

    # Optimizer
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

    else:
        loss_and_grad = mx.value_and_grad(loss_fn)

        def _loss_and_grad_step(x: mx.array, y: mx.array):
            return loss_and_grad(x, y)

    print("Beginning training (MLX)...")
    for epoch in range(num_epochs):
        for it, (input_ids, target_ids) in enumerate(train_handler):
            x = _as_int_batch(input_ids)
            y = _as_int_batch(target_ids)

            start = time.time()
            loss, grads = _loss_and_grad_step(x, y)
            optimizer.update(model, grads)
            mx.eval(loss)
            end = time.time()

            if it % 10 == 0:
                print(
                    f"[Epoch {epoch + 1:>3d}/{num_epochs}][It. {it + 1:>5d}/{len(train_handler)}]"
                    f" Loss {float(loss.item()):01.4f} | step {(end - start):01.3f}s"
                )

        validation(val_handler, model)

        # Save checkpoint each epoch
        try:
            model.save_weights(str(ckpt_path))
            print(f"Saved checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"Warning: failed to save checkpoint to {ckpt_path}: {e}")


