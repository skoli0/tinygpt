"""
Train a simple MNIST classifier using Apple's MLX backend (macOS / Apple Silicon).

Usage:
  python -m pip install -e .
  python examples/mnist_mlx.py
"""

from __future__ import annotations

import gzip
import os
import pickle
import struct
import time
from itertools import batched
from urllib import request
from urllib.error import HTTPError, URLError

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


# MNIST mirrors. The original Yann LeCun host is occasionally unavailable / returns 404.
MNIST_URLS = [
    "https://yann.lecun.com/exdb/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]

MNIST_FILES = [
    ("training_images", "train-images-idx3-ubyte.gz"),
    ("test_images", "t10k-images-idx3-ubyte.gz"),
    ("training_labels", "train-labels-idx1-ubyte.gz"),
    ("test_labels", "t10k-labels-idx1-ubyte.gz"),
]


def _download_file(urls: list[str], save_path: str) -> None:
    """Download a file from the first working URL in `urls`."""
    filename = os.path.basename(save_path)
    last_err: Exception | None = None

    for base_url in urls:
        url = base_url + filename
        print(f"Downloading file {filename} from {base_url} ...")
        try:
            req = request.Request(url, headers={"User-Agent": "tinygpt-mnist-downloader/1.0"})
            with request.urlopen(req) as resp, open(save_path, "wb") as f:
                f.write(resp.read())
            return
        except (HTTPError, URLError, OSError) as e:
            last_err = e
            continue

    raise RuntimeError(f"Failed to download {filename} from all mirrors") from last_err


def _extract_images(file_path: str) -> list[list[float]]:
    with gzip.open(file_path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">4I", f.read(16))
        assert magic == 2051, "Invalid magic number for image file"
        assert num_images in [10000, 60000], "Invalid number of images"
        assert rows == 28 and cols == 28, "Invalid image dimensions"

        images = [
            [float(pixel) / 255.0 for pixel in struct.unpack(f">{rows*cols}B", f.read(rows * cols))]
            for _ in range(num_images)
        ]
        return images


def _extract_labels(file_path: str) -> list[int]:
    with gzip.open(file_path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        assert magic == 2049, "Invalid magic number for label file"
        assert num_labels in [10000, 60000], "Invalid number of labels"
        labels = list(struct.unpack(f">{num_labels}b", f.read(num_labels)))
        return [int(x) for x in labels]


def download_mnist_dataset(save_dir: str = "/tmp", base_urls: list[str] = MNIST_URLS, filename: str = "mnist.pkl"):
    """
    Returns:
      train_img: list[list[float]] of length 60000, each 784 floats in [0,1]
      train_labels: list[int] of length 60000
      test_img: list[list[float]] of length 10000
      test_labels: list[int] of length 10000
    """
    dataset_path = os.path.join(save_dir, filename)
    if os.path.exists(dataset_path):
        print("MNIST dataset already downloaded.")
    else:
        mnist_data = {}
        for name, file in MNIST_FILES:
            file_path = os.path.join(save_dir, file)
            _download_file(base_urls, file_path)
            if "images" in name:
                mnist_data[name] = _extract_images(file_path)
            else:
                mnist_data[name] = _extract_labels(file_path)

        with open(dataset_path, "wb") as f:
            pickle.dump(mnist_data, f)
        print("MNIST dataset downloaded and saved.")

    print("Loading MNIST dataset...")
    with open(dataset_path, "rb") as f:
        mnist = pickle.load(f)

    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


def _log_softmax(logits: mx.array, axis: int = -1) -> mx.array:
    """Version-agnostic log_softmax for MLX builds that may not expose mx.log_softmax."""
    if hasattr(mx, "log_softmax"):
        return mx.log_softmax(logits, axis=axis)

    m = mx.max(logits, axis=axis)
    m = m[..., None] if axis == -1 else mx.expand_dims(m, axis)
    y = logits - m
    if hasattr(mx, "logsumexp"):
        try:
            lse = mx.logsumexp(y, axis=axis, keepdims=True)
        except TypeError:
            lse = mx.logsumexp(y, axis=axis)
            lse = lse[..., None] if axis == -1 else mx.expand_dims(lse, axis)
        return y - lse

    denom = mx.sum(mx.exp(y), axis=axis)
    denom = denom[..., None] if axis == -1 else mx.expand_dims(denom, axis)
    return y - mx.log(denom)


def cross_entropy_loss(logits: mx.array, labels: mx.array) -> mx.array:
    """
    logits: [B, C]
    labels: [B] int
    returns: scalar mean NLL
    """
    logits = logits.astype(mx.float32) if logits.dtype != mx.float32 else logits
    logprobs = _log_softmax(logits, axis=-1)
    row = mx.arange(labels.shape[0])
    nll = -logprobs[row, labels]
    return mx.mean(nll)


def accuracy(logits: mx.array, labels: mx.array) -> mx.array:
    preds = mx.argmax(logits, axis=-1)
    return mx.mean((preds == labels).astype(mx.float32))


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = nn.relu(x)
        return self.fc2(x)


if __name__ == "__main__":
    # Data (cached as /tmp/mnist.pkl by default)
    train_img, train_labels, test_img, test_labels = download_mnist_dataset()

    # Hyperparams
    num_epochs = 5
    batch_size = 256
    learning_rate = 1e-1

    # Model + optimizer
    model = MNISTModel()
    opt = optim.SGD(learning_rate=learning_rate, momentum=0.9)

    def loss_fn(x: mx.array, y: mx.array) -> mx.array:
        # For `nn.value_and_grad(model, ...)`, MLX captures model parameters separately,
        # so the loss function should only take the data inputs.
        return cross_entropy_loss(model(x), y)

    # Prefer module-aware gradients (more stable across MLX versions).
    if hasattr(nn, "value_and_grad"):
        loss_and_grad = nn.value_and_grad(model, loss_fn)

        def step(x: mx.array, y: mx.array):
            return loss_and_grad(x, y)

    else:
        # Fallback: differentiate loss_fn w.r.t. trainable parameters via closure.
        # (Some MLX versions only expose value_and_grad on mx.core.)
        loss_and_grad = mx.value_and_grad(loss_fn)

        def step(x: mx.array, y: mx.array):
            return loss_and_grad(x, y)

    num_train_iterations = int(len(train_img) / batch_size + 1)
    num_test_iterations = int(len(test_img) / batch_size + 1)

    print("Beginning training (MLX)...")
    for epoch in range(num_epochs):
        # Train
        for it, (images, labels) in enumerate(zip(batched(train_img, batch_size), batched(train_labels, batch_size))):
            x = mx.array(images, dtype=mx.float32)
            y = mx.array(labels, dtype=mx.int32)

            tic = time.perf_counter()
            loss, grads = step(x, y)
            opt.update(model, grads)
            mx.eval(loss)
            toc = time.perf_counter()

            if it % 20 == 0:
                acc = accuracy(model(x), y)
                mx.eval(acc)
                print(
                    f"[TRAIN][Epoch {epoch + 1}/{num_epochs}][it {it + 1}/{num_train_iterations}]"
                    f" loss={float(loss.item()):.4f} acc={float(acc.item()):.4f}"
                    f" | {toc - tic:.3f}s"
                )

        # Test
        test_losses = []
        test_accs = []
        for it, (images, labels) in enumerate(zip(batched(test_img, batch_size), batched(test_labels, batch_size))):
            x = mx.array(images, dtype=mx.float32)
            y = mx.array(labels, dtype=mx.int32)
            logits = model(x)
            loss = cross_entropy_loss(logits, y)
            acc = accuracy(logits, y)
            mx.eval(loss, acc)
            test_losses.append(float(loss.item()))
            test_accs.append(float(acc.item()))

            if it % 20 == 0:
                print(
                    f"[TEST][Epoch {epoch + 1}/{num_epochs}][it {it + 1}/{num_test_iterations}]"
                    f" loss={test_losses[-1]:.4f} acc={test_accs[-1]:.4f}"
                )

        print(
            f"[TEST][Epoch {epoch + 1}/{num_epochs}]"
            f" mean_loss={sum(test_losses)/max(1,len(test_losses)):.4f}"
            f" mean_acc={sum(test_accs)/max(1,len(test_accs)):.4f}"
        )


