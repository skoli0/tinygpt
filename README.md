<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)">
  <img alt="tinygpt" src="/assests/logo.svg" width="25%" height="25%">
</picture>

</div>

---
TinyGPT is a minimalistic library for implementing, training, and performing inference on GPT models from scratch in Python. Inspired by [NanoGPT](https://github.com/karpathy/nanoGPT), [Tinygrad](https://github.com/tinygrad/tinygrad), [Pytorch](https://github.com/pytorch/pytorch), and [MLX](https://github.com/ml-explore/mlx), TinyGPT aims to be as educational as possible, avoiding complex optimizations that might obscure the underlying concepts.

## Features
- **Pure Python Core**: The educational TinyGPT stack is pure Python; the optional MLX example uses Apple's `mlx` package for accelerated training on Apple Silicon.
- **Didactic Focus**: Prioritizes readability and understanding over optimization, making it an excellent learning tool.
- **Modular Design**: The library is divided into several modules, each focusing on a specific aspect of training and inference.

## Installation
TinyGPT requires **Python 3.12 or higher** (see `pyproject.toml`). For the best experience, use **uv** as the default virtual environment and package manager.

### From source
```bash
# Install uv (choose one)
brew install uv
# or: curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/skoli0/tinygpt.git
cd tinygpt

# Create .venv and install the project + dependencies
uv venv
source .venv/bin/activate
uv sync
```

## Build + run with Make (recommended)
This repo ships with a `Makefile` that wraps the `uv` workflow and provides a simple end-to-end loop for **data → train → inference**, with **MLflow tracking** built in.

### Common targets

```bash
make sync
make data
make train
make infer PROMPT="First Citizen:\n"
```

### Build / test / clean

```bash
make build
make test
make clean
```

### Troubleshooting: `VIRTUAL_ENV ... does not match ... .venv`
If you have **any** virtualenv activated (even this repo’s `.venv`), `uv sync` may warn that `$VIRTUAL_ENV` doesn’t match the project environment path `.venv`.

Fix options:

```bash
# Option A (recommended): don’t activate; just use uv
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
uv sync

# Option B: if you *do* want to target the active venv
uv sync --active
```

### MLflow UI
By default, `make train` / `make infer` will start an MLflow run and write to the local file store (usually `./mlruns/`).

```bash
make mlflow-ui
```

Then open `http://127.0.0.1:5000`.

### MLflow configuration (env vars)
- **Disable tracking**:

```bash
MLFLOW_ENABLE=0 make train
```

- **Change experiment name**:

```bash
MLFLOW_EXPERIMENT=tinygpt-dev make train
```

- **Remote tracking server**:

```bash
MLFLOW_TRACKING_URI="http://localhost:5001" make train
```

- **Name a run**:

```bash
MLFLOW_RUN_NAME="shakespeare-quick" make train
```

### MLflow Model Registry (store trained models)
After training, the script logs an **MLflow Model** and (by default) registers it in the **MLflow Model Registry**.

- **Enable/disable registration**:

```bash
MLFLOW_REGISTER_MODEL=0 make train
```

- **Choose a registered model name**:

```bash
MLFLOW_MODEL_NAME="tinygpt-mlx-shakespeare" make train
```

- **Load a registered model for inference** (example):

```bash
uv run python -c 'import mlflow; m = mlflow.pyfunc.load_model("models:/tinygpt-mlx-shakespeare/latest"); print(m.predict(["First Citizen:\\n"]))'
```

### Run without activating the venv
Use `uv run` to execute commands in the project environment:

```bash
uv run python -c "import tinygpt; print(tinygpt.__version__)"
```

## Train GPT on MLX (macOS / Apple Silicon)
This fork includes an MLX-backed training script: `examples/gpt_mlx.py` (optimized for Apple Silicon).

### Prerequisites

- macOS **on Apple Silicon** (`arm64`)
- Python **3.12** recommended for MLX

If you see `ModuleNotFoundError: No module named 'mlx'`, you’re almost always running outside the uv environment (e.g. `python examples/gpt_mlx.py` using system Python). Use `uv run ...` commands below.

### 1) Prepare the Shakespeare dataset
If you don't already have `data/shakespeare/{input.txt,train.txt,val.txt}`:

```bash
uv run python data/shakespeare/prepare.py
```

### 2) Run MLX training
You can run this from **any working directory** (the script resolves paths relative to the repo).

```bash
uv run python examples/gpt_mlx.py
```

### 4) Run inference (load checkpoint + generate text)
`examples/gpt_mlx.py` saves a checkpoint each epoch (default: `examples/gpt_mlx_weights.npz`) and can run inference later:

```bash
uv run python examples/gpt_mlx.py --inference --prompt "First Citizen:\n" --max-new-tokens 128 --temperature 0.8
```

You can also specify a custom checkpoint path:

```bash
uv run python examples/gpt_mlx.py --checkpoint examples/my_gpt_weights.npz
uv run python examples/gpt_mlx.py --inference --checkpoint examples/my_gpt_weights.npz --prompt "To be, or not to be:\n"
```

Note: inference prints decoded text directly (not `repr(...)`), so newlines render naturally in your terminal.

### Troubleshooting (MLX port notes)
- **`FileNotFoundError: data/shakespeare/input.txt`**: the dataset wasn't prepared, or you ran from a different CWD. `examples/gpt_mlx.py` now resolves paths relative to the repo root (via `__file__`), and prints a clearer message if data is missing.
- **`TypeError: value_and_grad(): incompatible function arguments`**: MLX expects `value_and_grad` to wrap a function. The script uses a compatible `value_and_grad` setup.
- **`AttributeError: module 'mlx.core' has no attribute 'log_softmax'`**: some MLX versions expose different APIs; `tinygpt.mlx_gpt.cross_entropy_loss` uses a compatibility implementation.
- **`ValueError: [astype] Type of cotangents does not match primal output type.`**: this is usually a dtype/autodiff quirk across MLX versions; `examples/gpt_mlx.py` uses `mlx.nn.value_and_grad` (module-aware) to make training stable.

## Project Structure
  - `data/`: This directory contains scripts to download and prepare datasets.
    - `data/shakespeare/prepare.py`: Script to download the [Shakespeare](https://github.com/karpathy/nanoGPT/tree/master/data/shakespeare) dataset.
  - `docs/`: This directory holds the documentation for the project.
  - `examples/`: A collection of example scripts that demonstrate how to use the various components of the project.
    - `gpt.py`: Train a GPT model on Shakespeare dataset.
    - `gpt_mlx.py`: Train a GPT model on Shakespeare dataset using MLX (macOS / Apple Silicon).
  - `src/:` The core library directory where all the main functionalities of the project are implemented.
    - `src/tinygpt/buffer.py`: Provides a low-level implementation of array operations, similar to NumPy arrays.
    - `src/tinygpt/dataset.py`: Handles data loading and preprocessing.
    - `src/tinygpt/losses.py`: Contains implementations of loss functions used for training.
    - `src/tinygpt/mlops.py`: Provides a low-level implementation of array operations, similar to NumPy arrays.
    - `src/tinygpt/module.py`: Defines the base module from which all model components inherit.
    - `src/tinygpt/nn.py`: Contains the neural network components, including layers and activation functions.
    - `src/tinygpt/optimizers.py`: Implements optimization algorithms used during training.
    - `src/tinygpt/tensor.py`: Provides a minimal tensor implementation to support basic tensor operations.
    - `src/tinygpt/tokenizer.py`: Handles tokenization of text data for model input.
    - `src/tinygpt/utils.py`: Contains miscellaneous utility functions used across the library.
  - `tests/`: This directory includes test files for the library's components. Each module in the `src/` directory has corresponding test file.

## Examples
The `examples/` directory contains scripts demonstrating TinyGPT on the Shakespeare dataset:

  - `gpt.py`: A basic example of training and using a GPT model.
  - `gpt_mlx.py`: Train a GPT model on Shakespeare dataset using MLX (macOS/Apple Silicon).

To run these examples:

```bash
uv run python examples/gpt.py
uv run python examples/gpt_mlx.py
```

These examples will guide you through setting up the model, training it, and performing inference, providing a hands-on understanding of how TinyGPT works.

## Documentation
Documentation along with a quick start guide can be found in the `docs/` directory.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

### Testing
Install dependencies (see [Installation](#installation)), then run:

```bash
uv run pytest
```
