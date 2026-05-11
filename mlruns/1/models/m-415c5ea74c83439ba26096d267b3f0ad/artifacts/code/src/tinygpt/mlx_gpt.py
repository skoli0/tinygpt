"""
MLX backend for training GPT-style models on macOS (Apple Silicon).

This module is intentionally self-contained and does not affect the existing
pure-Python TinyGPT stack. It is only imported when you opt into MLX usage.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import mlx.core as mx
    import mlx.nn as nn
except Exception as e:  # pragma: no cover
    raise ImportError(
        "MLX is required for `tinygpt.mlx_gpt`. "
        "Install dependencies with `uv sync` and run via `uv run ...`."
    ) from e


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    max_seq_length: int
    num_layers: int
    num_heads: int
    embedding_dim: int
    mlp_hidden_mult: int = 4


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.embedding_dim % config.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.n_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.embed_dim = config.embedding_dim
        self.max_seq_length = config.max_seq_length

        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # Precompute a causal mask of shape [1, 1, T, T] with True on allowed positions.
        t = self.max_seq_length
        mask = mx.tril(mx.ones((t, t), dtype=mx.bool_))
        self._causal_mask = mask[None, None, :, :]

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, T, C]
        b, t, c = x.shape
        if c != self.embed_dim:
            raise ValueError(f"Expected last dim {self.embed_dim}, got {c}")
        if t > self.max_seq_length:
            raise ValueError(f"Sequence length {t} exceeds max_seq_length {self.max_seq_length}")

        qkv = self.qkv(x)  # [B, T, 3C]
        q, k, v = mx.split(qkv, 3, axis=-1)

        # [B, T, C] -> [B, H, T, D]
        q = q.reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Attention scores: [B, H, T, T]
        scale = 1.0 / (self.head_dim**0.5)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale

        # Causal mask
        mask = self._causal_mask[:, :, :t, :t]
        scores = mx.where(mask, scores, mx.array(-1e9, dtype=scores.dtype))

        weights = mx.softmax(scores, axis=-1)
        out = weights @ v  # [B, H, T, D]

        # [B, H, T, D] -> [B, T, C]
        out = out.transpose(0, 2, 1, 3).reshape(b, t, c)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = config.mlp_hidden_mult * config.embedding_dim
        self.fc1 = nn.Linear(config.embedding_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, config.embedding_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.embedding_dim)
        self.mlp = MLP(config)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.pos_emb = nn.Embedding(config.max_seq_length, config.embedding_dim)
        self.blocks = [Block(config) for _ in range(config.num_layers)]
        self.ln_f = nn.LayerNorm(config.embedding_dim)
        self.head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

    def __call__(self, token_ids: mx.array) -> mx.array:
        # token_ids: [B, T] int
        _, t = token_ids.shape
        if t > self.config.max_seq_length:
            raise ValueError(f"Sequence length {t} exceeds max_seq_length {self.config.max_seq_length}")

        pos = mx.arange(t)[None, :]  # [1, T]
        x = self.tok_emb(token_ids) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)  # [B, T, V]

    def generate_greedy(self, token_ids: mx.array, max_new_tokens: int) -> mx.array:
        if token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError("generate_greedy expects token_ids with shape [1, T]")

        out = token_ids
        for _ in range(max_new_tokens):
            cond = out[:, -self.config.max_seq_length :]
            logits = self(cond)[:, -1, :]  # [1, V]
            next_id = mx.argmax(logits, axis=-1)  # [1]
            out = mx.concatenate([out, next_id[:, None]], axis=1)
            mx.eval(out)
        return out

    def generate_sample(self, token_ids: mx.array, max_new_tokens: int, temperature: float = 1.0) -> mx.array:
        if token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError("generate_sample expects token_ids with shape [1, T]")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        out = token_ids
        for _ in range(max_new_tokens):
            cond = out[:, -self.config.max_seq_length :]
            logits = self(cond)[:, -1, :] / temperature  # [1, V]
            next_id = mx.random.categorical(logits)  # [1]
            out = mx.concatenate([out, next_id[:, None]], axis=1)
            mx.eval(out)
        return out


def cross_entropy_loss(logits: mx.array, targets: mx.array) -> mx.array:
    """
    logits: [B, T, V]
    targets: [B, T] integer token ids
    returns: scalar mean NLL
    """

    b, t, v = logits.shape
    logits_2d = logits.reshape((b * t, v))
    targets_1d = targets.reshape((b * t,))

    # Some MLX versions don't expose `log_softmax` on `mlx.core`.
    # Compute log-softmax in a version-agnostic, numerically-stable way.
    if hasattr(mx, "log_softmax"):
        logprobs = mx.log_softmax(logits_2d, axis=-1)
    elif hasattr(mx, "logsumexp"):
        try:
            logprobs = logits_2d - mx.logsumexp(logits_2d, axis=-1, keepdims=True)
        except TypeError:
            logprobs = logits_2d - mx.logsumexp(logits_2d, axis=-1)[..., None]
    else:
        m = mx.max(logits_2d, axis=-1)[..., None]
        y = logits_2d - m
        denom = mx.sum(mx.exp(y), axis=-1)[..., None]
        logprobs = y - mx.log(denom)

    row = mx.arange(targets_1d.shape[0])
    nll = -logprobs[row, targets_1d]
    return mx.mean(nll)

