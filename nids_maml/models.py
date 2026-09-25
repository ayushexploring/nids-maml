"""Base learners for few-shot intrusion detection.

The Transformer here differs from the implementation it replaces in one
structural respect. The previous model projected the whole feature vector to a
single embedding and inserted a length-1 sequence axis, so self-attention ran
over a sequence of one token. Attention over a single position is the identity
map after softmax, which reduced every encoder block to a linear layer and made
the positional encoding a constant bias. This module tokenises *per feature*,
following the feature-tokenizer design of Gorishniy et al. (2021), so attention
runs over ``n_features + 1`` tokens and the attention matrix is a genuine
feature-interaction structure.

Every module is written so that ``torch.func.functional_call`` can drive it with
externally supplied parameters, which is what the MAML inner loop requires.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureTokenizer(nn.Module):
    """Map a numeric feature vector to one token per feature, plus a [CLS] token.

    Feature ``j`` with value ``x_j`` becomes ``x_j * W_j + b_j`` where ``W_j``
    and ``b_j`` are that feature's own learned vectors in R^d. This gives every
    column a distinct, learned identity in embedding space, which is what makes
    attention weights interpretable as feature interactions.
    """

    def __init__(self, n_features: int, d_model: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))
        self.cls = nn.Parameter(torch.empty(1, 1, d_model))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.d_model)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.uniform_(self.cls, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features) -> (B, n_features + 1, d_model)
        tokens = x.unsqueeze(-1) * self.weight + self.bias
        cls = self.cls.expand(x.shape[0], -1, -1)
        return torch.cat([cls, tokens], dim=1)


class SlotEncoding(nn.Module):
    """Additive per-slot identity codes over the token axis.

    Self-attention is permutation-equivariant, so without a per-slot signal the
    encoder cannot tell one column from another. These codes do not represent
    time or order: they assign each column of a fixed schema a unique identity.
    ``sinusoidal`` is parameter-free, which matters at this label budget;
    ``learned`` is the ablation alternative; ``none`` isolates the contribution
    of slot identity altogether -- note that the feature tokenizer already gives
    each column its own projection, so ``none`` is not a degenerate setting.
    """

    def __init__(self, n_tokens: int, d_model: int, kind: str = "sinusoidal") -> None:
        super().__init__()
        if kind not in ("sinusoidal", "learned", "none"):
            raise ValueError(f"unknown slot encoding: {kind}")
        self.kind = kind
        if kind == "sinusoidal":
            pos = torch.arange(n_tokens, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32)
                * (-math.log(10000.0) / d_model)
            )
            code = torch.zeros(1, n_tokens, d_model)
            code[0, :, 0::2] = torch.sin(pos * div)
            code[0, :, 1::2] = torch.cos(pos * div[: d_model // 2])
            self.register_buffer("code", code)
        elif kind == "learned":
            self.code = nn.Parameter(torch.zeros(1, n_tokens, d_model))
            nn.init.normal_(self.code, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "none":
            return x
        return x + self.code[:, : x.shape[1]]


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention written out in explicit tensor operations.

    ``nn.MultiheadAttention`` is deliberately not used. Its fused fast path
    (``aten::_native_multi_head_attention``) has no functorch derivative, so it
    cannot be driven by ``torch.func.vmap``, which is what makes vectorised
    meta-training possible. Writing the projections out has a second benefit
    for this work: ``q_proj``, ``k_proj`` and ``v_proj`` become individually
    named parameters, so the claim that the inner loop adapts the attention
    mechanism itself can be checked directly against per-module update norms
    rather than asserted.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        b, n, _ = t.shape
        return t.view(b, n, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self, x: torch.Tensor, need_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, n, d = x.shape
        q = self._split(self.q_proj(x))
        k = self._split(self.k_proj(x))
        v = self._split(self.v_proj(x))

        scores = (q @ k.transpose(-2, -1)) * self.scale
        weights = scores.softmax(dim=-1)
        out = self.dropout(weights) @ v
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.out_proj(out), (weights.mean(dim=1) if need_weights else None)


class EncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block.

    Pre-norm is used rather than the post-norm arrangement of the original
    implementation: with a small inner-loop learning rate applied to every
    parameter including the attention projections, pre-norm keeps the residual
    path well conditioned across adaptation steps.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, need_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out, attn_w = self.attn(self.norm1(x), need_weights=need_weights)
        x = x + self.dropout(attn_out)
        x = x + self.ff(self.norm2(x))
        return x, attn_w


class FTTransformer(nn.Module):
    """Feature-tokenizing Transformer encoder with a linear classification head.

    Dropout is governed by ``self.training`` in the standard way, so
    ``model.eval()`` genuinely disables it. The previous implementation baked
    ``training=True`` into the layer calls at graph-construction time, which
    left dropout active during every evaluation and injected noise into all
    reported accuracies.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_blocks: int = 3,
        d_ff: int | None = None,
        dropout: float = 0.1,
        slot_encoding: str = "sinusoidal",
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_features = n_features
        self.n_classes = n_classes
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.slots = SlotEncoding(n_features + 1, d_model, slot_encoding)
        self.blocks = nn.ModuleList(
            EncoderBlock(d_model, n_heads, d_ff or 2 * d_model, dropout)
            for _ in range(n_blocks)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.slots(self.tokenizer(x))
        for block in self.blocks:
            h, _ = block(h)
        return self.head(self.norm(h[:, 0]))  # [CLS] token

    @torch.no_grad()
    def attention_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Per-block attention weights, for the interpretability analysis.

        Returns one ``(B, n_tokens, n_tokens)`` tensor per block, head-averaged.
        Token 0 is [CLS]; token ``j+1`` corresponds to feature ``j``.
        """
        was_training = self.training
        self.eval()
        try:
            h = self.slots(self.tokenizer(x))
            maps = []
            for block in self.blocks:
                h, w = block(h, need_weights=True)
                maps.append(w)
            return maps
        finally:
            self.train(was_training)


class MLPBaseline(nn.Module):
    """Plain MLP base learner, isolating the contribution of the encoder.

    This is also, structurally, what the original 'Transformer' reduced to once
    the length-1 sequence axis collapsed its attention -- so it doubles as a
    reference point for the corrected results.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_classes)
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


class ConvEncoder(nn.Module):
    """1D-CNN base learner over the feature axis, a common IDS baseline."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        channels: tuple[int, ...] = (64, 128),
        kernel: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = 1
        for c in channels:
            layers += [
                nn.Conv1d(prev, c, kernel, padding=kernel // 2),
                nn.BatchNorm1d(c),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = c
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(prev, n_classes)
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x.unsqueeze(1))
        return self.head(self.pool(h).squeeze(-1))


def build_model(name: str, n_features: int, n_classes: int, **kwargs) -> nn.Module:
    """Construct a base learner by name, ignoring options it does not accept."""
    builders = {
        "transformer": FTTransformer,
        "mlp": MLPBaseline,
        "cnn": ConvEncoder,
    }
    if name not in builders:
        raise ValueError(f"unknown model '{name}'; choose from {sorted(builders)}")
    cls = builders[name]
    import inspect

    accepted = set(inspect.signature(cls.__init__).parameters)
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(n_features=n_features, n_classes=n_classes, **filtered)
