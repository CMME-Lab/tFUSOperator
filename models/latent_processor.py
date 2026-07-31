"""
TFUS Neural Operator — Latent Processor.

Stack of transformer blocks operating on the latent tokens (B, M, d).
Each block is a pre-norm self-attention + MLP block. If `use_dit=True`
(default), DiTModulation is attached and the block can be conditioned on
c ∈ R^(B, d).

Forward conventions:
    block(x, c=None)   -> vanilla pre-norm transformer block
    block(x, c=c_vec)  -> DiT-conditioned (scale/shift/gate on each sub-block)

DiT initialization:
    DiTModulation.proj (the Linear(d -> 6d)) is zero-init, so at training
    start mod(c) == 0 for any c. This means with c provided, the block
    behaves exactly as identity (γ=β=0, α=0), and conditioning ramps up
    as training proceeds — the original DiT recipe.

    With c=None, the block ignores modulation entirely and acts as a
    standard pre-norm transformer from step 0.

Notes:
    - We use PyTorch's scaled_dot_product_attention which dispatches to
      Flash / memory-efficient kernels when available.
    - Self-attention uses a single fused QKV linear projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditioning import DiTModulation, modulate


# ---------------------------------------------------------------------------
# Multi-head self-attention
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """Fused-QKV multi-head self-attention. No masking (all tokens valid)."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d)
        B, N, d = x.shape
        H, Hd = self.num_heads, self.head_dim
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # reshape to (B, H, N, Hd)
        q = q.view(B, N, H, Hd).transpose(1, 2)
        k = k.view(B, N, H, Hd).transpose(1, 2)
        v = v.view(B, N, H, Hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, N, d)
        return self.proj_drop(self.proj(out))


# ---------------------------------------------------------------------------
# Transformer block (DiT-capable)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block with optional DiT-style conditioning.

    Forward modes:
        forward(x, c=None)            -- vanilla pre-norm:
            x = x + Attn(LN(x))
            x = x + MLP (LN(x))
        forward(x, c=c_vec) [DiT]     -- DiT-modulated pre-norm:
            split c -> (γ₁, β₁, α₁, γ₂, β₂, α₂)
            x = x + α₁ * Attn(modulate(LN(x), γ₁, β₁))
            x = x + α₂ * MLP (modulate(LN(x), γ₂, β₂))

    If `use_dit=False`, the modulation module is omitted entirely and
    passing a non-None c raises an error.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_dit: bool = True,
    ):
        super().__init__()
        self.use_dit = use_dit

        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

        if use_dit:
            self.mod = DiTModulation(dim)
        else:
            self.mod = None

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        # Dispatch by conditioning availability.
        if c is None or self.mod is None:
            if c is not None and self.mod is None:
                raise RuntimeError(
                    "TransformerBlock built with use_dit=False but a "
                    "condition vector c was provided."
                )
            # Vanilla pre-norm path.
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

        # DiT-conditioned path.
        # mod(c): (B, 6d) -> 6 chunks of (B, d)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.mod(c).chunk(6, dim=-1)

        h = modulate(self.norm1(x), gamma1, beta1)
        x = x + alpha1.unsqueeze(1) * self.attn(h)

        h = modulate(self.norm2(x), gamma2, beta2)
        x = x + alpha2.unsqueeze(1) * self.mlp(h)
        return x


# ---------------------------------------------------------------------------
# Latent processor (stack of blocks)
# ---------------------------------------------------------------------------

class LatentProcessor(nn.Module):
    """
    Stack of L TransformerBlocks operating on latent tokens.

    Args:
        dim:        d, latent token width.
        depth:      L, number of blocks.
        num_heads:  attention heads per block.
        mlp_ratio:  MLP expansion ratio inside each block.
        dropout:    dropout in attention and MLP.
        use_dit:    if True, each block has its own DiTModulation. If False,
                    blocks are vanilla and c must be None at forward time.
        final_norm: apply a final LayerNorm to the output (standard ViT).
    """

    def __init__(
        self,
        dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_dit: bool = True,
        final_norm: bool = True,
    ):
        super().__init__()
        self.depth = depth
        self.use_dit = use_dit

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, dropout=dropout,
                use_dit=use_dit,
            )
            for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(dim) if final_norm else nn.Identity()

    def forward(self, z: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        """
        z: (B, M, d)
        c: (B, d) or None
        returns: (B, M, d)
        """
        for blk in self.blocks:
            z = blk(z, c=c)
        return self.norm_out(z)