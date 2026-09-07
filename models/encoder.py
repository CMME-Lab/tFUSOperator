"""
TFUS Neural Operator — Encoder (dual modality, coord-decoupled).

Encoder pipeline:

    p_ff volume (B, 1, H_A, W_A, D_A), coords_A (B, N_A, 3) in [-1,1]^3
        -> PatchEmbed3D_field (Conv3d k=s=P) -> features_A (B, N_A, d)

    CT volume   (B, 1, H_B, W_B, D_B), coords_B (B, N_B, 3) in [-1,1]^3
        -> PatchEmbed3D_skull (Conv3d k=s=P) -> features_B (B, N_B, d)

    coord_pe_A = MLP_coord( sinusoidal_PE_3D(coords_A) )       (B, N_A, d)
    coord_pe_B = MLP_coord( sinusoidal_PE_3D(coords_B) )       (B, N_B, d)

    value_A = MLP_value( cat[coord_pe_A, features_A] )         (B, N_A, d)
    value_A += modality_embed_field
    value_B = MLP_value( cat[coord_pe_B, features_B] )         (B, N_B, d)
    value_B += modality_embed_skull

    K = concat([coord_pe_A, coord_pe_B], axis=1)              (B, N_A+N_B, d)
    V = concat([value_A,    value_B   ], axis=1)              (B, N_A+N_B, d)

    Perceiver pool:
        H_enc: learnable (M, d)
        z0   = CrossAttn(Q=H_enc, K=K, V=V)                    (B, M, d)

Design rationale:
    - LNO-style decoupling: K depends on COORDINATES ONLY. V carries the
      (coord, feature) bundle. The attention score thus expresses
      "what physical region does this latent slot read from", separating
      where-to-look from what-to-read.
    - MLP_coord and MLP_value are SHARED across modalities so that physical
      position carries identical semantics regardless of which volume
      a token comes from. Modality is disambiguated only via the learned
      modality embedding added to V (not to K, so a focal-region patch and
      a wave-path patch at the same physical position would still get the
      same K but differ in V).
    - Sinusoidal PE in 3D is computed per-sample on the provided mm coords;
      it is no longer a fixed buffer, since coordinates can vary per sample.

Inputs expected by forward:
    field_vol   : (B, 1, H_A, W_A, D_A)   p_ff volume.
    field_coords: (B, N_A, 3)             patch-center mm coords for the
                                          field ROI, normalized so the
                                          combined (field, skull) bbox
                                          maps roughly to [-1, 1]^3.
    skull_vol   : (B, 1, H_B, W_B, D_B)   CT/MR volume.
    skull_coords: (B, N_B, 3)             patch-center coords for the
                                          skull ROI, same normalized frame.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding on continuous mm coordinates
# ---------------------------------------------------------------------------

class SinusoidalPE3D(nn.Module):
    """
    Sinusoidal positional encoding for batched 3D coordinates.

    Maps (B, N, 3) coords in [-1, 1]^3 to (B, N, dim).

    The hidden dim is split into 3 per-axis groups, each carrying
    K = (dim // 6) sin/cos frequency pairs. The lowest-frequency band has
    period 2 (matching the [-1, 1] normalization) and frequencies span a
    geometric series up to a maximum band.

    Args:
        dim:        output dim. Must be divisible by 6.
        num_bands:  number of frequency bands per axis (== dim // 6).
                    Provided redundantly only as a sanity check; if not
                    None and != dim // 6, raises.
        max_freq:   highest angular frequency for the top band, in
                    cycles per unit length. With coords in [-1, 1] the
                    period of the lowest band is 2, so we set the lowest
                    angular freq to pi (one full cycle across the domain).
                    Frequencies form a geometric sequence pi * base^k.
    """

    def __init__(self, dim: int, num_bands: int | None = None, max_freq: float | None = None):
        super().__init__()
        if dim % 6 != 0:
            raise ValueError(f"dim must be divisible by 6, got {dim}")
        K = dim // 6
        if num_bands is not None and num_bands != K:
            raise ValueError(f"num_bands={num_bands} inconsistent with dim={dim} (expected {K})")
        self.dim = dim
        self.K = K

        # Frequencies: pi * 2^k for k = 0..K-1.
        # k=0    -> lowest band, period 2 (matches [-1,1] domain).
        # k=K-1  -> highest band, period 2 / 2^(K-1).
        freqs = math.pi * (2.0 ** torch.arange(K, dtype=torch.float32))
        if max_freq is not None:
            # Optional override: cap top frequency, distribute geometrically.
            freqs = math.pi * torch.logspace(0, math.log2(max_freq / math.pi),
                                             steps=K, base=2.0)
        self.register_buffer("freqs", freqs, persistent=False)   # (K,)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (B, N, 3) in [-1, 1]^3 (or thereabouts).
        returns: (B, N, dim)
        """
        if coords.ndim != 3 or coords.size(-1) != 3:
            raise ValueError(f"expected coords shape (B, N, 3), got {tuple(coords.shape)}")
        # angles[..., axis, k] = coord_axis * freqs_k
        # broadcast: (B, N, 3, 1) * (1, 1, 1, K) -> (B, N, 3, K)
        angles = coords.unsqueeze(-1) * self.freqs.view(1, 1, 1, -1)
        pe = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, N, 3, 2K)
        pe = pe.flatten(2)                                    # (B, N, 3*2K) = (B, N, dim)
        return pe


# ---------------------------------------------------------------------------
# Patch embedding (single modality; one instance per modality)
# ---------------------------------------------------------------------------

class PatchEmbed3D(nn.Module):
    """
    Non-overlapping 3D patch embedding via strided Conv3d.

    Input : (B, C_in, H, W, D), with H, W, D divisible by P.
    Output: (B, Np, d), where Np = (H/P) * (W/P) * (D/P).

    Token order after flatten(2).transpose(1, 2) is z-fastest, i.e.,
    index = iz + Gz * (iy + Gy * ix). (PyTorch's contiguous memory layout
    has the LAST axis varying fastest.) The caller must produce
    patch-center coordinates in the matching order, e.g. via
        coord_grid = torch.stack(torch.meshgrid(
            x_axis, y_axis, z_axis, indexing='ij'), dim=-1)
        coords = coord_grid.view(-1, 3)
    which yields the same flattening convention by construction.
    """

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected 5D input, got shape {tuple(x.shape)}")
        x = self.proj(x)                            # (B, d, Gx, Gy, Gz)
        x = x.flatten(2).transpose(1, 2)            # (B, Np, d)
        return x


# ---------------------------------------------------------------------------
# Deep local CT stem (drop-in for PatchEmbed3D on the skull modality)
# ---------------------------------------------------------------------------

class ResBlock3d(nn.Module):
    """Pre-norm residual conv block, stride-1, kernel 3 (RF += 4 per block).

    GroupNorm (batch-independent) is deliberate: BatchNorm running stats would
    be computed over train skulls and become a distribution-shift liability at
    test time. GroupNorm normalizes per-sample, which is safer under the
    skull-level OOD we are fighting.
    """

    def __init__(self, c: int, groups: int = 8):
        super().__init__()
        g = min(groups, c)
        while c % g != 0:
            g -= 1
        self.norm1 = nn.GroupNorm(g, c)
        self.conv1 = nn.Conv3d(c, c, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(g, c)
        self.conv2 = nn.Conv3d(c, c, kernel_size=3, padding=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class CTEncoder(nn.Module):
    """Deep coord-free local CT stem. Drop-in replacement for PatchEmbed3D.

    Interface (identical to PatchEmbed3D):
        forward: (B, 1, H, W, D) -> (B, Np, d)
        Np and z-fastest token order are IDENTICAL to
        PatchEmbed3D(1, embed_dim, patch_size), because the final tokenizer is
        the same Conv3d(kernel=stride=P) + flatten(2).transpose(1,2), and the
        stride-1 ResBlocks preserve spatial dims (so coords still align).

    Why this is needed: the original skull pathway is a single Conv3d
        (~embed_dim*P^3 params; at P=1 only ~embed_dim*1 = 768 params), so each
        token sees only its own patch's raw HU and cannot read surrounding bone
        geometry. The stride-1 ResBlock stack gives each token a receptive
        field of RF = 3 + 4*depth voxels (e.g. depth=3 -> 15 voxels) so it can
        encode local bone thickness / curvature — the input to the
        (CT -> aberration) map that fails to generalize on unseen skulls.
    """

    def __init__(self, embed_dim: int, patch_size: int = 1,
                 channels: int = 64, depth: int = 3):
        super().__init__()
        if depth < 1:
            raise ValueError("CTEncoder depth must be >= 1; use PatchEmbed3D for depth 0")
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.stem_in = nn.Conv3d(1, channels, kernel_size=3, padding=1, bias=True)
        self.blocks = nn.ModuleList([ResBlock3d(channels) for _ in range(depth)])
        self.tokenize = nn.Conv3d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected 5D input, got shape {tuple(x.shape)}")
        h = self.stem_in(x)                         # (B, c, H, W, D)
        for blk in self.blocks:
            h = blk(h)                              # stride-1, dims preserved
        h = self.tokenize(h)                        # (B, d, Gx, Gy, Gz)
        return h.flatten(2).transpose(1, 2)         # (B, Np, d), z-fastest


# ---------------------------------------------------------------------------
# Coord-decoupled cross-attention (Perceiver pool)
# ---------------------------------------------------------------------------

class CoordDecoupledCrossAttention(nn.Module):
    """
    Cross-attention block where K is built from coordinate embeddings only
    and V is built from coordinate + feature embeddings. The query Q is
    typically the learned latent (B, M, d).

    Forward:
        z = LN(Q + Attn(Q_in=LN(Q), K_in=K, V_in=V))
        z = z + MLP(LN(z))

    Args:
        dim:        d.
        num_heads:  attention heads.
        mlp_ratio:  MLP expansion in the post-attn block.
        dropout:    dropout in attn projection and MLP.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)

        # Separate linear projections for Q, K, V (K and V come from different
        # input tensors here, so we can't share a fused QKV).
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(dropout)

        hidden = int(dim * mlp_ratio)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor) -> torch.Tensor:
        """
        q:    (B, Nq, d)
        k_in: (B, Nkv, d)
        v_in: (B, Nkv, d)
        returns: (B, Nq, d)
        """
        B, Nq, d = q.shape
        Nkv = k_in.size(1)
        H, Hd = self.num_heads, self.head_dim

        qn = self.norm_q(q)
        kn = self.norm_k(k_in)
        vn = self.norm_v(v_in)

        qh = self.to_q(qn).view(B, Nq, H, Hd).transpose(1, 2)   # (B, H, Nq, Hd)
        kh = self.to_k(kn).view(B, Nkv, H, Hd).transpose(1, 2)
        vh = self.to_v(vn).view(B, Nkv, H, Hd).transpose(1, 2)

        out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, Nq, d)
        out = self.proj_drop(self.proj(out))

        q = q + out
        q = q + self.mlp(self.norm_mlp(q))
        return q


# ---------------------------------------------------------------------------
# Dual-modality encoder
# ---------------------------------------------------------------------------

class TFUSDualEncoder(nn.Module):
    """
    Dual-modality encoder for (p_ff field, CT skull) -> fixed latent.

    Two separate PatchEmbed3D modules produce per-modality features.
    Coord embeddings are SHARED (single SinusoidalPE3D + single coord_mlp)
    so that physical position carries the same meaning across modalities.
    Modality is disambiguated by a learned per-modality vector added to V.

    Args:
        embed_dim:      d. Must be divisible by 6 (for sinusoidal PE on 3 axes)
                        and by num_heads.
        patch_size:     P. Patch side length in voxels. Applied to both
                        modalities (each ROI is assumed to share the same
                        voxel spacing).
        num_latents:    M, number of learnable latent tokens.
        num_heads:      attention heads.
        mlp_ratio:      MLP expansion in the cross-attention block.
        dropout:        dropout in cross-attention and MLP.
        coord_pe_bands: number of sinusoidal bands per axis (== d // 6).
                        Passed for sanity; if None, derived from embed_dim.
    """

    MODALITY_FIELD = "field"
    MODALITY_SKULL = "skull"

    def __init__(
        self,
        embed_dim: int = 384,
        patch_size: int = 4,
        num_latents: int = 512,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        coord_pe_bands: int | None = None,
        coord_max_freq: float | None = None,
        ct_stem_depth: int = 0,
        ct_stem_channels: int = 64,
    ):
        super().__init__()
        if embed_dim % 6 != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by 6")

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_latents = num_latents

        # Per-modality patch embedders (input channels = 1 for each).
        # field stays a plain patch embed (p_ff is free-field -> carries no
        # skull info, so it is not the OOD bottleneck). The skull pathway can
        # optionally use a deeper local conv stem (ct_stem_depth>0) to give
        # each token a receptive field over surrounding bone.
        self.patch_embed_field = PatchEmbed3D(1, embed_dim, patch_size)
        if ct_stem_depth > 0:
            self.patch_embed_skull = CTEncoder(
                embed_dim, patch_size,
                channels=ct_stem_channels, depth=ct_stem_depth,
            )
        else:
            self.patch_embed_skull = PatchEmbed3D(1, embed_dim, patch_size)

        # Shared coord PE and coord MLP (identical semantics across modalities).
        self.coord_pe = SinusoidalPE3D(embed_dim, num_bands=coord_pe_bands,
                                       max_freq=coord_max_freq)
        self.coord_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Shared value MLP: takes (coord_pe, feature) -> value.
        # Concatenation: coord_pe (d) || feature (d) -> Linear(2d -> d).
        self.value_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Per-modality learned embedding added to V.
        self.modality_embed = nn.ParameterDict({
            self.MODALITY_FIELD: nn.Parameter(torch.zeros(embed_dim)),
            self.MODALITY_SKULL: nn.Parameter(torch.zeros(embed_dim)),
        })
        for p in self.modality_embed.values():
            nn.init.trunc_normal_(p, std=0.02)

        # Learnable latent queries.
        self.latent_queries = nn.Parameter(torch.zeros(num_latents, embed_dim))
        nn.init.trunc_normal_(self.latent_queries, std=0.02)

        # Perceiver-style cross-attention pool.
        self.pool = CoordDecoupledCrossAttention(
            dim=embed_dim, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout,
        )

    def _build_kv_for_modality(
        self,
        features: torch.Tensor,     # (B, N, d)
        coords: torch.Tensor,        # (B, N, 3)
        modality: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            k: (B, N, d) — coord-only key embedding.
            v: (B, N, d) — (coord, feature)-based value embedding +
                          per-modality bias.
        """
        coord_pe_raw = self.coord_pe(coords)                # (B, N, d)
        coord_pe = self.coord_mlp(coord_pe_raw)             # (B, N, d)

        # V: built from concatenation of coord_pe and feature.
        v_in = torch.cat([coord_pe, features], dim=-1)      # (B, N, 2d)
        v = self.value_mlp(v_in)                            # (B, N, d)                     # (B, N, d)

        # Modality bias on V only — K stays purely coordinate-driven.
        v = v + self.modality_embed[modality].view(1, 1, -1)

        return coord_pe, v

    def forward(
        self,
        field_vol: torch.Tensor,
        field_coords: torch.Tensor,
        skull_vol: torch.Tensor,
        skull_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        field_vol   : (B, 1, H_A, W_A, D_A)
        field_coords: (B, N_A, 3)   patch-center coords in normalized frame.
        skull_vol   : (B, 1, H_B, W_B, D_B)
        skull_coords: (B, N_B, 3)

        N_A and N_B must match the per-modality patch counts implied by
        Conv3d(kernel=stride=P) applied to the respective volume.

        Returns:
            z0: (B, M, d)
        """
        B = field_vol.size(0)

        # 1. Patch embed each modality.
        feat_A = self.patch_embed_field(field_vol)       # (B, N_A, d)
        feat_B = self.patch_embed_skull(skull_vol)       # (B, N_B, d)

        # Sanity check: provided coords must match patch counts.
        if feat_A.size(1) != field_coords.size(1):
            raise ValueError(
                f"field patch count {feat_A.size(1)} != coord count {field_coords.size(1)}"
            )
        if feat_B.size(1) != skull_coords.size(1):
            raise ValueError(
                f"skull patch count {feat_B.size(1)} != coord count {skull_coords.size(1)}"
            )

        # 2. Build K, V per modality with SHARED coord/value MLPs.
        k_A, v_A = self._build_kv_for_modality(feat_A, field_coords, self.MODALITY_FIELD)
        k_B, v_B = self._build_kv_for_modality(feat_B, skull_coords, self.MODALITY_SKULL)

        # 3. Concatenate along token axis to feed the Perceiver pool.
        K = torch.cat([k_A, k_B], dim=1)                  # (B, N_A + N_B, d)
        V = torch.cat([v_A, v_B], dim=1)                  # (B, N_A + N_B, d)                         # (B, N_A, d)

        # 4. Pool to fixed latent.
        H = self.latent_queries.unsqueeze(0).expand(B, -1, -1)  # (B, M, d)
        z0 = self.pool(q=H, k_in=K, v_in=V)                     # (B, M, d)
        return z0