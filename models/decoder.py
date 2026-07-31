"""
TFUS Neural Operator — Decoder.

Decoder pipeline:

    z^L (B, M, d)                                ← latent processor output
    focal_coords (B, N_A, 3) in [-1, 1]^3        ← reuse encoder's field_coords
    grid_shape (Gx, Gy, Gz)                      ← focal patch grid

    coord_pe_q = MLP_coord_dec( SinusoidalPE3D(focal_coords) )  (B, N_A, d)

    Cross-attention:
        Q = coord_pe_q
        K = z^L
        V = z^L
        -> patch_tokens (B, N_A, d)

    Unembed: Linear(d -> P^3) per token -> (B, N_A, P, P, P)
    Fold: reshape + permute to (B, Gx*P, Gy*P, Gz*P) = (B, H, W, D)
    Output: p_max ∈ R^(B, H, W, D)

Design notes:
    - Decoder owns its own SinusoidalPE3D and coord MLP, distinct from the
      encoder's. They could be shared (LNO option, eq. (4)/(5) with W1=W2),
      and we keep that as a future ablation option. Separate is the safer
      default — encoder coord-mlp learns "how to read at a position",
      decoder coord-mlp learns "how to query at a position".
    - The cross-attention here does NOT decouple K from V (both are z^L).
      The LNO-style decoupling matters when V carries position-dependent
      values (encoder case); on the decoder side, z^L is already a
      content representation and K=V is standard.
    - The patch grid is required at construction time. Token ordering
      MUST match the encoder's flatten convention (x-fastest), so coords
      passed to the decoder must follow that order.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import SinusoidalPE3D


# ---------------------------------------------------------------------------
# Plain cross-attention block (K and V from the same source)
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """
    Pre-norm cross-attention with K and V drawn from the same tensor.

    Forward:
        q_n  = LN(q)
        kv_n = LN(kv)
        q   <- q + Attn(q_n, kv_n, kv_n)
        q   <- q + MLP(LN(q))
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_kv = nn.Linear(dim, 2 * dim, bias=True)
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

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: (B, Nq, d) | kv: (B, Nkv, d)
        B, Nq, d = q.shape
        Nkv = kv.size(1)
        H, Hd = self.num_heads, self.head_dim

        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)

        qh = self.to_q(qn).view(B, Nq, H, Hd).transpose(1, 2)
        kh, vh = self.to_kv(kvn).chunk(2, dim=-1)
        kh = kh.view(B, Nkv, H, Hd).transpose(1, 2)
        vh = vh.view(B, Nkv, H, Hd).transpose(1, 2)

        out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, Nq, d)
        out = self.proj_drop(self.proj(out))

        q = q + out
        q = q + self.mlp(self.norm_mlp(q))
        return q


# ---------------------------------------------------------------------------
# Patch unembedding and folding back to a volume
# ---------------------------------------------------------------------------

def fold_patches_to_volume(
    patch_voxels: torch.Tensor,         # (B, Np, P, P, P)
    grid_shape: tuple[int, int, int],   # (Gx, Gy, Gz)
) -> torch.Tensor:
    """
    Inverse of (Conv3d, k=s=P).flatten(2).transpose(1, 2) used by the encoder.

    The encoder flatten convention is x-fastest, i.e., token index
        idx = ix + Gx * (iy + Gy * iz)
    So Conv3d output (B, d, Gx, Gy, Gz) is read with ix varying fastest
    when we call .flatten(2). We must invert that index mapping here.

    Args:
        patch_voxels: per-patch local voxel block, shape (B, Gx*Gy*Gz, P, P, P).
        grid_shape:   (Gx, Gy, Gz).
    Returns:
        volume: (B, Gx*P, Gy*P, Gz*P) — single channel output (squeezed).
    """
    B, Np, P, _, _ = patch_voxels.shape
    Gx, Gy, Gz = grid_shape
    if Np != Gx * Gy * Gz:
        raise ValueError(f"patch count {Np} != Gx*Gy*Gz = {Gx*Gy*Gz}")

    # (B, Np, P, P, P) -> (B, Gx, Gy, Gz, P, P, P)   [matches x-fastest token order]
    # ... but reshape order in numpy/torch is row-major (last axis fastest).
    # To recover x-fastest tokens into (Gx, Gy, Gz), we reshape to (Gz, Gy, Gx)
    # then permute. Concretely, in the encoder we did:
    #     conv_out: (B, d, Gx, Gy, Gz)
    #     .flatten(2)        : (B, d, Gx*Gy*Gz) with memory order (iz, iy, ix)
    #                          since contiguous default has the LAST axis
    #                          varying fastest in memory.
    #     .transpose(1, 2)   : (B, Gx*Gy*Gz, d)
    # That means token at flat index t corresponds to
    #     ix = t // (Gy * Gz)  ?  NO — careful.
    # Let's reason rigorously using torch indexing semantics.
    #
    # For a contiguous tensor of shape (B, d, Gx, Gy, Gz), the linear memory
    # index iterates with Gz varying FASTEST, then Gy, then Gx, then d, then B.
    # So .flatten(2) which flattens axes [Gx, Gy, Gz] produces, at position t:
    #     iz = t %  Gz
    #     iy = (t // Gz) % Gy
    #     ix = t // (Gy * Gz)
    # i.e., z is fastest in TOKEN ORDER (not x). Our earlier comment about
    # "x-fastest" was incorrect.
    #
    # To invert, reshape (B, Np, ...) -> (B, Gx, Gy, Gz, P, P, P).
    voxels = patch_voxels.view(B, Gx, Gy, Gz, P, P, P)

    # Within each patch, we used Conv3d(k=s=P) on input (H, W, D). The kernel
    # iterates over (P_x, P_y, P_z) in the same order as the spatial axes.
    # So patch_voxels[..., px, py, pz] sits at absolute voxel
    #     (ix*P + px, iy*P + py, iz*P + pz).
    # We assemble the volume by interleaving:
    #   final_x = ix * P + px,  final_y = iy*P + py,  final_z = iz*P + pz
    # Achieve this via permute + reshape:
    #     current axes: (B, Gx, Gy, Gz, Px, Py, Pz)
    #     target axes : (B, Gx, Px, Gy, Py, Gz, Pz)
    voxels = voxels.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    volume = voxels.view(B, Gx * P, Gy * P, Gz * P)
    return volume

def _build_intra_patch_coords(P: int, dtype=torch.float32) -> torch.Tensor:
        """
        Local coords of P^3 voxels inside one patch, in [-1, 1]^3.
        Order matches the row-major z-fastest convention used by fold_patches_to_volume:
            local index l = lz + P * (ly + P * lx),  lz fastest.
        Returns (P^3, 3).
        """
        # P samples in [-1+1/P, 1-1/P]: voxel centers within the patch.
        s = (torch.arange(P, dtype=dtype) + 0.5) / P * 2 - 1   # (P,)
        LX, LY, LZ = torch.meshgrid(s, s, s, indexing='ij')     # (P, P, P) each
        coords = torch.stack([LX, LY, LZ], dim=-1).reshape(-1, 3)  # (P^3, 3)
        return coords

# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class TFUSDecoder(nn.Module):
    """
    Map latent z^L (B, M, d) + focal patch coords -> output volume (B, H, W, D).

    Args:
        embed_dim:     d. Must match encoder/processor.
        patch_size:    P. Must match encoder.
        grid_shape:    (Gx, Gy, Gz) patch grid of the focal ROI.
                       For a 112^3 input with P=4 this is (28, 28, 28).
        num_heads:     attention heads for the decoder cross-attention.
        mlp_ratio:     MLP expansion in the cross-attention block.
        dropout:       dropout in attention/MLP.
        coord_pe_bands: number of sinusoidal bands per axis. None -> d//6.

    Forward inputs:
        z          : (B, M, d) — latent processor output.
        focal_coords: (B, N_A, 3) — patch-center coords in normalized frame,
                      in the same token order as the encoder produced
                      (z-fastest within (Gx, Gy, Gz)).

    Returns:
        p_max_pred: (B, Gx*P, Gy*P, Gz*P) — predicted output field.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        patch_size: int = 4,
        grid_shape: tuple[int, int, int] = (28, 28, 28),
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        coord_pe_bands: int | None = None,
        intra_pe_bands: int | None = None,
    ):
        super().__init__()
        if embed_dim % 6 != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by 6")
        Gx, Gy, Gz = grid_shape
        self.grid_shape = grid_shape
        self.patch_size = patch_size
        self.num_patches = Gx * Gy * Gz

        # Decoder owns its own coord PE + MLP, distinct from the encoder's.
        self.coord_pe = SinusoidalPE3D(embed_dim, num_bands=coord_pe_bands)
        self.coord_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Cross-attention: Q from focal coords, K=V from z^L.
        self.xattn = CrossAttentionBlock(
            dim=embed_dim, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout,
        )

        # Unembed each patch token to P^3 voxel values (scalar output channel).
        # self.unembed = nn.Linear(embed_dim, patch_size ** 3, bias=True)

        # --- intra-patch coord-query head (replaces Linear(d, P^3)) ---
        # local coord -> PE -> MLP -> d-dim "basis" vector.
        # Then voxel value = <patch_token, basis(x_local)>.
        self.intra_pe = SinusoidalPE3D(embed_dim, num_bands=intra_pe_bands)
        self.intra_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Pre-compute intra-patch local coords as buffer.
        local = _build_intra_patch_coords(patch_size)         # (P^3, 3)
        self.register_buffer("intra_local_coords", local, persistent=False)

    def forward(self, z: torch.Tensor, focal_coords: torch.Tensor) -> torch.Tensor:
        """
        z           : (B, M, d)
        focal_coords: (B, N_A, 3) with N_A == Gx*Gy*Gz, in normalized [-1, 1]^3.

        Returns:
            (B, Gx*P, Gy*P, Gz*P)
        """
        B, M, d = z.shape
        Np = focal_coords.size(1)
        if Np != self.num_patches:
            raise ValueError(
                f"focal_coords has {Np} tokens but grid_shape implies "
                f"{self.num_patches}"
            )

        # Build queries from focal patch coordinates.
        q_pe = self.coord_pe(focal_coords)                  # (B, Np, d)
        q = self.coord_mlp(q_pe)                            # (B, Np, d)

        # Cross-attend into the latent.
        patch_tokens = self.xattn(q=q, kv=z)                # (B, Np, d)

        # Unembed each token to P^3 voxel values.
        P = self.patch_size
        # patch_voxels = self.unembed(patch_tokens)           # (B, Np, P^3)
        # patch_voxels = patch_voxels.view(B, Np, P, P, P)    # (B, Np, P, P, P)

        # # Fold back to the volume.
        # volume = fold_patches_to_volume(patch_voxels, self.grid_shape)
        # return volume                                       # (B, Gx*P, Gy*P, Gz*P)

        # local coords -> basis vectors of shape (P^3, d).
        # intra_pe expects (B, N, 3); we add a singleton batch dim and squeeze back.
        local = self.intra_local_coords.unsqueeze(0)          # (1, P^3, 3)
        basis = self.intra_mlp(self.intra_pe(local))          # (1, P^3, d)
        basis = basis.squeeze(0)                              # (P^3, d)

        # voxel value = inner product between patch_token and basis row.
        # patch_tokens: (B, Np, d). basis: (P^3, d).
        # out: (B, Np, P^3)
        patch_voxels = torch.einsum("bnd,pd->bnp", patch_tokens, basis)

        # Reshape to (B, Np, P, P, P) using the same row-major (z-fastest within patch)
        # order assumed by fold_patches_to_volume.
        patch_voxels = patch_voxels.view(B, Np, P, P, P)

        # Fold back to volume.
        volume = fold_patches_to_volume(patch_voxels, self.grid_shape)
        return volume