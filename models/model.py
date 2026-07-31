"""
TFUS Neural Operator — full model.

Composition:

    ff_vol   (B, 1, H_A, W_A, D_A)            ┐
    ff_coords(B, N_A, 3) in [-1,1]^3          │
    skull_vol   (B, 1, H_B, W_B, D_B)            │  encoder
    skull_coords(B, N_B, 3) in [-1,1]^3          ├──> z^0 (B, M, d)
                                                  ┘
    freq (B, 1) [Hz]                              ┐
    pos  (B, 3) [mm in focal-centric frame]       │  conditioning
    angle(B, 3) [unit vector, focal axis dir]     ├──> c   (B, d)
    active=("freq","pos","angle"|...) for abl.    ┘

    z^0, c                                        ┐  processor (DiT-conditioned
                                                  ├   L transformer blocks)
                                                  └──> z^L (B, M, d)

    z^L, ff_coords                             ┐  decoder
                                                  ├──> p_max (B, H_A, W_A, D_A)
                                                  ┘

The model treats `ff_coords` as both encoder input (to build the
field-modality K, V) and decoder input (as the query for output patches).
Re-using the same coords guarantees input grid = output grid by
construction; no separate query positions are required.

Sanity:
    - encoder/decoder SinusoidalPE3D + coord MLP are NOT shared (per design).
    - conditioning module performs CFG-style per-source dropout in train mode.
    - latent processor uses DiT modulation; pass c=None to bypass conditioning
      entirely (useful for the conditioning-ablation baseline).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import TFUSDualEncoder
from .conditioning import ConditioningModule
from .latent_processor import LatentProcessor
from .decoder import TFUSDecoder


class TFUSOperator(nn.Module):
    """
    Full transcranial focused-ultrasound neural operator.

    Args:
        embed_dim:      d, shared across all modules. Must be div by 6 and by num_heads.
        patch_size:     P, voxel-side of each patch. Same value used by encoder and
                        decoder; assumes both ROIs share the same voxel spacing.
        focal_grid:     (Gx, Gy, Gz) patch grid of the focal ROI (output grid).
                        For a 112^3 input with P=4, this is (28, 28, 28).
        num_latents:    M, fixed latent token count after Perceiver pool.
        num_heads:      attention heads for encoder pool, decoder cross-attn,
                        and latent processor blocks.
        depth:          L, number of latent processor blocks.
        mlp_ratio:      MLP expansion in all blocks.
        dropout:        attention/MLP dropout.
        use_dit:        if True, latent processor is DiT-conditioned via c.
                        If False, the processor ignores conditioning and the
                        model behaves as an unconditional (in the
                        transducer-parameter sense) operator.
        freq_range_hz:  range for frequency normalization (passed to
                        ConditioningModule).
        pos_scale_mm:   scale for transducer-position normalization.
        cond_dropout:   per-source null dropout probability for training-time
                        CFG-style conditioning robustness.
        coord_pe_bands: passed to encoder/decoder PE modules (defaults to d/6).
    """

    def __init__(
        self,
        embed_dim: int = 384,
        patch_size: int = 4,
        focal_grid: tuple[int, int, int] = (28, 28, 28),
        num_latents: int = 512,
        num_heads: int = 6,
        depth: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_dit: bool = True,
        # conditioning
        cond_sources: tuple[str, ...] = ("freq", "pos", "angle"),
        freq_range_hz: tuple[float, float] = (200e3, 700e3),
        domain_size_mm: tuple[float, float, float] = (225.0, 225.0, 150.0),
        cond_dropout: float = 0.1,
        # PE
        coord_pe_bands: int | None = None,
        # CT stem
        ct_stem_depth: int = 0,
        ct_stem_channels: int = 64,
    ):
        super().__init__()

        self.encoder = TFUSDualEncoder(
            embed_dim=embed_dim,
            patch_size=patch_size,
            num_latents=num_latents,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            coord_pe_bands=coord_pe_bands,
            ct_stem_depth=ct_stem_depth,
            ct_stem_channels=ct_stem_channels,
        )

        self.conditioning = ConditioningModule(
            embed_dim=embed_dim,
            domain_size_mm=domain_size_mm,
            sources=cond_sources,
            freq_range_hz=freq_range_hz,
            dropout_p=cond_dropout,
        )

        self.cond_sources = cond_sources

        self.processor = LatentProcessor(
            dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_dit=use_dit,
            final_norm=True,
        )

        self.decoder = TFUSDecoder(
            embed_dim=embed_dim,
            patch_size=patch_size,
            grid_shape=focal_grid,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            coord_pe_bands=coord_pe_bands,
        )

        # self.refine = nn.Sequential(
        #     nn.Conv3d(1, 16, kernel_size=3, padding=1),
        #     nn.GELU(),
        #     nn.Conv3d(16, 1, kernel_size=3, padding=1),
        # )
        # nn.init.zeros_(self.refine[-1].weight)
        # nn.init.zeros_(self.refine[-1].bias)

        self.use_dit = use_dit
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.focal_grid = focal_grid
        self.num_latents = num_latents

    # ------------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------------

    def forward(
        self,
        ff_vol: torch.Tensor,
        ff_coords: torch.Tensor,
        skull_vol: torch.Tensor,
        skull_coords: torch.Tensor,
        freq: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        angle: torch.Tensor | None = None,
        active: tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        """
        ff_vol   : (B, 1, H_A, W_A, D_A)
        ff_coords: (B, N_A, 3) normalized patch-center coords (focal ROI).
        skull_vol   : (B, 1, H_B, W_B, D_B)
        skull_coords: (B, N_B, 3) normalized patch-center coords (skull ROI),
                      in the same normalized frame as ff_coords.
        freq, pos, angle : transducer condition. Required when use_dit=True
                           (or when any condition is meant to be active);
                           must each be a (B, ...) tensor. Pass None when
                           use_dit=False; processor will use c=None.
        active      : tuple of source names that ARE active for this forward.
                      None  -> all active. ()    -> all dropped (null).
                      Used for inference-time ablation.

        Returns:
            p_max_pred: (B, H_A, W_A, D_A)
        """
        # 1. Encode.
        z = self.encoder(ff_vol, ff_coords, skull_vol, skull_coords)   # (B, M, d)

        # 2. Build condition vector (or pass-through None).
        if self.use_dit:
            c = self.conditioning(freq=freq, pos=pos, angle=angle, active=active)
        else:
            c = None

        # 3. Propagate in latent space.
        z = self.processor(z, c=c)                                            # (B, M, d)

        # 4. Decode back to volume using focal coords as queries.
        p_max = self.decoder(z, ff_coords)                                 # (B, H_A, W_A, D_A)

        # p_max = p_max.unsqueeze(1)
        # p_max = p_max + self.refine(p_max)
        # p_max = p_max.squeeze(1)

        return p_max

    # ------------------------------------------------------------------------
    # Convenience: module breakdown
    # ------------------------------------------------------------------------

    @torch.no_grad()
    def param_breakdown(self) -> dict[str, float]:
        """Returns a dict of submodule -> M-params for inspection."""
        return {
            "encoder":      sum(p.numel() for p in self.encoder.parameters()) / 1e6,
            "conditioning": sum(p.numel() for p in self.conditioning.parameters()) / 1e6,
            "processor":    sum(p.numel() for p in self.processor.parameters()) / 1e6,
            "decoder":      sum(p.numel() for p in self.decoder.parameters()) / 1e6,
            "total":        sum(p.numel() for p in self.parameters()) / 1e6,
        }


# ---------------------------------------------------------------------------
# Helper: build z-fastest patch-center coords for a uniform grid
# ---------------------------------------------------------------------------

def make_uniform_patch_coords(
    grid_shape: tuple[int, int, int],
    bbox_min: tuple[float, float, float] = (-1.0, -1.0, -1.0),
    bbox_max: tuple[float, float, float] = ( 1.0,  1.0,  1.0),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build patch-center coordinates for a uniform grid in [bbox_min, bbox_max],
    in the z-fastest token order expected by the encoder/decoder.

    Args:
        grid_shape: (Gx, Gy, Gz).
        bbox_min:   per-axis minimum of the bounding box (normalized frame).
        bbox_max:   per-axis maximum of the bounding box (normalized frame).
        device, dtype: standard.

    Returns:
        (Gx * Gy * Gz, 3) tensor of coords.
    """
    Gx, Gy, Gz = grid_shape
    xmin, ymin, zmin = bbox_min
    xmax, ymax, zmax = bbox_max
    x = torch.linspace(xmin, xmax, Gx, device=device, dtype=dtype)
    y = torch.linspace(ymin, ymax, Gy, device=device, dtype=dtype)
    z = torch.linspace(zmin, zmax, Gz, device=device, dtype=dtype)
    grid = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1)   # (Gx, Gy, Gz, 3)
    return grid.view(-1, 3)