"""
TFUS Neural Operator — Conditioning module.

Handles transducer-related conditioning variables and produces:
  (a) a single global condition vector c of shape (B, d), to be consumed by
      DiT-style block-wise modulation in the latent processor.
  (b) per-block (gamma, beta, alpha) tuples via DiTModulation (one instance
      per latent block).

Conditioning sources currently supported (each can be independently dropped
for ablation via the null-embedding mechanism):
    - frequency : scalar in Hz (4 values used in training: 250/400/500/650 kHz).
    - position  : transducer center (x, y, z) in mm relative to volume origin.
    - angle     : transducer focal-axis direction as a unit vector (u, v, w).
                  Assumes transducer is symmetric around its axis (rotation
                  about the axis is ignored). If you use Euler angles or
                  quaternions instead, swap the angle embedder accordingly.

Ablation interface:
    ConditioningModule.forward(..., active=("freq", "pos", "angle"))
    -> sources not in `active` are replaced by their learned null vector.
    During training, individual sources are also dropped at random with
    probability `dropout_p` (independent per sample, per source) so the model
    learns to handle null on each axis.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fourier feature encoding
# ---------------------------------------------------------------------------

class FourierFeatures(nn.Module):
    """
    Multi-band sinusoidal feature encoding.

    For input s ∈ R^n (per-sample), output:
        [sin(2^0 π s), cos(2^0 π s), ..., sin(2^(K-1) π s), cos(2^(K-1) π s)]
    flattened along the last axis. Output dim = 2 * K * n.

    Inputs to this module must already be normalized to roughly [-1, 1]
    (per channel) so the lowest-frequency band has useful gradients.
    """

    def __init__(self, num_bands: int = 8):
        super().__init__()
        self.num_bands = num_bands
        # Frequencies: pi * 2^k for k = 0..K-1
        freqs = math.pi * (2.0 ** torch.arange(num_bands).float())
        self.register_buffer("freqs", freqs, persistent=False)  # (K,)

    @property
    def out_dim_per_channel(self) -> int:
        return 2 * self.num_bands

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n) -> (B, n * 2 * K)
        # angles: (B, n, K)
        angles = x.unsqueeze(-1) * self.freqs.view(1, 1, -1)
        out = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, n, 2K)
        return out.flatten(1)


# ---------------------------------------------------------------------------
# Per-source embedder
# ---------------------------------------------------------------------------

class SourceEmbedder(nn.Module):
    """
    Generic embedder: raw scalar/vector -> normalized -> Fourier -> MLP -> R^d.

    Normalization is handled externally (we just multiply by an internal scale
    and bias if provided); each call takes already-normalized inputs in the
    expected range. The caller (ConditioningModule) is responsible for taking
    raw physical values and shaping them to roughly [-1, 1].
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_fourier_bands: int = 8,
        hidden_ratio: float = 2.0,
    ):
        super().__init__()
        self.fourier = FourierFeatures(num_bands=num_fourier_bands)
        feat_dim = in_channels * self.fourier.out_dim_per_channel
        hidden = int(embed_dim * hidden_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels) normalized to ~[-1, 1]
        return self.mlp(self.fourier(x))


# ---------------------------------------------------------------------------
# Conditioning module
# ---------------------------------------------------------------------------

class ConditioningModule(nn.Module):
    """
    Aggregates per-source condition embeddings into a single condition vector
    c of shape (B, d).

    Args:
        embed_dim:      d, matched to latent token width.
        domain_size_mm: (Lx, Ly, Lz) full simulation domain size in mm.
                        Used to normalize transducer position to [-1, 1]^3.
        sources:        tuple of source names to use. Subset of
                        ('freq', 'pos', 'angle'). Embedders are created only
                        for declared sources.
        freq_range_hz:  (f_min, f_max) for frequency normalization. Ignored
                        if 'freq' not in sources.
        num_fourier_bands: K for Fourier feature encoding.
        dropout_p:      per-source CFG-style null dropout probability.

    Inputs to forward (all batched):
        freq:   (B, 1) or (B,) Hz. None => use null (must be None if not declared).
        pos:    (B, 3) mm absolute. None => use null.
        angle:  (B, 3) unit vector. None => use null.
        active: subset of declared sources that ARE active for this forward
                pass. Others are forced to null. None => all declared active.
    """

    SOURCE_DIMS = {"freq": 1, "pos": 3, "angle": 3}

    def __init__(
        self,
        embed_dim: int,
        domain_size_mm: tuple[float, float, float],
        sources: tuple[str, ...] = ("freq", "pos", "angle"),
        freq_range_hz: tuple[float, float] = (200e3, 700e3),
        num_fourier_bands: int = 8,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        for s in sources:
            if s not in self.SOURCE_DIMS:
                raise ValueError(
                    f"unknown source {s!r}; expected from {tuple(self.SOURCE_DIMS)}"
                )
        if len(sources) == 0:
            raise ValueError("at least one conditioning source required")
        self.sources = tuple(sources)

        self.embed_dim = embed_dim
        self.dropout_p = float(dropout_p)

        # Per-source normalization buffers — only registered if needed.
        if "pos" in self.sources:
            domain = torch.tensor(domain_size_mm, dtype=torch.float32)   # (3,)
            self.register_buffer("domain_half_mm",   domain * 0.5, persistent=False)
            self.register_buffer("domain_center_mm", domain * 0.5, persistent=False)
        if "freq" in self.sources:
            self.register_buffer(
                "freq_lo", torch.tensor(freq_range_hz[0], dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "freq_hi", torch.tensor(freq_range_hz[1], dtype=torch.float32),
                persistent=False,
            )

        # Embedders + null vectors only for declared sources.
        self.embedders = nn.ModuleDict({
            name: SourceEmbedder(self.SOURCE_DIMS[name], embed_dim, num_fourier_bands)
            for name in self.sources
        })
        self.null_vectors = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(embed_dim))
            for name in self.sources
        })
        for p in self.null_vectors.values():
            nn.init.trunc_normal_(p, std=0.02)

    # ----- normalization (no change in logic, just call-site protected) -----

    def _normalize_freq(self, freq_hz: torch.Tensor) -> torch.Tensor:
        if freq_hz.ndim == 1:
            freq_hz = freq_hz.unsqueeze(-1)
        center = 0.5 * (self.freq_hi + self.freq_lo)
        half   = 0.5 * (self.freq_hi - self.freq_lo)
        return (freq_hz - center) / half

    def _normalize_pos(self, pos_mm: torch.Tensor) -> torch.Tensor:
        return (pos_mm - self.domain_center_mm) / self.domain_half_mm

    def _normalize_angle(self, angle: torch.Tensor) -> torch.Tensor:
        norm = angle.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return angle / norm

    # ----- dropout mask -----

    def _sample_dropout_mask(self, B: int, device: torch.device) -> dict[str, torch.Tensor]:
        if not self.training or self.dropout_p <= 0.0:
            return {n: torch.ones(B, dtype=torch.bool, device=device) for n in self.sources}
        return {n: (torch.rand(B, device=device) >= self.dropout_p) for n in self.sources}

    # ----- forward -----

    def forward(
        self,
        freq:  torch.Tensor | None = None,
        pos:   torch.Tensor | None = None,
        angle: torch.Tensor | None = None,
        active: tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        raw_inputs = {"freq": freq, "pos": pos, "angle": angle}

        # Infer batch from any non-None tensor.
        ref = next((v for v in raw_inputs.values() if v is not None), None)
        if ref is None:
            raise ValueError("at least one of (freq, pos, angle) must be provided")
        B, device, dtype = ref.size(0), ref.device, ref.dtype

        # Validate active.
        if active is None:
            active_set = set(self.sources)
        else:
            for name in active:
                if name not in self.sources:
                    raise ValueError(
                        f"active source {name!r} not in declared sources {self.sources}"
                    )
            active_set = set(active)

        train_mask = self._sample_dropout_mask(B, device)
        normalizers = {
            "freq":  self._normalize_freq,
            "pos":   self._normalize_pos,
            "angle": self._normalize_angle,
        }

        c = torch.zeros(B, self.embed_dim, device=device, dtype=dtype)
        for name in self.sources:
            null = self.null_vectors[name].unsqueeze(0).expand(B, -1)
            x_raw = raw_inputs[name]

            if x_raw is None:
                # declared but not provided this step -> null
                c = c + null
                continue

            e = self.embedders[name](normalizers[name](x_raw))
            use   = train_mask[name] & (name in active_set)         # (B,) bool
            use_f = use.to(e.dtype).unsqueeze(-1)                   # (B, 1)
            chosen = use_f * e + (1.0 - use_f) * null               # (B, d)
            c = c + chosen
        return c


# ---------------------------------------------------------------------------
# DiT-style block modulation
# ---------------------------------------------------------------------------

class DiTModulation(nn.Module):
    """
    Per-block adaptive layernorm modulation, DiT-style.

    Given c ∈ R^(B, d), produce (gamma_attn, beta_attn, alpha_attn,
    gamma_mlp, beta_mlp, alpha_mlp), each of shape (B, d).

    The final linear layer is initialized to zero so the block starts as
    identity (alpha=0 -> residual only, gamma=0 -> no scale change, beta=0 ->
    no shift). The conditioning then ramps up as training progresses.

    Usage inside a block:
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.mod(c).chunk(6, dim=-1)
        x = x + alpha1.unsqueeze(1) * attn(modulate(ln1(x), gamma1, beta1))
        x = x + alpha2.unsqueeze(1) * mlp(modulate(ln2(x), gamma2, beta2))
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim, bias=True),
        )
        # Zero-init final linear so block is identity at init.
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        # c: (B, d) -> (B, 6d), to be split by the consumer.
        return self.proj(c)


def modulate(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Apply DiT-style scale-shift to a token sequence.
    x:     (B, N, d)
    gamma: (B, d)   -- broadcast over tokens
    beta:  (B, d)
    Returns (1 + gamma) * x + beta   (init gamma=beta=0 => identity).
    """
    return x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)