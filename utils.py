"""
TFUS Neural Operator — training/evaluation utilities.

This module is intentionally `train_one_epoch` / `validate_one_epoch` /
metrics + helpers. The outer training script wires these together with
config, data loaders, optimizer, logging, checkpointing.

Contents
--------
- build_criterion(name)              : returns a callable loss(pred, target).
- build_lr_scheduler(opt, args, n)   : LambdaLR with warmup + cosine, or None.
- EarlyStopping(patience)            : updates best metric, signals stop.
- train_one_epoch(...)               : one pass over the training loader.
- validate_one_epoch(...)            : one pass over a validation loader,
                                       reporting both loss and clinical metrics.
- compute_metrics(pred, target, voxel_mm)
                                     : per-sample {dice, peak_dist, peak_diff}.

Metric definitions (all per-sample, mean over batch):
  - Dice over FWHM masks.  M(x) := |x| > 0.5 * |x|.max().
                           Dice = 2 |M_pred ∩ M_target| / (|M_pred| + |M_target|).
  - Peak distance.         arg of |x|.max(), converted to mm via voxel_mm,
                           then Euclidean distance.
  - Peak value % error.    100 * |peak_pred - peak_target| / |peak_target|,
                           using max-of-absolute (signed peak preserved).

All metrics work on volumes of arbitrary 3D shape and tolerate empty masks /
zero peaks gracefully (return 0 dice or 0 distance / NaN-safe denominators).
"""

from __future__ import annotations

import math
from time import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pathlib import Path
import matplotlib.pyplot as plt


# =========================================================================
# Misc helpers
# =========================================================================

def format_eta(seconds: float) -> str:
    """Human-readable duration. Returns '--' for non-finite/negative input."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--"
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}d{h:02d}h{m:02d}m"
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# =========================================================================
# Criterion
# =========================================================================

class _SignLogMSE(nn.Module):
    """MSE in sign-log space.  L = mean( (sgn(p)·log(1+|p|) - sgn(t)·log(1+|t|))^2 )."""
    def __init__(self, eps: float = 1.0):
        super().__init__()
        self.eps = float(eps)

    @staticmethod
    def _signlog(x: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.sign(x) * torch.log1p(x.abs() / eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self._signlog(pred, self.eps), self._signlog(target, self.eps))


def _build_base_recon(name: str) -> nn.Module:
    """Voxelwise reconstruction loss (the 'recon' term)."""
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    if name == "huber":
        return nn.HuberLoss(delta=1.0)
    if name == "log_mse":
        return _SignLogMSE()
    raise ValueError(f"unknown criterion {name!r}; expected mse/l1/huber/log_mse")


# =========================================================================
# Soft, differentiable analogs of the three eval metrics
# =========================================================================


def _softmax_weights(absf_flat: torch.Tensor, beta: float) -> torch.Tensor:
    """(B, M) abs-field -> (B, M) softmax weights over the spatial axis."""
    return torch.softmax(beta * absf_flat, dim=1)


def _soft_peak_value(absf_flat: torch.Tensor, beta: float) -> torch.Tensor:
    w = _softmax_weights(absf_flat, beta)
    return (absf_flat * w).sum(dim=1)                         # (B,)


def _soft_centroid(absf_flat: torch.Tensor, grid_flat: torch.Tensor,
                   beta: float) -> torch.Tensor:
    """grid_flat: (3, M) normalized coords. Returns (B, 3) soft-argmax position."""
    w = _softmax_weights(absf_flat, beta)                     # (B, M)
    return w @ grid_flat.t()                                  # (B, 3)


class CompositeCriterion(nn.Module):
    """Weighted combination of {recon, dice, peak_dist, peak_diff} losses.

    Weighting:
      - 'uncertainty' (default for multi-term): Kendall et al. 2018
        homoscedastic uncertainty.
            total = sum_i [ exp(-s_i) * L_i + s_i ],  s_i = log(sigma_i^2).
        The +s_i term penalizes sigma_i -> inf, so a term's weight CANNOT be
        driven to 0 to cheat (which a plain learnable w_i would do, collapsing
        onto the easiest term). s_i are nn.Parameters -> they MUST be added to
        the optimizer (build_optimizer handles this).
      - 'fixed': total = sum_i w_i * L_i with constant w_i.

    Each L_i is roughly O(1): soft-dice in [0,1]; peak_dist is a squared
    distance in NORMALIZED [0,1]^3 coords (not mm) so it is O(1); peak_diff is
    a relative error.
    """

    TERMS = ("recon", "dice", "peak_dist", "peak_diff")

    def __init__(self, base_recon: nn.Module, terms, weighting: str = "uncertainty",
                 fixed_weights=None, peak_beta: float = 30.0, mask_beta: float = 10.0,
                 eps: float = 1e-6):
        super().__init__()
        terms = [t for t in terms]
        for t in terms:
            if t not in self.TERMS:
                raise ValueError(f"unknown loss term {t!r}; expected subset of {self.TERMS}")
        if not terms:
            raise ValueError("at least one loss term required")
        self.terms = terms
        self.base_recon = base_recon
        self.weighting = weighting
        self.peak_beta = float(peak_beta)
        self.mask_beta = float(mask_beta)
        self.eps = float(eps)

        if weighting == "uncertainty":
            self.log_vars = nn.Parameter(torch.zeros(len(terms)))
        elif weighting == "fixed":
            w = fixed_weights if fixed_weights is not None else [1.0] * len(terms)
            if len(w) != len(terms):
                raise ValueError(f"fixed_weights len {len(w)} != #terms {len(terms)}")
            self.register_buffer("fixed_w", torch.tensor(w, dtype=torch.float32))
        else:
            raise ValueError(f"weighting must be 'uncertainty' or 'fixed', got {weighting!r}")

        self._grid_cache: dict = {}

    def _grid_flat(self, shape, device, dtype):
        """(3, M) normalized [0,1] coordinate grid, z-fastest flatten."""
        key = (shape, device, dtype)
        g = self._grid_cache.get(key)
        if g is None:
            Nx, Ny, Nz = shape
            ax = torch.linspace(0, 1, Nx, device=device, dtype=dtype)
            ay = torch.linspace(0, 1, Ny, device=device, dtype=dtype)
            az = torch.linspace(0, 1, Nz, device=device, dtype=dtype)
            gx, gy, gz = torch.meshgrid(ax, ay, az, indexing="ij")
            g = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=0)
            self._grid_cache[key] = g
        return g

    def _compute_terms(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        out = {}
        if "recon" in self.terms:
            out["recon"] = self.base_recon(pred, target)

        if not any(t in self.terms for t in ("dice", "peak_dist", "peak_diff")):
            return out

        B = pred.size(0)
        ap = pred.abs().reshape(B, -1)
        at = target.abs().reshape(B, -1)

        if "dice" in self.terms:
            vp = _soft_peak_value(ap, self.peak_beta).detach()   # threshold ref, no grad through max
            vt = _soft_peak_value(at, self.peak_beta)
            mp = torch.sigmoid(self.mask_beta * (ap - 0.5 * vp.unsqueeze(1)))
            mt = torch.sigmoid(self.mask_beta * (at - 0.5 * vt.unsqueeze(1)))
            inter = (mp * mt).sum(dim=1)
            denom = mp.sum(dim=1) + mt.sum(dim=1)
            dice = (2 * inter + self.eps) / (denom + self.eps)
            out["dice"] = (1.0 - dice).mean()

        if "peak_dist" in self.terms:
            grid = self._grid_flat(tuple(pred.shape[1:]), pred.device, pred.dtype)
            cp = _soft_centroid(ap, grid, self.peak_beta)        # (B,3) normalized
            ct = _soft_centroid(at, grid, self.peak_beta)
            out["peak_dist"] = ((cp - ct) ** 2).sum(dim=1).mean()  # squared, O(1)

        if "peak_diff" in self.terms:
            vp = _soft_peak_value(ap, self.peak_beta)
            vt = _soft_peak_value(at, self.peak_beta)
            out["peak_diff"] = ((vp - vt).abs() / (vt + self.eps)).mean()

        return out

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        comp = self._compute_terms(pred, target)
        losses = torch.stack([comp[t] for t in self.terms])      # (T,)
        if self.weighting == "uncertainty":
            return (torch.exp(-self.log_vars) * losses + self.log_vars).sum()
        return (self.fixed_w * losses).sum()


def build_criterion(args) -> nn.Module:
    """Build the (possibly composite) training criterion from config.

    Backward compatible: with default --loss_terms ['recon'] this reduces to the
    single reconstruction loss selected by --criterion.
    """
    base = _build_base_recon(args.criterion)
    terms = list(getattr(args, "loss_terms", ["recon"]))
    default_w = "fixed" if terms == ["recon"] else "uncertainty"
    weighting = getattr(args, "loss_weighting", default_w) or default_w
    return CompositeCriterion(
        base_recon=base,
        terms=terms,
        weighting=weighting,
        fixed_weights=getattr(args, "loss_fixed_weights", None),
        peak_beta=getattr(args, "peak_beta", 30.0),
        mask_beta=getattr(args, "mask_beta", 10.0),
    )


# =========================================================================
# LR scheduling
# =========================================================================

def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    args,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    """
    Linear warmup → cosine decay, or None.

    Returns a LambdaLR; caller must call `scheduler.step()` every
    training step (NOT every epoch).
    """
    if not getattr(args, "lr_schedule", False):
        return None

    warmup_steps = max(1, int(args.warmup_epochs * steps_per_epoch))
    total_steps  = max(warmup_steps + 1, int(args.num_epochs * steps_per_epoch))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / float(warmup_steps)
        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =========================================================================
# Early stopping
# =========================================================================

@dataclass
class EarlyStopping:
    """
    Track the best validation metric (lower-is-better by default) and signal
    when to stop. Use `.step(value)` once per epoch (after validation);
    returns True if training should stop.

    Attributes set after each step:
        best:        best value seen so far.
        best_epoch:  epoch index of best value (0-based, by external counter).
        counter:     epochs without improvement.
        is_improved: True iff the most recent step improved on `best`.
    """
    patience: int
    mode: str = "min"           # "min" or "max"
    min_delta: float = 0.0

    best: float = float("inf")
    best_epoch: int = -1
    counter: int = 0
    is_improved: bool = False

    def __post_init__(self):
        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode!r}")
        self.best = float("inf") if self.mode == "min" else float("-inf")

    def _is_better(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float, epoch: int) -> bool:
        """Update state with a new metric value. Returns True if should stop."""
        if self._is_better(value):
            self.best = float(value)
            self.best_epoch = int(epoch)
            self.counter = 0
            self.is_improved = True
        else:
            self.counter += 1
            self.is_improved = False
        return self.counter >= self.patience


# =========================================================================
# Metrics
# =========================================================================

def _peak_voxel_coords(vol: torch.Tensor) -> torch.Tensor:
    """
    Locate argmax of |vol| and return its (x, y, z) voxel indices as a
    length-3 float tensor on the same device as `vol`.

    `vol` must be a single 3D volume (Nx, Ny, Nz).
    """
    flat = vol.abs().reshape(-1)
    idx = int(torch.argmax(flat).item())
    Nx, Ny, Nz = vol.shape
    # row-major (z fastest, then y, then x) per our convention.
    iz = idx % Nz
    iy = (idx // Nz) % Ny
    ix = idx // (Ny * Nz)
    return torch.tensor([ix, iy, iz], dtype=torch.float32, device=vol.device)


def _peak_signed_value(vol: torch.Tensor) -> torch.Tensor:
    """Signed value at argmax(|vol|). Returns 0-D tensor."""
    flat = vol.reshape(-1)
    idx = int(torch.argmax(flat.abs()).item())
    return flat[idx]


def fwhm_mask(vol: torch.Tensor) -> torch.Tensor:
    """
    Full-width-half-maximum mask for a single 3D volume:
        M(x) = |vol(x)| >= 0.5 * |vol|.max()

    Uses absolute value so the definition is well-posed for signed fields.
    If the volume is identically zero, returns an all-False mask.
    """
    abs_max = vol.abs().max()
    if not torch.isfinite(abs_max) or abs_max <= 0:
        return torch.zeros_like(vol, dtype=torch.bool)
    return vol.abs() >= 0.5 * abs_max


def dice_score(mask_a: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
    """
    Dice = 2 |A ∩ B| / (|A| + |B|).  Both inputs bool tensors of same shape.
    Returns 0-D tensor. Returns 1.0 when both masks are empty (convention).
    """
    a = mask_a.to(torch.float32)
    b = mask_b.to(torch.float32)
    inter = (a * b).sum()
    denom = a.sum() + b.sum()
    if denom.item() == 0:
        return torch.tensor(1.0, device=mask_a.device)
    return 2.0 * inter / denom


@torch.no_grad()
def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    voxel_mm: float = 0.5,
) -> dict[str, torch.Tensor]:
    """
    Per-sample metrics, averaged over the batch.

    Args:
        pred, target : (B, Nx, Ny, Nz) tensors. Float dtype.
        voxel_mm     : isotropic voxel spacing for distance unit conversion.

    Returns dict with 0-D tensors (all on `pred`'s device):
        dice         : mean Dice over FWHM masks.
        peak_dist    : mean Euclidean distance between peak locations (mm).
        peak_diff     : mean relative peak-value error in percent.

    Peak value error: 100 * |peak_pred - peak_target| / |peak_target|.
    If |peak_target| == 0, that sample's percent error is treated as 0 to
    avoid divide-by-zero; in practice a real target should never have a
    zero peak, so this affects only degenerate cases.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} != target {tuple(target.shape)}")
    if pred.ndim != 4:
        raise ValueError(f"expected (B, Nx, Ny, Nz), got {tuple(pred.shape)}")

    B = pred.size(0)
    device = pred.device

    dice_sum = torch.zeros((), device=device)
    dist_sum = torch.zeros((), device=device)
    pct_sum  = torch.zeros((), device=device)

    for b in range(B):
        p = pred[b]
        t = target[b]

        # Dice on FWHM masks.
        m_p = fwhm_mask(p)
        m_t = fwhm_mask(t)
        dice_sum = dice_sum + dice_score(m_p, m_t)

        # Peak distance (mm).
        coord_p = _peak_voxel_coords(p)
        coord_t = _peak_voxel_coords(t)
        dist_sum = dist_sum + ((coord_p - coord_t) * voxel_mm).norm()

        # Peak value relative error (%).
        v_p = _peak_signed_value(p).abs()
        v_t = _peak_signed_value(t).abs()
        if v_t.item() == 0:
            pct_sum = pct_sum + torch.zeros((), device=device)
        else:
            pct_sum = pct_sum + 100.0 * (v_p - v_t).abs() / v_t

    return {
        "dice":      dice_sum / B,
        "peak_dist": dist_sum / B,
        "peak_diff":  pct_sum  / B,
    }


# =========================================================================
# Batch unpacking helper
# =========================================================================

def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor entries of a dataset batch to `device`; leave others as-is."""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _forward_model(model, batch: dict[str, Any]) -> torch.Tensor:
    """
    Wraps model.forward(...) for the standard batch dict from TFUSDataset.
    Pulls freq/pos/angle only if the model is DiT-conditioned (use_dit=True).
    """
    kwargs = dict(
        ff_vol=batch["ff_vol"],
        ff_coords=batch["ff_coords"],
        skull_vol=batch["skull_vol"],
        skull_coords=batch["skull_coords"],
    )
    # Pass conditioning if the model expects it. Easier than introspection:
    # the model raises if use_dit=True and freq/pos/angle are None, so always
    # pass them when present in the batch.
    if "freq" in batch:           kwargs["freq"]  = batch["freq"]
    if "transducer_pos" in batch: kwargs["pos"]   = batch["transducer_pos"]
    if "transducer_angle" in batch: kwargs["angle"] = batch["transducer_angle"]
    return model(**kwargs)


# =========================================================================
# Training / validation loops
# =========================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp_dtype: torch.dtype | None = None,
    grad_clip: float | None = None,
    voxel_mm: float = 0.5,
    log_every: int = 0,
    epoch: int = 0,
    total_epochs: int = 0,
) -> dict[str, float]:
    """
    One pass over `loader`. Returns dict with epoch-average loss AND metrics.

    Args:
        scheduler   : called every optimizer step (None to disable).
        scaler      : torch.cuda.amp.GradScaler for fp16. Pass None for
                      bf16 or fp32.
        amp_dtype   : torch.bfloat16 / torch.float16 / None. Determines
                      autocast dtype. None disables autocast entirely.
        grad_clip   : max norm for gradient clipping. None or <=0 disables.
        voxel_mm    : passed to compute_metrics for peak-distance unit.
        log_every   : if > 0, print a one-line status every N optimizer steps.
        epoch       : epoch index used only for log lines.
        total_epochs: total scheduled epochs. If > 0, the step log also reports
                      an overall training ETA (to the scheduled end; early
                      stopping makes it an upper bound). Intra-epoch ETA is
                      always reported when log_every fires.

    Returns:
        {"train_loss", "dice", "peak_dist", "peak_diff"} epoch averages.

    Notes:
        - Metrics are computed every step (fp32 detached) so progress is
          visible during training. This is cheaper than a second pass over
          the train set, but it does add a small per-step overhead.
        - DiT zero-init means some parameters get zero gradient on the
          first few steps; that is expected and not a bug.
    """
    model.train()
    use_amp = amp_dtype is not None

    n_seen = 0
    loss_sum = 0.0
    dice_sum = 0.0
    dist_sum = 0.0
    diff_sum = 0.0

    n_steps = len(loader)
    ema_step_time: float | None = None     # EMA of per-step wall time (s)
    _t_prev = time()

    for step, batch in enumerate(loader):
        batch = _move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                pred = _forward_model(model, batch)
                loss = criterion(pred, batch["target"])
        else:
            pred = _forward_model(model, batch)
            loss = criterion(pred, batch["target"])

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Metrics in fp32 to avoid low-precision argmax/threshold pathologies.
        pred_f = pred.float().detach()
        tgt_f  = batch["target"].float().detach()
        metrics = compute_metrics(pred_f, tgt_f, voxel_mm=voxel_mm)

        bsz = tgt_f.size(0)
        loss_sum += float(loss.detach().item())          * bsz
        dice_sum += float(metrics["dice"].item())        * bsz
        dist_sum += float(metrics["peak_dist"].item())   * bsz
        diff_sum += float(metrics["peak_diff"].item())   * bsz
        n_seen   += bsz

        # Per-step wall time (data fetch + compute), EMA-smoothed for ETA.
        _now = time()
        _dt = _now - _t_prev
        _t_prev = _now
        ema_step_time = _dt if ema_step_time is None else 0.9 * ema_step_time + 0.1 * _dt

        if log_every and (step + 1) % log_every == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            steps_left_epoch = n_steps - (step + 1)

            steps_left_total = steps_left_epoch + (total_epochs - epoch - 1) * n_steps
            eta_total = ema_step_time * steps_left_total
            eta_str = (f"  ETA~{format_eta(eta_total)}")
            print(f"  [epoch {epoch:3d} step {step+1:5d}/{n_steps}] "
                  f"loss={loss.item():.4f}  lr={lr_now:.2e}{eta_str}")

    n = max(1, n_seen)
    return {
        "train_loss": loss_sum / n,
        "dice":       dice_sum / n,
        "peak_dist":  dist_sum / n,
        "peak_diff":  diff_sum / n,
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    amp_dtype: torch.dtype | None = None,
    voxel_mm: float = 0.5,
) -> dict[str, float]:
    """
    Single validation/test pass.

    Returns dict with epoch-average:
        loss        : same criterion as training (for early stopping).
        dice        : mean FWHM Dice across all samples.
        peak_dist   : mean peak-location Euclidean distance (mm).
        peak_diff    : mean peak-value relative error in percent.
    """
    model.eval()
    use_amp = amp_dtype is not None

    n_seen = 0
    loss_sum = 0.0
    dice_sum = 0.0
    dist_sum = 0.0
    diff_sum = 0.0

    for batch in loader:
        batch = _move_batch_to_device(batch, device)

        if use_amp:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                pred = _forward_model(model, batch)
                loss = criterion(pred, batch["target"])
        else:
            pred = _forward_model(model, batch)
            loss = criterion(pred, batch["target"])

        # Metrics in fp32 to avoid low-precision argmax/threshold pathologies.
        pred_f  = pred.float()
        tgt_f   = batch["target"].float()
        metrics = compute_metrics(pred_f, tgt_f, voxel_mm=voxel_mm)

        bsz = batch["target"].size(0)
        loss_sum += float(loss.item()) * bsz
        dice_sum += float(metrics["dice"].item())      * bsz
        dist_sum += float(metrics["peak_dist"].item()) * bsz
        diff_sum += float(metrics["peak_diff"].item()) * bsz
        n_seen   += bsz

    n = max(1, n_seen)
    return {
        "val_loss":  loss_sum / n,
        "dice":      dice_sum / n,
        "peak_dist": dist_sum / n,
        "peak_diff": diff_sum / n,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    amp_dtype: torch.dtype | None = None,
    voxel_mm: float = 0.5,
    collect_predictions: bool = False,
    plot: bool = False,
    plot_dir: str | Path | None = None,
) -> dict:
    """
    Test-set evaluation.

    Args:
        collect_predictions : if True, returns CPU prediction/target tensors
                              in the output dict (memory-heavy).
        plot                : if True, call plot_fn on every sample and save
                              as `{skull_id}_{freq_hz}_{position_idx:03d}.png`
                              under `plot_dir`.
        plot_dir            : destination directory for plots. Required when
                              plot=True. Created if missing.
    """
    
    """
    Test-set evaluation. Like validate_one_epoch but additionally:
      - returns per-sample metric arrays (not just batch means) for
        distribution analysis.
      - optionally collects predictions/targets on CPU for downstream
        plotting (memory-heavy; opt in via `collect_predictions=True`).
      - optionally invokes `plot_fn(pred, target, meta)` per sample for
        custom visualization, without keeping anything in memory.

    Args:
        collect_predictions :
            if True, returned dict includes:
                "predictions" : list of CPU tensors  (Nx, Ny, Nz)  one per sample
                "targets"     : list of CPU tensors  same shape
            Keep False for large test sets — predictions are 5 MB each in fp32.
        plot_fn :
            optional callable. Signature:
                plot_fn(pred: Tensor, target: Tensor, meta: dict) -> None
            where pred/target are single 3D volumes (Nx, Ny, Nz) on CPU and
            meta is the corresponding sample's metadata dict (skull_id,
            position_idx, freq_idx, freq_hz). Called once per sample so the
            user-side plotting code stays simple.

    Returns dict with:
        loss, dice, peak_dist, peak_diff   : mean over all samples
        per_sample : {
            "loss":      list[float] of length N
            "dice":      list[float]
            "peak_dist": list[float]
            "peak_diff": list[float]
            "meta":      list[dict]
        }
        predictions/targets : if collect_predictions=True
    """
    if plot and plot_dir is None:
        raise ValueError("evaluate(plot=True) requires plot_dir to be set")
    if plot:
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    use_amp = amp_dtype is not None

    per_dice:  list[float] = []
    per_dist:  list[float] = []
    per_diff:  list[float] = []
    per_loss:  list[float] = []
    per_meta:  list[dict]  = []
    preds_cpu: list[torch.Tensor] = []
    tgts_cpu:  list[torch.Tensor] = []
    inference_time:  list[float] = []

    for batch in loader:
        batch = _move_batch_to_device(batch, device)

        if use_amp:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                start = time()
                pred = _forward_model(model, batch)
        else:
            start = time()
            pred = _forward_model(model, batch)

        total = time() - start
        
        pred_f = pred.float()
        tgt_f  = batch["target"].float()
        bsz = tgt_f.size(0)

        inference_time.append(total / bsz)

        for b in range(bsz):
            # Per-sample metrics
            single_metrics = compute_metrics(
                pred_f[b:b+1], tgt_f[b:b+1], voxel_mm=voxel_mm,
            )
            per_dice.append(float(single_metrics["dice"].item()))
            per_dist.append(float(single_metrics["peak_dist"].item()))
            per_diff.append(float(single_metrics["peak_diff"].item()))

            # Per-sample loss (exact for elementwise reductions = mean).
            single_loss = criterion(pred_f[b:b+1], tgt_f[b:b+1])
            per_loss.append(float(single_loss.item()))

            # Per-sample meta. After DataLoader.default_collate, dict values
            # are list/tensor of length B.
            meta_b = {}
            if "meta" in batch:
                for k, v in batch["meta"].items():
                    if torch.is_tensor(v):
                        meta_b[k] = v[b].item() if v.ndim > 0 else v.item()
                    else:
                        meta_b[k] = v[b]
            per_meta.append(meta_b)

            # Optional per-sample side effects: plot or collect.
            need_cpu = plot or collect_predictions
            if need_cpu:
                p_cpu = pred_f[b].detach().cpu()
                t_cpu = tgt_f[b].detach().cpu()
                if plot:
                    skull_id = meta_b.get("skull_id", "S?")
                    freq_hz  = int(meta_b.get("freq_hz", 0))
                    pos_idx  = int(meta_b.get("position_idx", -1))
                    save_path = plot_dir / f"{skull_id}_{freq_hz}_{pos_idx:03d}"
                    plot_fn(p_cpu, t_cpu, meta_b, str(save_path))
                if collect_predictions:
                    preds_cpu.append(p_cpu)
                    tgts_cpu.append(t_cpu)

    n = max(1, len(per_loss))
    out = {
        "loss":      sum(per_loss) / n,
        "dice":      sum(per_dice) / n,
        "peak_dist": sum(per_dist) / n,
        "peak_diff": sum(per_diff) / n,
        "time":      sum(inference_time) / n,
        "per_sample": {
            "loss":      per_loss,
            "dice":      per_dice,
            "peak_dist": per_dist,
            "peak_diff": per_diff,
            "meta":      per_meta,
            "time":      inference_time
        },
    }
    if collect_predictions:
        out["predictions"] = preds_cpu
        out["targets"]     = tgts_cpu
    return out

def plot_fn(
    p_cpu: torch.Tensor,
    t_cpu: torch.Tensor,
    meta_b: dict,
    save_path: str,
) -> None:
    """
    Per-sample comparison figure for test-time visualization.

    Called once per sample from `evaluate(..., plot=True)`. Caller handles
    filename composition and ensures the parent directory exists.

    Args:
        p_cpu  : (Nx, Ny, Nz) prediction on CPU.
        t_cpu  : (Nx, Ny, Nz) ground truth p_max on CPU.
        meta_b : per-sample metadata dict with scalar values
                 (skull_id, position_idx, freq_idx, freq_hz).
        save_path : full destination path ending in '.png'.
    """
    if p_cpu.shape != t_cpu.shape:
        raise ValueError(f"pred {tuple(p_cpu.shape)} != target {tuple(t_cpu.shape)}")
    if p_cpu.ndim != 3:
        raise ValueError(f"expected (Nx, Ny, Nz), got {tuple(p_cpu.shape)}")
    
    s = int(p_cpu.size(0) / 2)
    
    def plot_and_save(x, split, save_path):
        plt.figure(figsize=(15,5.3))

        plt.subplot(1,3,1)
        plt.title('yz')
        plt.imshow(x[s,:,:], cmap='turbo')
        plt.axis('off')

        plt.subplot(1,3,2)
        plt.title('xz')
        plt.imshow(x[:,s,:], cmap='turbo')
        plt.axis('off')

        plt.subplot(1,3,3)
        plt.title('xy')
        plt.imshow(x[:,:,s], cmap='turbo')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f"{save_path}_{split}.png", dpi=150, bbox_inches='tight')
        plt.close()

    plot_and_save(p_cpu, 'pred', save_path)
    plot_and_save(t_cpu, 'true', save_path)