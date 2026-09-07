"""
TFUS Neural Operator — training entry point.

Reads config (config.py), builds dataloaders, model, optimizer,
scheduler, criterion; runs train/val loop with early stopping and
checkpointing; evaluates on the held-out test set with the best ckpt.

Run:
    python train.py --data_path /path/to/h5_dir --run_name baseline_v1
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from config import load_config
from dataset import build_dataloaders, describe_loaders
from models.model import TFUSOperator
from utils import (
    build_criterion, build_lr_scheduler, EarlyStopping,
    train_one_epoch, validate_one_epoch, evaluate, format_eta,
)


# =========================================================================
# Setup helpers
# =========================================================================

def resolve_best_metric(name: str) -> tuple[str, str, float]:
    """
    Map config flag to (EarlyStopping mode, val_stats key, sentinel init value).
    """
    if name == "loss":
        return "min", "val_loss", float("inf")
    if name == "dice":
        return "max", "dice",      float("-inf")
    raise ValueError(f"unknown best_metric {name!r}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(args) -> torch.device:
    if args.gpu_num is None or args.gpu_num < 0:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        print("[warn] CUDA not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(f"cuda:{args.gpu_num}")


def precision_to_amp_dtype(name: str) -> torch.dtype | None:
    return {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[name]


def build_split(args) -> dict[str, list[str]] | None:
    """If any of --split_{train,val,test} given, build a dict; else None (use default)."""
    if args.split_train is None and args.split_val is None and args.split_test is None:
        return None
    if args.split_train is None or args.split_val is None or args.split_test is None:
        raise ValueError("If overriding the split, provide all three of "
                         "--split_train, --split_val, --split_test")
    return {"train": list(args.split_train),
            "val":   list(args.split_val),
            "test":  list(args.split_test)}


def build_optimizer(model: torch.nn.Module, args,
                    criterion: torch.nn.Module | None = None) -> torch.optim.Optimizer:
    """
    AdamW with weight-decay applied only to weight tensors that are not
    biases, LayerNorm/Norm scales, or learnable embeddings/positions.

    If `criterion` carries learnable parameters (e.g. CompositeCriterion's
    uncertainty log_vars), they are added with NO weight decay — otherwise
    decay would pull them toward an implicit equal weighting.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Heuristics: 1D params (biases, norms), embedding-like params,
        # latent queries, modality embeddings, null vectors -> no decay.
        if p.ndim <= 1 or name.endswith(".bias") \
           or "norm" in name.lower() \
           or "modality_embed" in name \
           or "latent_queries" in name \
           or "null" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    if criterion is not None:
        for p in criterion.parameters():
            if p.requires_grad:
                no_decay.append(p)
    groups = [
        {"params": decay,    "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=args.learning_rate, betas=(0.9, 0.95))


def build_model(args) -> TFUSOperator:
    if args.data_shape % args.patch_vox != 0:
        raise ValueError(f"data_shape={args.data_shape} not divisible by patch_vox={args.patch_vox}")
    G = args.data_shape // args.patch_vox
    domain_size_mm = tuple(d * args.voxel_mm for d in args.domain_size_vox)
    return TFUSOperator(
        embed_dim=args.embed_dim,
        patch_size=args.patch_vox,
        focal_grid=(G, G, G),
        num_latents=args.num_latents,
        num_heads=args.num_heads,
        depth=args.depth,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        use_dit=not args.no_dit,
        freq_range_hz=tuple(args.freq_range_hz),
        cond_sources=tuple(args.cond_sources),
        domain_size_mm=domain_size_mm,
        cond_dropout=args.cond_dropout,
        coord_max_freq=args.coord_max_freq,
        ct_stem_depth=args.ct_stem_depth,
        ct_stem_channels=args.ct_stem_channels,
    )


# =========================================================================
# Checkpoint I/O
# =========================================================================

def save_checkpoint(
    path: str,
    *,
    model, optimizer, scheduler, scaler,
    epoch: int, best_val_metric: float, best_metric_kind: str, args,
) -> None:
    state = {
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler":    scaler.state_dict() if scaler is not None else None,
        "epoch":     epoch,
        "best_val_metric":  best_val_metric,
        "best_metric_kind": best_metric_kind,   # 'loss' or 'dice'
        "args":      vars(args),
    }
    tmp = str(path) + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: str,
    *,
    model, optimizer, scheduler, scaler, device,
    expected_metric_kind: str | None = None,
) -> tuple[int, float]:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])

    ck_kind = state.get("best_metric_kind", None)
    if expected_metric_kind is not None and ck_kind is not None \
       and ck_kind != expected_metric_kind:
        raise ValueError(
            f"resume: checkpoint was tracked on best_metric={ck_kind!r} "
            f"but current run uses best_metric={expected_metric_kind!r}. "
            f"This invalidates the stored best value. Change --best_metric "
            f"to match, or start a fresh run."
        )

    best = state.get("best_val_metric",
                     state.get("best_val_loss", None))   # backward-compat
    if best is None:
        best = float("inf") if expected_metric_kind == "loss" else float("-inf")
    return int(state.get("epoch", 0)) + 1, float(best)


# =========================================================================
# Logger
# =========================================================================

class EpochLogger:
    """Writes one line per epoch to stdout and `training.log`."""

    HEADER = (
        "epoch\t"
        "train_loss\ttrain_dice\ttrain_peak_dist\ttrain_peak_diff\t"
        "val_loss\tval_dice\tval_peak_dist\tval_peak_diff\t"
        "lr\ttime\n"
    )

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with open(self.log_path, "w") as f:
                f.write(self.HEADER)

    def log(self, epoch: int, train_stats: dict, val_stats: dict,
            lr: float, elapsed: float, eta_seconds: float) -> None:
        # Console: compact two-line layout to keep the train/val split readable.
        line1 = (
            f"  ep{epoch:3d}  TRAIN  "
            f"loss={train_stats['train_loss']:.4f}  "
            f"dice={train_stats['dice']:.3f}  "
            f"peak_dist={train_stats['peak_dist']:.2f}mm  "
            f"peak_diff={train_stats['peak_diff']:.2f}%"
        )
        eta_txt = (f"  ETA~{format_eta(eta_seconds)}"
                   if eta_seconds is not None else "")
        line2 = (
            f"          VAL    "
            f"loss={val_stats['val_loss']:.4f}  "
            f"dice={val_stats['dice']:.3f}  "
            f"peak_dist={val_stats['peak_dist']:.2f}mm  "
            f"peak_diff={val_stats['peak_diff']:.2f}%  "
            f"lr={lr:.2e}  ({elapsed:.1f}s){eta_txt}"
        )
        print(line1)
        print(line2)
        with open(self.log_path, "a") as f:
            f.write(
                f"{epoch}\t"
                f"{train_stats['train_loss']:.6f}\t{train_stats['dice']:.6f}\t"
                f"{train_stats['peak_dist']:.6f}\t{train_stats['peak_diff']:.6f}\t"
                f"{val_stats['val_loss']:.6f}\t{val_stats['dice']:.6f}\t"
                f"{val_stats['peak_dist']:.6f}\t{val_stats['peak_diff']:.6f}\t"
                f"{lr:.6e}\t{elapsed:.2f}\n"
            )


# =========================================================================
# Main
# =========================================================================

def main():
    args = load_config()

    # --- Bookkeeping ---
    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Save full config for reproducibility
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"Run directory: {run_dir.resolve()}")

    set_seed(args.seed)
    device = resolve_device(args)
    print(f"Device: {device}")

    # --- Data ---
    split = build_split(args)
    loaders = build_dataloaders(
        root=args.data_path,
        frequencies=args.frequencies,
        split=split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        skull_modality=args.skull_modality,
        positions=args.positions,
        roi_size_vox=args.data_shape,
        patch_vox=args.patch_vox,
        domain_size_vox=tuple(args.domain_size_vox),
        voxel_mm=args.voxel_mm,
        normalize_field=args.normalize_field,
        seed=args.seed,
        downsample=args.downsample,
        augment_mirror=args.mirror_aug,
        mirror_prob=args.mirror_prob,
        mirror_axis=args.mirror_axis,
        eval_heldout_per_pair=args.eval_heldout_per_pair,
    )
    train_loader, val_loader, test_loader, train_heldout_loader = loaders
    print(describe_loaders(loaders))

    metric_voxel_mm = args.voxel_mm * args.downsample

    # --- Model ---
    model = build_model(args).to(device)
    bd = model.param_breakdown()
    print(f"Model params (M): {bd}")

    # --- Criterion, optimizer, scheduler, scaler ---
    criterion = build_criterion(args).to(device)
    optimizer = build_optimizer(model, args, criterion=criterion)
    scheduler = build_lr_scheduler(optimizer, args, steps_per_epoch=len(train_loader))
    amp_dtype = precision_to_amp_dtype(args.precision)
    scaler = torch.cuda.amp.GradScaler() if (amp_dtype == torch.float16 and device.type == "cuda") else None

    # --- Resume ---
    # --- Best-metric resolution ---
    mode, val_key, init_best = resolve_best_metric(args.best_metric)
    best_val_metric = init_best

    start_epoch = 0
    if args.resume is not None:
        start_epoch, best_val_metric = load_checkpoint(
            args.resume,
            model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, device=device,
            expected_metric_kind=args.best_metric,
        )
        print(f"Resumed from {args.resume}; starting epoch={start_epoch}, "
            f"best_val_{args.best_metric}={best_val_metric:.4f}")

    # --- WandB (optional) ---
    use_wandb = args.wandb_pj is not None
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_pj, name=args.run_name, config=vars(args))
            wandb.watch(model, log='all')
        except ImportError:
            print("[warn] wandb requested but not installed; continuing without it.")
            use_wandb = False

    # --- Loop ---
    logger = EpochLogger(run_dir / "training.log")

    early = EarlyStopping(patience=args.patience, mode=mode)
    if start_epoch > 0 and best_val_metric != init_best:
        early.best = best_val_metric

    grad_clip_val = args.grad_clip if args.grad_clip and args.grad_clip > 0 else None

    # History buffer: every epoch's full train/val stats appended.
    history: list[dict] = []
    history_path = run_dir / "history.json"
    epoch_times: list[float] = []   # measured per-epoch wall times, for ETA

    print("=" * 60)
    print("Starting training")
    print("=" * 60)

    for epoch in range(start_epoch, args.num_epochs):
        t0 = time.time()

        train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scheduler=scheduler, scaler=scaler, amp_dtype=amp_dtype,
            grad_clip=grad_clip_val, voxel_mm=metric_voxel_mm,
            log_every=args.log_every, epoch=epoch,
            total_epochs=args.num_epochs,
        )
        val_stats = validate_one_epoch(
            model, val_loader, criterion, device,
            amp_dtype=amp_dtype, voxel_mm=metric_voxel_mm,
        )
        # In-distribution generalization: seen skulls, unseen positions.
        heldout_stats = None
        if train_heldout_loader is not None:
            heldout_stats = validate_one_epoch(
                model, train_heldout_loader, criterion, device,
                amp_dtype=amp_dtype, voxel_mm=metric_voxel_mm,
            )

        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        epoch_times.append(elapsed)
        recent = epoch_times[-5:]
        avg_epoch = sum(recent) / len(recent)
        remaining_epochs = max(0, args.num_epochs - (epoch + 1))
        eta_seconds = avg_epoch * remaining_epochs

        logger.log(epoch, train_stats, val_stats, cur_lr, elapsed, eta_seconds=eta_seconds)
        if heldout_stats is not None:
            print(f"            [heldout (seen skull, unseen pos)] "
                  f"loss={heldout_stats['val_loss']:.4f}  "
                  f"dice={heldout_stats['dice']:.3f}  "
                  f"peak_dist={heldout_stats['peak_dist']:.2f}mm  "
                  f"peak_diff={heldout_stats['peak_diff']:.2f}%")

        # Accumulate history and persist (atomic-ish: write each epoch).
        history.append({
            "epoch":   epoch,
            "lr":      cur_lr,
            "time":    elapsed,
            "train":   {k: float(v) for k, v in train_stats.items()},
            "val":     {k: float(v) for k, v in val_stats.items()},
            **({"train_heldout": {k: float(v) for k, v in heldout_stats.items()}}
               if heldout_stats is not None else {}),
        })
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        if use_wandb:
            log_dict = {
                "epoch":            epoch,
                "train_loss":       train_stats["train_loss"],
                "train_dice":       train_stats["dice"],
                "train_peak_dist":  train_stats["peak_dist"],
                "train_peak_diff":  train_stats["peak_diff"],
                "val_loss":         val_stats["val_loss"],
                "val_dice":         val_stats["dice"],
                "val_peak_dist":    val_stats["peak_dist"],
                "val_peak_diff":    val_stats["peak_diff"],
                "lr":               cur_lr,
            }
            if heldout_stats is not None:
                log_dict.update({
                    "heldout_loss":      heldout_stats["val_loss"],
                    "heldout_dice":      heldout_stats["dice"],
                    "heldout_peak_dist": heldout_stats["peak_dist"],
                    "heldout_peak_diff": heldout_stats["peak_diff"],
                })
            wandb.log(log_dict)

        # Early stopping
        should_stop = early.step(val_stats[val_key], epoch)

        if early.is_improved:
            best_val_metric = early.best
            save_checkpoint(
                run_dir / "ckpt_best.pt",
                model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, epoch=epoch,
                best_val_metric=best_val_metric,
                best_metric_kind=args.best_metric,
                args=args,
            )

        save_checkpoint(
            run_dir / "ckpt_last.pt",
            model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, epoch=epoch,
            best_val_metric=best_val_metric,
            best_metric_kind=args.best_metric,
            args=args,
        )

        # Periodic snapshot.
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                run_dir / f"ckpt_epoch_{epoch:04d}.pt",
                model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, epoch=epoch,
                best_val_metric=best_val_metric,
                best_metric_kind=args.best_metric,
                args=args,
            )

        if should_stop:
            print(f"\nEarly stop at epoch {epoch} "
                  f"(best {args.best_metric} {early.best:.4f} at epoch {early.best_epoch})")
            break

    # --- Final test evaluation with best checkpoint ---
    best_path = run_dir / "ckpt_best.pt"
    if best_path.exists():
        print("\n" + "=" * 60)
        print(f"Test evaluation with best checkpoint (epoch {early.best_epoch})")
        print("=" * 60)
        _ = load_checkpoint(
            str(best_path), model=model, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler, device=device,
            expected_metric_kind=args.best_metric,
        )

        test_stats = evaluate(
        model, test_loader, criterion, device,
        amp_dtype=amp_dtype, voxel_mm=metric_voxel_mm,
        collect_predictions=args.collect_predictions,
        plot=args.plot,
        plot_dir=(run_dir / "figs") if args.plot else None,
            )   
        print(f"  test_loss = {test_stats['loss']:.4f}")
        print(f"  dice      = {test_stats['dice']:.3f}")
        print(f"  peak_dist = {test_stats['peak_dist']:.2f} mm")
        print(f"  peak_diff = {test_stats['peak_diff']:.2f}%")
        print(f"  inference_time = {test_stats['time']:.6f}s")

        # Persist test results: mean metrics + per-sample lists.
        with open(run_dir / "test_results.json", "w") as f:
            # per_sample contains list-of-dicts metadata; tensor-free already.
            json.dump({
                "mean": {
                    "loss":      float(test_stats["loss"]),
                    "dice":      float(test_stats["dice"]),
                    "peak_dist": float(test_stats["peak_dist"]),
                    "peak_diff": float(test_stats["peak_diff"]),
                    "time": float(test_stats["time"]),
                },
                "per_sample": test_stats["per_sample"],
            }, f, indent=2)

        if use_wandb:
            wandb.log({
                "test_loss":      test_stats["loss"],
                "test_dice":      test_stats["dice"],
                "test_peak_dist": test_stats["peak_dist"],
                "test_peak_diff": test_stats["peak_diff"],
            })

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()