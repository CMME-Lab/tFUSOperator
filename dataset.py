"""
TFUS Neural Operator — Dataset.

Wraps per-skull HDF5 files into a flat indexable dataset.

H5 layout per file (S01.h5 ... S13.h5), shapes as read by h5py:
    field/F{FREQ}/ff      (300, 112, 112, 112)    free-field volumes
    field/F{FREQ}/pmax    (300, 112, 112, 112)    target maximum-pressure volumes
    input/P_ROI           (300, 3)                field ROI center, voxel index (x,y,z)
    input/S_ROI           (300, 3)                skull ROI center, voxel index (x,y,z)
    input/T_pos           (300, 3)                transducer position
    input/T_angle         (300, 3)                transducer focal-axis unit vector
    skull/CT              (300, 112, 112, 112)
    skull/MR              (300, 112, 112, 112)
    cond/<varname>        (300, ...)              future scalar/vector conditions

A sample is identified by (skull_id, position_idx, freq_idx). The Dataset
exposes a flat 1-D index over the Cartesian product of the requested skull
list and frequency list. Position index ranges 0..299 (or a subset).

Volume axis convention
----------------------
MATLAB arrays were written with shape (Nx, Ny, Nz, Npos) and h5py exposes
them with axes reversed, i.e. (Npos, Nz, Ny, Nx). Per-sample slicing yields
(Nz, Ny, Nx). We transpose to (Nx, Ny, Nz) so that z varies fastest in
memory, matching the encoder/decoder z-fastest token order. For 112^3
cubes the shape stays (112, 112, 112) either way; what matters is the
memory layout, not the shape tuple.

ROI center axis convention
--------------------------
MATLAB writes input/P_ROI as (3, 300) with the leading 3 being (x, y, z).
h5py reads this back as (300, 3) with the trailing 3 being (x, y, z).
We index sample p as `arr[p]` -> length-3 vector in (x, y, z) order.
"""

from __future__ import annotations

import os
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Default split. Can be overridden when calling build_dataloaders.
DEFAULT_SPLIT = {
    "train": ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"],
    "val":   ["s9", "s10"],
    "test":  ["s11", "s12", "s13"],
}

# ---------------------------------------------------------------------------
# Coordinate computation
# ---------------------------------------------------------------------------

def make_patch_coords(
    roi_center_vox: torch.Tensor,
    *,
    roi_size_vox: int = 112,
    patch_vox: int = 4,
    domain_size_vox: tuple[int, int, int] = (450, 450, 300),
    voxel_mm: float = 0.5,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build normalized patch-center coordinates for one ROI in z-fastest token
    order, matching the encoder's Conv3d.flatten(2) output convention.

    Conventions
    -----------
    - roi_center_vox: 1-based voxel index in (x, y, z), matching MATLAB.
      e.g. tensor([225.5, 225.5, 150.5]).
    - Volume axis order: (x, y, z) with z varying fastest in memory.
    - Token order: z-fastest, t = iz + Nz * (iy + Ny * ix).
    - Normalization: simulation domain mapped to [-1, 1]^3 per axis
      (axis-anisotropic when domain is non-cubic).

    Args:
        roi_center_vox:    (3,) tensor, voxel-index center of the ROI.
        roi_size_vox:      ROI side length in voxels. Must be divisible by
                           patch_vox.
        patch_vox:         patch side length in voxels.
        domain_size_vox:   simulation domain size (Nx, Ny, Nz) in voxels.
        voxel_mm:          isotropic voxel spacing in mm.

    Returns:
        coords: (Np, 3) tensor of normalized patch-center coords,
                Np = (roi_size_vox / patch_vox) ** 3.
    """
    if roi_size_vox % patch_vox != 0:
        raise ValueError(f"roi_size_vox={roi_size_vox} not divisible by patch_vox={patch_vox}")
    G = roi_size_vox // patch_vox
    patch_mm = patch_vox * voxel_mm

    if not isinstance(roi_center_vox, torch.Tensor):
        roi_center_vox = torch.as_tensor(roi_center_vox)
    roi_center_vox = roi_center_vox.to(dtype=dtype, device=device)
    if roi_center_vox.shape != (3,):
        raise ValueError(f"roi_center_vox must be shape (3,), got {tuple(roi_center_vox.shape)}")

    # MATLAB voxel i in [1, N] -> mm = (i - 0.5) * voxel_mm.
    roi_center_mm = (roi_center_vox - 0.5) * voxel_mm                # (3,)

    # Per-axis patch-center offsets from ROI center, in mm.
    j = torch.arange(G, dtype=dtype, device=device)
    rel = (j + 0.5 - G / 2.0) * patch_mm                              # (G,)

    px = roi_center_mm[0] + rel
    py = roi_center_mm[1] + rel
    pz = roi_center_mm[2] + rel

    # Build (G, G, G, 3) grid in (x, y, z) axis order then flatten in
    # z-fastest (default contiguous) order to match the encoder.
    PX, PY, PZ = torch.meshgrid(px, py, pz, indexing='ij')             # (G, G, G)
    coords_mm = torch.stack([PX, PY, PZ], dim=-1).reshape(-1, 3)       # (G^3, 3)

    # Normalize per axis to [-1, 1].
    domain_vox = torch.tensor(domain_size_vox, dtype=dtype, device=device)
    domain_mm = domain_vox * voxel_mm
    domain_center_mm = domain_mm / 2.0
    domain_half_mm = domain_mm / 2.0

    coords = (coords_mm - domain_center_mm) / domain_half_mm
    return coords

# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def sagittal_mirror(
    ff: np.ndarray,
    pmax: np.ndarray,
    skull: np.ndarray,
    p_roi_vox: np.ndarray,
    s_roi_vox: np.ndarray,
    t_pos_vox: np.ndarray,
    t_angle: np.ndarray,
    *,
    domain_size_vox: tuple[int, int, int],
    axis: int,
):
    """Reflect the whole scene across ONE coordinate plane (sagittal mirror).

    `axis` is a spatial axis index into the (Nx, Ny, Nz) volume / the (x, y, z)
    vectors: 0 = x, 1 = y, 2 = z. A valid sagittal mirror flips exactly ONE
    axis — the left-right (sagittal-plane-normal) axis. Set `axis` to whichever
    of x / y is left-right in your data; keep z (beam / superior-inferior).
    Do NOT flip two axes at once: flipping two axes is a 180-degree rotation
    (orientation-preserving), not a reflection, and the (input, output) pair is
    then not a valid simulation pair (the skull is not rotation-symmetric).

    Correctness:
      - Volume content flip is `np.flip` on `axis`, INDEPENDENT of where the
        reflection plane sits (crop-content reflection lemma).
      - Point vectors (ROI centers, transducer position) reflect about the
        domain-center plane:  v -> (domain_size_vox[axis] + 1) - v  (1-based
        voxel index). make_patch_coords(reflected_center) then reproduces the
        correctly mirrored, normalized coords (verified identity).
      - Direction vector (t_angle, a unit vector) flips sign on `axis` only.
    Both ROIs use the SAME plane, so field<->skull relative geometry is kept.
    """
    a = int(axis)
    ff    = np.ascontiguousarray(np.flip(ff,    axis=a))
    pmax  = np.ascontiguousarray(np.flip(pmax,  axis=a))
    skull = np.ascontiguousarray(np.flip(skull, axis=a))

    p_roi_vox = p_roi_vox.copy()
    s_roi_vox = s_roi_vox.copy()
    t_pos_vox = t_pos_vox.copy()
    t_angle   = t_angle.copy()

    reflect = (float(domain_size_vox[a]) + 1.0)
    p_roi_vox[a] = reflect - p_roi_vox[a]
    s_roi_vox[a] = reflect - s_roi_vox[a]
    t_pos_vox[a] = reflect - t_pos_vox[a]
    t_angle[a]   = -t_angle[a]

    return ff, pmax, skull, p_roi_vox, s_roi_vox, t_pos_vox, t_angle



# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TFUSDataset(Dataset):
    """
    Dataset over the Cartesian product of (skull_ids, positions, frequencies).

    Args:
        root:              directory containing S01.h5, S02.h5, ... files.
        skull_ids:         iterable of skull identifiers, e.g. ["S01", "S07"].
        frequencies:       iterable of frequencies in Hz, e.g. [250000, 400000, 500000].
                           Each must correspond to an H5 group "field/F{freq}".
        skull_modality:    "CT" (default) or "MR". Selects skull/{modality}.
        positions:         iterable of position indices to include. None -> all 300.
                           Useful for subset experiments.
        roi_size_vox:      passed to make_patch_coords.
        patch_vox:         passed to make_patch_coords. CHANGE THIS WHEN
                           CHANGING MODEL PATCH SIZE.
        domain_size_vox:   passed to make_patch_coords.
        voxel_mm:          passed to make_patch_coords.
        normalize_field:   if True, applies sign-log compression to ff and
                           pmax (apply consistently between train and eval).
        return_metadata:   include a 'meta' dict in each sample.

    Returned sample (dict):
        ff_vol         (1, Nx, Ny, Nz)   torch.float32   p_ff
        ff_coords      (Np, 3)           torch.float32   normalized, z-fastest
        skull_vol         (1, Nx, Ny, Nz)   torch.float32   CT or MR
        skull_coords      (Np, 3)           torch.float32   normalized, z-fastest
        target            (Nx, Ny, Nz)      torch.float32   p_max
        freq              (1,)              torch.float32   Hz
        transducer_pos    (3,)              torch.float32
        transducer_angle  (3,)              torch.float32
        extra_conditions  dict[str, Tensor] (optional, from cond/*)
        meta              dict              (optional)
    """

    POS_PER_SKULL = 300

    def __init__(
        self,
        root: str,
        skull_ids: Sequence[str],
        frequencies: Sequence[int],
        skull_modality: str = "CT",
        positions: Sequence[int] | None = None,
        roi_size_vox: int = 112,
        patch_vox: int = 4,
        domain_size_vox: tuple[int, int, int] = (450, 450, 300),
        voxel_mm: float = 0.5,
        normalize_field: bool = False,
        return_metadata: bool = True,
        downsample: int = 1,
        augment_mirror: bool = False,
        mirror_prob: float = 0.5,
        mirror_axis: int = 0,
        aug_seed: int | None = None,
        sample_filter: set | None = None,
    ):
        super().__init__()
        if skull_modality not in ("CT", "MR"):
            raise ValueError(f"skull_modality must be 'CT' or 'MR', got {skull_modality!r}")
        self.root = root
        self.skull_ids = list(skull_ids)
        self.frequencies = list(frequencies)
        self.skull_modality = skull_modality
        self.positions = list(positions) if positions is not None else list(range(self.POS_PER_SKULL))
        self.roi_size_vox = roi_size_vox
        self.patch_vox = patch_vox
        self.domain_size_vox = tuple(domain_size_vox)
        self.voxel_mm = voxel_mm
        self.normalize_field = normalize_field
        self.return_metadata = return_metadata
        self.downsample = downsample
        self.domain_size_vox_eff = tuple(d // self.downsample for d in self.domain_size_vox)
        self.voxel_mm_eff        = self.voxel_mm        *  self.downsample

        # Mirror augmentation (train only; build_dataloaders enables it on the
        # train split). RNG is created lazily per worker for fork-safety.
        self.augment_mirror = augment_mirror
        self.mirror_prob = float(mirror_prob)
        self.mirror_axis = int(mirror_axis)
        self._aug_seed = aug_seed
        self._rng_obj: np.random.Generator | None = None

        # Pre-flight: every requested H5 file must exist.
        for sid in self.skull_ids:
            p = self._path_for(sid)
            if not os.path.isfile(p):
                raise FileNotFoundError(f"missing H5 file: {p}")

        # Flat index: (skull_id, position_idx, freq_idx), skull outermost.
        self._index: list[tuple[str, int, int]] = [
            (sid, p, fi)
            for sid in self.skull_ids
            for p in self.positions
            for fi in range(len(self.frequencies))
        ]
        # Optional allowlist: restrict to an explicit subset of samples. Used to
        # carve a held-out eval set out of the train split (train loader gets the
        # complement, eval loader gets the held-out set) so the two are disjoint
        # and there is no leakage.
        if sample_filter is not None:
            allow = set(sample_filter)
            self._index = [t for t in self._index if t in allow]

        # Lazy per-worker H5 handles.
        self._h5_files: dict[str, h5py.File] | None = None

    # ------------------------------------------------------------------
    # H5 handle management
    # ------------------------------------------------------------------

    def _path_for(self, skull_id: str) -> str:
        return os.path.join(self.root, f"{skull_id}.h5")

    def _get_file(self, skull_id: str) -> h5py.File:
        """Lazy-open per worker (h5py file handles are not fork-safe)."""
        if self._h5_files is None:
            self._h5_files = {}
        if skull_id not in self._h5_files:
            self._h5_files[skull_id] = h5py.File(
                self._path_for(skull_id), 'r', libver='latest', swmr=False,
            )
        return self._h5_files[skull_id]

    def close(self):
        if self._h5_files is not None:
            for f in self._h5_files.values():
                try:
                    f.close()
                except Exception:
                    pass
            self._h5_files = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _rng(self) -> np.random.Generator:
        """Lazy per-worker RNG (fork-safe). Distinct stream per worker; advances
        across epochs so augmentation differs each epoch."""
        if self._rng_obj is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            base = 0 if self._aug_seed is None else int(self._aug_seed)
            self._rng_obj = np.random.default_rng(base + 1000 + wid)
        return self._rng_obj

    # ------------------------------------------------------------------
    # Volume axis canonicalization
    # ------------------------------------------------------------------

    @staticmethod
    def _vol_matlab_to_torch(arr: np.ndarray, downsample: int) -> np.ndarray:
        """
        Convert h5py-read volume (Nz, Ny, Nx) (i.e. MATLAB (Nx, Ny, Nz)
        with axes auto-reversed) to PyTorch-row-major (Nx, Ny, Nz) so that
        z varies fastest in memory.
        """
        if arr.ndim != 3:
            raise ValueError(f"expected 3D volume, got shape {arr.shape}")
        if downsample > 1:
            t = torch.from_numpy(arr)[None, None].float()         # (1, 1, Nz, Ny, Nx)
            t = F.avg_pool3d(t, kernel_size=downsample)
            arr = t.squeeze(0).squeeze(0).numpy().astype(arr.dtype, copy=False)
        return np.transpose(arr, (2, 1, 0))

    # ------------------------------------------------------------------
    # Loaders for groups
    # ------------------------------------------------------------------

    def _load_field_volumes(
        self,
        f: h5py.File,
        position_idx: int,
        freq_hz: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(p_ff, p_max), both (Nx, Ny, Nz) float32."""
        grp = f[f"field/F{freq_hz}"]
        ff = np.asarray(grp["ff"][position_idx],   dtype=np.float32)   # (Nz, Ny, Nx)
        pm = np.asarray(grp["pmax"][position_idx], dtype=np.float32)
        return self._vol_matlab_to_torch(ff, self.downsample), self._vol_matlab_to_torch(pm, self.downsample)

    def _load_skull_volume(self, f: h5py.File, position_idx: int) -> np.ndarray:
        arr = np.asarray(f[f"skull/{self.skull_modality}"][position_idx], dtype=np.float32)
        return self._vol_matlab_to_torch(arr, self.downsample)

    def _load_input_vectors(self, f: h5py.File, position_idx: int) -> dict[str, np.ndarray]:
        return {
            "P_ROI":   np.asarray(f["input/P_ROI"][:,position_idx],   dtype=np.float32),
            "S_ROI":   np.asarray(f["input/S_ROI"][:,position_idx],   dtype=np.float32),
            "T_pos":   np.asarray(f["input/T_pos"][:,position_idx],   dtype=np.float32),
            "T_angle": np.asarray(f["input/T_angle"][:,position_idx], dtype=np.float32),
        }

    def _load_extra_conditions(self, f: h5py.File, position_idx: int) -> dict[str, np.ndarray]:
        """Loader for future scalar/vector conditions under cond/<varname>."""
        if "cond" not in f:
            return {}
        out = {}
        for name in f["cond"].keys():
            out[name] = np.asarray(f["cond"][name][position_idx], dtype=np.float32)
        return out

    # ------------------------------------------------------------------
    # Sign-log normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _min_max_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        vmin = x.min()
        vmax = x.max()
        denom = vmax - vmin
        if denom <= eps:
            return np.zeros_like(x)
        return ((x - vmin) / denom).astype(x.dtype, copy=False)

    def _to_eff_vox(self, roi_vox_np):
        v = torch.from_numpy(roi_vox_np)
        mm = (v - 0.5) * self.voxel_mm            # original-frame mm
        return mm / self.voxel_mm_eff + 0.5       # eff-frame voxel index

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        skull_id, position_idx, freq_idx = self._index[idx]
        freq_hz = self.frequencies[freq_idx]

        f = self._get_file(skull_id)

        ff_np, pmax_np = self._load_field_volumes(f, position_idx, freq_hz)
        skull_np       = self._load_skull_volume(f, position_idx)
        if self.normalize_field:
            ff_np   = self._min_max_normalize(ff_np)
            pmax_np = self._min_max_normalize(pmax_np)

        vecs = self._load_input_vectors(f, position_idx)
        p_roi_vox = vecs["P_ROI"]
        s_roi_vox = vecs["S_ROI"]
        t_pos_vox = vecs["T_pos"]
        t_angle   = vecs["T_angle"]

        # Sagittal mirror augmentation (train only). min-max normalize is
        # flip-invariant, so order vs normalization above does not matter; this
        # must precede coord computation since coords come from the (reflected)
        # ROI centers.
        if self.augment_mirror and float(self._rng().random()) < self.mirror_prob:
            (ff_np, pmax_np, skull_np,
             p_roi_vox, s_roi_vox, t_pos_vox, t_angle) = sagittal_mirror(
                ff_np, pmax_np, skull_np,
                p_roi_vox, s_roi_vox, t_pos_vox, t_angle,
                domain_size_vox=self.domain_size_vox, axis=self.mirror_axis,
            )

        field_coords = make_patch_coords(
            self._to_eff_vox(p_roi_vox),
            roi_size_vox=self.roi_size_vox, patch_vox=self.patch_vox,
            domain_size_vox=self.domain_size_vox_eff, voxel_mm=self.voxel_mm_eff,
        )
        skull_coords = make_patch_coords(
            self._to_eff_vox(s_roi_vox),
            roi_size_vox=self.roi_size_vox, patch_vox=self.patch_vox,
            domain_size_vox=self.domain_size_vox_eff, voxel_mm=self.voxel_mm_eff,
        )

        t_pos_mm = (t_pos_vox - 0.5) * self.voxel_mm

        extras = self._load_extra_conditions(f, position_idx)

        sample = {
            "ff_vol":        torch.from_numpy(ff_np).unsqueeze(0).contiguous(),
            "ff_coords":     field_coords.contiguous(),
            "skull_vol":        torch.from_numpy(skull_np).unsqueeze(0).contiguous(),
            "skull_coords":     skull_coords.contiguous(),
            "target":           torch.from_numpy(pmax_np).contiguous(),
            "freq":             torch.tensor([float(freq_hz)], dtype=torch.float32),
            "transducer_pos":   torch.from_numpy(t_pos_mm).contiguous(),
            "transducer_angle": torch.from_numpy(t_angle).contiguous(),
        }
        if extras:
            sample["extra_conditions"] = {k: torch.from_numpy(v) for k, v in extras.items()}
        if self.return_metadata:
            sample["meta"] = {
                "skull_id":     skull_id,
                "position_idx": int(position_idx),
                "freq_idx":     int(freq_idx),
                "freq_hz":      float(freq_hz),
            }
        return sample

def _validate_split(split: dict[str, Sequence[str]]) -> None:
    required = {"train", "val", "test"}
    missing = required - set(split.keys())
    if missing:
        raise ValueError(f"split missing keys: {sorted(missing)}")
    seen: dict[str, str] = {}
    for k in required:
        for sid in split[k]:
            if sid in seen:
                raise ValueError(
                    f"skull {sid!r} appears in both {seen[sid]} and {k} splits"
                )
            seen[sid] = k
 
 
def build_dataloaders(
    root: str,
    frequencies: Sequence[int],
    *,
    split: dict[str, Sequence[str]] | None = None,
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    skull_modality: str = "CT",
    positions: Sequence[int] | None = None,
    downsample: int = 1,
    roi_size_vox: int = 112,
    patch_vox: int = 4,
    domain_size_vox: tuple[int, int, int] = (450, 450, 300),
    voxel_mm: float = 0.5,
    normalize_field: bool = False,
    drop_last_train: bool = True,
    seed: int | None = None,
    augment_mirror: bool = False,
    mirror_prob: float = 0.5,
    mirror_axis: int = 0,
    eval_heldout_per_pair: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader | None]:
    """
    Build (train, val, test) DataLoaders with skull-level split.
 
    Args:
        root:                directory containing S01.h5 ... S13.h5
        frequencies:         e.g. [250000, 400000, 500000]
        split:               override DEFAULT_SPLIT.
        batch_size:          per-step batch size (same for train/val/test).
                             val/test use this too; reduce if memory tight.
        num_workers:         DataLoader workers per loader.
        pin_memory:          standard CUDA optimization.
        persistent_workers:  keep workers alive across epochs (faster).
                             Requires num_workers > 0.
        skull_modality:      "CT" or "MR" (forwarded to TFUSDataset).
        positions:           subset of position indices in [0, 300). None = all.
        roi_size_vox, patch_vox, domain_size_vox, voxel_mm: passed through.
        normalize_field:     sign-log compression of ff/pmax.
        drop_last_train:     drop trailing partial batch on the train loader
                             so that batch shape is constant (useful for some
                             optimizers / mixed-precision).
        seed:                seed for the train-loader generator. If None,
                             a default Generator is used.
 
    Returns:
        (train_loader, val_loader, test_loader)
    """
    if split is None:
        split = DEFAULT_SPLIT
    _validate_split(split)
 
    # Persistent workers requires num_workers > 0.
    persistent = persistent_workers and num_workers > 0

    # Optional held-out eval set carved from the TRAIN split: for each train
    # skull, reserve `eval_heldout_per_pair` POSITIONS and exclude them from
    # training across ALL frequencies. A held-out position is therefore never
    # seen at any frequency, giving a clean "seen skull, UNSEEN position"
    # in-distribution generalization signal (distinct from val/test = UNSEEN
    # skulls). The reserved positions are sampled once per skull and applied to
    # every frequency.
    train_filter = None
    heldout_filter: set = set()
    pos_list = list(positions) if positions is not None else list(range(TFUSDataset.POS_PER_SKULL))
    n_freq = len(frequencies)
    if eval_heldout_per_pair > 0:
        n_hold = min(eval_heldout_per_pair, max(0, len(pos_list) - 1))  # keep >=1 for train
        rng = np.random.default_rng(0 if seed is None else int(seed))
        train_filter = set()
        for sid in split["train"]:
            perm = rng.permutation(pos_list)
            held_pos = {int(p) for p in perm[:n_hold]}        # same positions for all freqs
            for p in pos_list:
                bucket = heldout_filter if int(p) in held_pos else train_filter
                for fi in range(n_freq):
                    bucket.add((sid, int(p), fi))
 
    train_ds = TFUSDataset(
        root=root, skull_ids=split["train"], frequencies=frequencies,
        skull_modality=skull_modality, positions=positions,
        roi_size_vox=roi_size_vox, patch_vox=patch_vox,
        domain_size_vox=domain_size_vox, voxel_mm=voxel_mm,
        normalize_field=normalize_field, downsample=downsample,
        augment_mirror=augment_mirror, mirror_prob=mirror_prob,
        mirror_axis=mirror_axis, aug_seed=seed,
        sample_filter=train_filter,
    )
    val_ds = TFUSDataset(
        root=root, skull_ids=split["val"], frequencies=frequencies,
        skull_modality=skull_modality, positions=positions,
        roi_size_vox=roi_size_vox, patch_vox=patch_vox,
        domain_size_vox=domain_size_vox, voxel_mm=voxel_mm,
        normalize_field=normalize_field, downsample=downsample,
    )
    test_ds = TFUSDataset(
        root=root, skull_ids=split["test"], frequencies=frequencies,
        skull_modality=skull_modality, positions=positions,
        roi_size_vox=roi_size_vox, patch_vox=patch_vox,
        domain_size_vox=domain_size_vox, voxel_mm=voxel_mm,
        normalize_field=normalize_field, downsample=downsample,
    )
 
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(int(seed))
 
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent, drop_last=drop_last_train,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent, drop_last=False,
    )

    # Held-out eval loader over the reserved train-skull positions (seen skulls,
    # unseen positions). Augmentation OFF, no shuffle. None if disabled.
    train_heldout_loader = None
    if eval_heldout_per_pair > 0 and len(heldout_filter) > 0:
        train_heldout_ds = TFUSDataset(
            root=root, skull_ids=split["train"], frequencies=frequencies,
            skull_modality=skull_modality, positions=positions,
            roi_size_vox=roi_size_vox, patch_vox=patch_vox,
            domain_size_vox=domain_size_vox, voxel_mm=voxel_mm,
            normalize_field=normalize_field, downsample=downsample,
            augment_mirror=False,
            sample_filter=heldout_filter,
        )
        train_heldout_loader = DataLoader(
            train_heldout_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=persistent, drop_last=False,
        )
 
    return train_loader, val_loader, test_loader, train_heldout_loader
 
 
def describe_loaders(loaders: tuple) -> str:
    """Pretty-print sample counts and skull IDs for each split. Sanity helper.
    Accepts the (train, val, test) tuple or the 4-tuple with train_heldout."""
    names = ("train", "val", "test", "train_heldout")
    lines = []
    for name, loader in zip(names, loaders):
        if loader is None:
            continue
        ds: TFUSDataset = loader.dataset                   # type: ignore[assignment]
        lines.append(
            f"{name:13s}: {len(ds):>5d} samples  "
            f"({len(ds.skull_ids)} skulls × {len(ds.positions)} positions × "
            f"{len(ds.frequencies)} freqs)  "
            f"skulls={ds.skull_ids}"
        )
    return "\n".join(lines)