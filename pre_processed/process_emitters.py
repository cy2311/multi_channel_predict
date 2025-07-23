#!/usr/bin/env python3
"""process_emitters.py

Batch-process OME-Tiff stacks to extract patches centred on *isolated* bead
emitters (no other peak inside a 25×25 window), perform camera → photon
conversion, normalise, upsample and persist both data and metadata.

Outputs are written to an output directory defined in the JSON config (default
``processed_patches``) with the following layout::

    processed_patches/
    ├── patches/
    │   ├── patch_001.npy
    │   ├── ...
    ├── photon_stats/
    │   ├── patch_001_photon_stats.csv
    │   ├── ...
    ├── figures/
    │   ├── patch_001_frames.png
    │   ├── emitters_xy.png
    └── emitters_meta.csv

Run simply as:

    python pre_processed/process_emitters.py
"""

from __future__ import annotations

import logging

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py

# Optional PyTorch acceleration
try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover – PyTorch not installed
    TORCH_AVAILABLE = False

# Local library
from beadlib import (
    PATCH_Z,
    PATCH_XY,
    UPSAMPLE_SIZE,
    detect_emitters,
    extract_patch,
    filter_isolated_emitters,
    load_config,
    load_tiff_stack,
    normalize_patch,
    photons_per_pixel,
    select_frames,
    upsample_xy,
    visualize_frames,
)

# -----------------------------------------------------------------------------
# Configuration paths (change as required)
# -----------------------------------------------------------------------------

CONFIG_PATH = Path("configs/default_config.json")
DATA_DIR = Path("beads/spool_100mW_30ms_3D_1_2")
OME_TIFF_NAME = "spool_100mW_30ms_3D_1_2_MMStack_Default.ome.tif"

# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------


def prepare_output_dirs(root: Path) -> tuple[Path, Path, Path]:
    """Return (patches_dir, stats_dir, figures_dir) and ensure they exist."""

    patches_dir = root / "patches"
    stats_dir = root / "photon_stats"
    figures_dir = root / "figures"
    for d in (patches_dir, stats_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)
    return patches_dir, stats_dir, figures_dir


def process_emitters():
    # ---------------------------------------------------------------------
    # Initial setup
    # ---------------------------------------------------------------------

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    cfg = load_config(str(CONFIG_PATH))
    cam_cfg = cfg["camera"]

    stack_path = DATA_DIR / OME_TIFF_NAME
    logging.info("Loading stack from %s", stack_path)
    stack = load_tiff_stack(str(stack_path))
    logging.info("Stack shape: %s", stack.shape)

    # ---------------------------------------------------------------------
    # Emitter detection + isolation filtering
    # ---------------------------------------------------------------------

    n_emitters: int = cfg.get("emitter", {}).get("n_emitters", 100)
    candidates: List[Tuple[int, int]] = detect_emitters(stack, n_emitters=n_emitters)
    logging.info("Detected %d candidate emitters", len(candidates))

    isolated_coords = filter_isolated_emitters(candidates, patch_xy=PATCH_XY)
    logging.info("%d emitters remain after isolation filter", len(isolated_coords))

    # ---------------------------------------------------------------------
    # Output structure
    # ---------------------------------------------------------------------

    out_root = Path(cfg.get("simulation", {}).get("output_dir", "processed_patches"))
    # Ensure root & figures directory exist
    out_root.mkdir(parents=True, exist_ok=True)
    figures_dir = out_root / "figures"
    figures_dir.mkdir(exist_ok=True, parents=True)

    # HDF5 container
    h5_path = out_root / "patches.h5"
    h5_file = h5py.File(h5_path, "w")

    # Create resizable datasets
    d_patches = h5_file.create_dataset(
        "patches",
        shape=(0, PATCH_Z, UPSAMPLE_SIZE, UPSAMPLE_SIZE),
        maxshape=(None, PATCH_Z, UPSAMPLE_SIZE, UPSAMPLE_SIZE),
        dtype="float32",
        compression="gzip",
        chunks=(1, PATCH_Z, UPSAMPLE_SIZE, UPSAMPLE_SIZE),
    )
    d_coords = h5_file.create_dataset(
        "coords",
        shape=(0, 2),
        maxshape=(None, 2),
        dtype="int32",
        compression="gzip",
        chunks=True,
    )
    d_totals = h5_file.create_dataset(
        "photon_totals",
        shape=(0, PATCH_Z),
        maxshape=(None, PATCH_Z),
        dtype="float32",
        compression="gzip",
        chunks=True,
    )

    e_adu: float = cam_cfg["e_adu"]
    baseline: float = cam_cfg["baseline"]
    qe: float | None = cam_cfg.get("qe")

    meta_records: list[dict[str, object]] = []

    # Choose compute device if PyTorch is available
    device = None
    if TORCH_AVAILABLE:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Using PyTorch on %s device", device)

    for idx, coord in enumerate(isolated_coords, start=1):
        logging.info("Processing patch %d at (y=%d, x=%d)", idx, coord[0], coord[1])

        # --------------------------------------------------------------
        # Patch extraction (skip if near border)
        # --------------------------------------------------------------
        try:
            patch = extract_patch(stack, coord)
        except ValueError as err:
            logging.warning("Skipping emitter at %s: %s", coord, err)
            continue

        # --------------------------------------------------------------
        # Camera → photon conversion and statistics (PyTorch accelerated)
        # --------------------------------------------------------------

        if TORCH_AVAILABLE:
            patch_t = torch.as_tensor(patch, dtype=torch.float32, device=device)
            electrons = (patch_t - baseline) * e_adu
            electrons.clamp_(min=0)
            if qe is not None and qe > 0:
                photons_patch_t = electrons / qe
            else:
                photons_patch_t = electrons

            # Photon totals per slice
            totals_np = torch.sum(photons_patch_t, dim=(1, 2)).cpu().numpy()

            # Normalise
            vmin = torch.min(photons_patch_t)
            vmax = torch.max(photons_patch_t)
            if vmax == vmin:
                patch_norm_t = torch.zeros_like(photons_patch_t)
            else:
                patch_norm_t = (photons_patch_t - vmin) / (vmax - vmin)

            # Upsample XY dims with bicubic interpolation
            zoom_factor = UPSAMPLE_SIZE / PATCH_XY
            patch_up_t = F.interpolate(
                patch_norm_t.unsqueeze(0),
                scale_factor=(zoom_factor, zoom_factor),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)

            patch_up = patch_up_t.cpu().numpy().astype(np.float32)

        else:
            photons_patch = photons_per_pixel(
                patch, e_adu=e_adu, baseline=baseline, qe=qe
            )
            totals_np = photons_patch.sum(axis=(1, 2))

            # Normalise & upsample via NumPy/SciPy
            patch_norm = normalize_patch(photons_patch)
            patch_up = upsample_xy(patch_norm)

        # --------------------------------------------------------------
        # Append to H5 datasets
        # --------------------------------------------------------------

        cur_n = d_patches.shape[0]
        d_patches.resize(cur_n + 1, axis=0)
        d_coords.resize(cur_n + 1, axis=0)
        d_totals.resize(cur_n + 1, axis=0)

        d_patches[cur_n] = patch_up.astype(np.float32)
        d_coords[cur_n] = np.array(coord, dtype=np.int32)
        d_totals[cur_n] = totals_np.astype(np.float32)

        # --------------------------------------------------------------
        # Metadata accumulation
        # --------------------------------------------------------------
        meta_records.append({
            "idx": idx,
            "y": coord[0],
            "x": coord[1],
        })

    # ---------------------------------------------------------------------
    # Persist global metadata & scatter plot
    # ---------------------------------------------------------------------

    if not meta_records:
        logging.warning("No valid patches were generated — nothing to save.")
        return

    df_meta = pd.DataFrame(meta_records)
    meta_csv = out_root / "emitters_meta.csv"
    df_meta.to_csv(meta_csv, index=False)
    logging.info("Metadata written to %s", meta_csv)

    # Scatter plot of emitter positions
    plt.figure(figsize=(6, 6))
    plt.scatter(df_meta["x"], df_meta["y"], c="lime", edgecolors="k", s=20)
    plt.gca().invert_yaxis()
    plt.title(f"Isolated Emitters ({len(df_meta)})")
    plt.xlabel("x (pixel)")
    plt.ylabel("y (pixel)")
    plt.tight_layout()
    scatter_path = figures_dir / "emitters_xy.png"
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    logging.info("Global scatter plot saved to %s", scatter_path)

    # --------------------------------------------------------------
    # Visualise 10 random patches (5 frames each)
    # --------------------------------------------------------------

    num_samples = min(10, d_patches.shape[0])
    sample_indices = np.random.choice(d_patches.shape[0], num_samples, replace=False)

    for sel in sample_indices:
        patch_up_np = d_patches[sel]
        frames_sel = select_frames(patch_up_np, center=80, step=2, n_each_side=20)
        fig_path = figures_dir / f"random_patch_{sel+1:03d}_frames.png"
        visualize_frames(frames_sel, save_path=str(fig_path))

    logging.info("Saved figures for %d random patches", num_samples)

    # Close HDF5 file
    h5_file.close()


if __name__ == "__main__":
    process_emitters() 