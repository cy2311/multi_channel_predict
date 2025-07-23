# beadlib: reusable utilities for bead emitter processing

import os
import json
from typing import List, Tuple

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, gaussian_filter
from scipy.spatial import KDTree
from skimage.feature import peak_local_max

# ---------------------------------
# Public constants
# ---------------------------------

PATCH_Z: int = 201
PATCH_XY: int = 25
UPSAMPLE_SIZE: int = 128
VISUAL_FRAME_OFFSETS = [0, 10, 20, 30, 40]  # indices 0-based (display as 1,11,...)

__all__ = [
    "PATCH_Z",
    "PATCH_XY",
    "UPSAMPLE_SIZE",
    "VISUAL_FRAME_OFFSETS",
    "load_config",
    "load_tiff_stack",
    "detect_emitters",
    "extract_patch",
    "photons_per_pixel",
    "compute_photon_stats",
    "normalize_patch",
    "upsample_xy",
    "select_frames",
    "visualize_frames",
    "filter_isolated_emitters",
]

# ---------------------------------
# I/O helpers
# ---------------------------------

def load_config(path: str) -> dict:
    """Load a JSON configuration file."""
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg


def load_tiff_stack(path: str) -> np.ndarray:
    """Memory-map OME-Tiff stack to avoid loading everything into RAM.

    Returns a (Z, Y, X) ndarray.
    """
    with tiff.TiffFile(path) as tif:
        arr = tif.asarray(out="memmap")  # lazy mem-map
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D stack (Z,Y,X) but got shape {arr.shape}")
    return arr

# ---------------------------------
# Peak detection & geometry helpers
# ---------------------------------

def detect_emitters(
    stack: np.ndarray,
    n_emitters: int = 100,
    min_distance: int = 5,
) -> List[Tuple[int, int]]:
    """Detect candidate emitter (y,x) positions using a MIP + local maxima."""

    mip = stack.max(axis=0)
    mip_smooth = gaussian_filter(mip.astype(float), sigma=1)

    coords = peak_local_max(
        mip_smooth,
        min_distance=min_distance,
        threshold_abs=np.percentile(mip_smooth, 99),  # type: ignore[arg-type]
        num_peaks=n_emitters,
    )
    return [tuple(c) for c in coords]  # row,col → y,x


def filter_isolated_emitters(
    coords: List[Tuple[int, int]],
    patch_xy: int = PATCH_XY,
) -> List[Tuple[int, int]]:
    """Keep only emitters that have *no* neighbor within a patch-sized window.

    If another emitter lies within ``patch_xy`` (Chebyshev distance \<= half patch)
    both emitters are discarded. This follows the requirement that a 25×25 field
    must contain at most one emitter.
    """

    if len(coords) < 2:
        return coords  # nothing to compare with

    coords_arr = np.array(coords)
    tree = KDTree(coords_arr)
    half_w = patch_xy // 2  # radius in pixels

    isolated: List[Tuple[int, int]] = []
    for i, pt in enumerate(coords_arr):
        # query_ball_point uses Euclidean distance (p=2)
        nbrs = tree.query_ball_point(pt, r=half_w, p=2)
        if len(nbrs) == 1:  # only itself
            isolated.append(tuple(pt))
    return isolated

# ---------------------------------
# Patch extraction & processing
# ---------------------------------

def extract_patch(stack: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
    """Extract a (Z, PATCH_XY, PATCH_XY) patch centered at given (y,x)."""
    y, x = center
    half = PATCH_XY // 2
    y_start, y_end = y - half, y + half + 1  # +1 because end is exclusive
    x_start, x_end = x - half, x + half + 1

    if (
        y_start < 0
        or y_end > stack.shape[1]
        or x_start < 0
        or x_end > stack.shape[2]
    ):
        raise ValueError("Emitter too close to border for required patch size")

    patch = stack[:PATCH_Z, y_start:y_end, x_start:x_end]
    if patch.shape != (PATCH_Z, PATCH_XY, PATCH_XY):
        raise ValueError(f"Unexpected patch shape {patch.shape}")
    return patch


# ---------------------------------
# Camera / photon conversion helpers
# ---------------------------------

def photons_per_pixel(
    patch: np.ndarray,
    e_adu: float,
    baseline: float,
    qe: float | None = None,
) -> np.ndarray:
    """Convert ADU values in patch to photon counts per pixel."""

    electrons = (patch.astype(float) - baseline) * e_adu
    electrons[electrons < 0] = 0
    photons = electrons if (qe is None or qe <= 0) else electrons / qe
    return photons


def compute_photon_stats(photons_patch: np.ndarray) -> pd.DataFrame:
    """Return total photons per Z-slice as a pandas DataFrame."""

    totals = photons_patch.sum(axis=(1, 2))
    return pd.DataFrame({
        "slice_index": np.arange(len(totals)),
        "total_photons": totals,
    })

# ---------------------------------
# Normalisation & upsampling
# ---------------------------------

def normalize_patch(patch: np.ndarray) -> np.ndarray:
    """Scale patch voxel values to the [0,1] range (global normalisation)."""

    vmin, vmax = patch.min(), patch.max()
    if vmax == vmin:
        return np.zeros_like(patch, dtype=float)
    return (patch - vmin) / (vmax - vmin)


def upsample_xy(patch: np.ndarray, output_xy: int = UPSAMPLE_SIZE) -> np.ndarray:
    """Upsample XY dimensions of a 3-D patch via cubic interpolation."""

    zoom_factor = output_xy / PATCH_XY
    zooms = (1.0, zoom_factor, zoom_factor)
    return zoom(patch, zoom=zooms, order=3)

# ---------------------------------
# Frame selection & visualisation
# ---------------------------------

def select_frames(
    patch: np.ndarray,
    center: int = 80,
    step: int = 2,
    n_each_side: int = 20,
) -> np.ndarray:
    """Return 41 frames centred around ``center`` with equal ``step`` spacing."""

    indices = center + np.arange(-n_each_side, n_each_side + 1) * step
    if indices.min() < 0 or indices.max() >= patch.shape[0]:
        raise ValueError("Selected frame indices out of bounds")
    return patch[indices]


def visualize_frames(frames: np.ndarray, save_path: str | None = None):
    """Save a figure showing 5 representative frames (1,11,21,31,41)."""

    fig, axes = plt.subplots(1, len(VISUAL_FRAME_OFFSETS), figsize=(15, 3))
    for ax, offset in zip(axes, VISUAL_FRAME_OFFSETS):
        img = frames[offset]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Frame {offset + 1}")
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close(fig) 