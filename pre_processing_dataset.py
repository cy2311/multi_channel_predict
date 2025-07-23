import os
import json
import random
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, gaussian_filter
from skimage.feature import peak_local_max

# ------------------------------
# Configuration & constants
# ------------------------------
CONFIG_PATH = os.path.join('configs', 'default_config.json')
DATA_DIR = os.path.join('beads', 'spool_100mW_30ms_3D_1_2')
OME_TIFF_NAME = 'spool_100mW_30ms_3D_1_2_MMStack_Default.ome.tif'
PATCH_Z = 201
PATCH_XY = 25
UPSAMPLE_SIZE = 128
VISUAL_FRAME_OFFSETS = [0, 10, 20, 30, 40]  # corresponding to 1,11,21,31,41 (1-based)


def load_config(path: str) -> dict:
    """Load the simulation / acquisition config."""
    with open(path, 'r') as f:
        cfg = json.load(f)
    return cfg


def load_tiff_stack(path: str) -> np.ndarray:
    """Memory-map OME-Tiff stack to avoid loading everything into RAM."""
    with tiff.TiffFile(path) as tif:
        # out='memmap' returns numpy.memmap – behaves like ndarray but is lazy
        arr = tif.asarray(out='memmap')
    # Ensure the array is (Z, Y, X). For many OME-Tiff acquisitions the first
    # axis can be time or channel; we squeeze singleton dims.
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D stack (Z,Y,X) but got shape {arr.shape}")
    return arr


def detect_emitters(stack: np.ndarray, n_emitters: int = 100, min_distance: int = 5) -> list[tuple[int, int]]:
    """Detect candidate emitter (x,y) positions using a MIP + local maxima.

    Parameters
    ----------
    stack : np.ndarray
        3-D volume (Z,Y,X).
    n_emitters : int
        Maximum number of emitters to keep.
    min_distance : int
        Minimum distance between peaks.

    Returns
    -------
    List[(y,x)]  coordinates sorted by peak intensity.
    """
    # Maximum intensity projection along Z
    mip = stack.max(axis=0)
    # Smooth to suppress pixel noise
    mip_smooth = gaussian_filter(mip.astype(float), sigma=1)
    # Use skimage to locate peaks
    coords = peak_local_max(
        mip_smooth,
        min_distance=min_distance,
        threshold_abs=np.percentile(mip_smooth, 99),  # type: ignore[arg-type]
        num_peaks=n_emitters,
    )
    # peak_local_max returns (row,col)
    return [tuple(coord) for coord in coords]


def extract_patch(stack: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    """Extract a (Z, PATCH_XY, PATCH_XY) patch centered at given (y,x)."""
    y, x = center
    half = PATCH_XY // 2
    y_start, y_end = y - half, y + half + 1  # +1 because end is exclusive
    x_start, x_end = x - half, x + half + 1
    # Guard against border cases
    if y_start < 0 or y_end > stack.shape[1] or x_start < 0 or x_end > stack.shape[2]:
        raise ValueError("Emitter too close to border for required patch size")
    patch = stack[:PATCH_Z, y_start:y_end, x_start:x_end]
    if patch.shape != (PATCH_Z, PATCH_XY, PATCH_XY):
        raise ValueError(f"Unexpected patch shape {patch.shape}")
    return patch


def photons_per_pixel(patch: np.ndarray, e_adu: float, baseline: float, qe: float | None = None) -> np.ndarray:
    """Convert ADU values in patch to photon counts per pixel."""
    # Subtract camera baseline
    electrons = (patch.astype(float) - baseline) * e_adu
    electrons[electrons < 0] = 0  # clip negative
    if qe is not None and qe > 0:
        photons = electrons / qe
    else:
        photons = electrons
    return photons


def compute_photon_stats(photons_patch: np.ndarray) -> pd.DataFrame:
    """Return total photons per Z-slice as DataFrame."""
    totals = photons_patch.sum(axis=(1, 2))
    df = pd.DataFrame({
        'slice_index': np.arange(len(totals)),
        'total_photons': totals,
    })
    return df


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    """Normalize patch to [0,1] globally (across all voxels)."""
    vmin = patch.min()
    vmax = patch.max()
    if vmax == vmin:
        return np.zeros_like(patch, dtype=float)
    return (patch - vmin) / (vmax - vmin)


def upsample_xy(patch: np.ndarray, output_xy: int = UPSAMPLE_SIZE) -> np.ndarray:
    """Upsample XY dimensions of 3-D patch to (output_xy, output_xy)."""
    zoom_factor = output_xy / PATCH_XY
    # zoom expects sequence per dimension: (Z, Y, X)
    zooms = (1.0, zoom_factor, zoom_factor)
    upsampled = zoom(patch, zoom=zooms, order=3)  # cubic
    return upsampled


def select_frames(patch: np.ndarray, center: int = 80, step: int = 2, n_each_side: int = 20) -> np.ndarray:
    """Select frames around center with fixed step."""
    indices = center + np.arange(-n_each_side, n_each_side + 1) * step
    if indices.min() < 0 or indices.max() >= patch.shape[0]:
        raise ValueError("Selected frame indices out of bounds")
    return patch[indices]


def visualize_frames(frames: np.ndarray, save_path: str | None = None):
    """Visualize specific frames (1,11,21,31,41)."""
    fig, axes = plt.subplots(1, len(VISUAL_FRAME_OFFSETS), figsize=(15, 3))
    for ax, offset in zip(axes, VISUAL_FRAME_OFFSETS):
        img = frames[offset]
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Frame {offset + 1}')
        ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


# ------------------------------
# Main routine
# ------------------------------

def main():
    cfg = load_config(CONFIG_PATH)
    cam_cfg = cfg['camera']

    # Load stack
    stack_path = os.path.join(DATA_DIR, OME_TIFF_NAME)
    print(f'Loading stack: {stack_path}')
    stack = load_tiff_stack(stack_path)
    print(f'Stack shape: {stack.shape}')

    # Detect emitters
    emitter_coords = detect_emitters(stack, n_emitters=cfg.get('emitter', {}).get('n_emitters', 100))
    if not emitter_coords:
        raise RuntimeError('No emitters detected')
    print(f'Detected {len(emitter_coords)} emitters (candidates)')

    # Randomly pick 5 emitters
    random.seed(cfg['simulation'].get('random_seed', 42))
    selected_coords = random.sample(emitter_coords, k=min(5, len(emitter_coords)))
    print(f'Selected coordinates for patches: {selected_coords}')

    # Output directory
    out_dir = cfg['simulation'].get('output_dir', 'processed_patches')
    os.makedirs(out_dir, exist_ok=True)

    e_adu = cam_cfg['e_adu']
    baseline = cam_cfg['baseline']
    qe = cam_cfg.get('qe', None)

    for idx, coord in enumerate(selected_coords, start=1):
        print(f'Processing patch {idx} at {coord}')
        patch = extract_patch(stack, coord)

        # Convert to photons
        photons_patch = photons_per_pixel(patch, e_adu=e_adu, baseline=baseline, qe=qe)

        # Photon statistics per slice
        df_stats = compute_photon_stats(photons_patch)
        csv_path = os.path.join(out_dir, f'patch_{idx:02d}_photon_stats.csv')
        df_stats.to_csv(csv_path, index=False)
        print(f'Photon statistics saved to {csv_path}')

        # Normalize and upsample
        patch_norm = normalize_patch(photons_patch)
        patch_up = upsample_xy(patch_norm)

        # Save full upsampled patch
        npy_path = os.path.join(out_dir, f'patch_{idx:02d}_upsampled.npy')
        np.save(npy_path, patch_up.astype(np.float32))
        print(f'Upsampled patch saved to {npy_path}')

        # Select 41 frames around frame 80
        frames_selected = select_frames(patch_up, center=80, step=2, n_each_side=20)

        # Visualize 5 frames
        fig_path = os.path.join(out_dir, f'patch_{idx:02d}_frames.png')
        visualize_frames(frames_selected, save_path=fig_path)
        print(f'Visualization saved to {fig_path}')


if __name__ == '__main__':
    main()
