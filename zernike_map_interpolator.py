#!/usr/bin/env python3
"""zernike_map_interpolator.py

Generate full-field (1200×1200) Zernike coefficient maps from emitter-wise
phase coefficients stored in ``simulated_data/patches.h5``. For each of the 21
Zernike orders, a cubic-spline (scipy griddata ``method='cubic'``) interpolation
is performed using the (x,y) pixel coordinates of isolated emitters. The 21 maps
are visualised in a 7×3 panel figure with a **cool-warm** colour scale.
"""
# mypy: ignore-errors
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import griddata

# -----------------------------
# Configuration
# -----------------------------
H5_PATH = Path('simulated_data') / 'patches.h5'
ROI_YX = (1200, 1200)  # (height, width)
FIG_PATH = Path('simulated_data') / 'zernike_phase_maps.png'

# colormap
CMAP = 'coolwarm'


def load_data():
    if not H5_PATH.exists():
        raise FileNotFoundError(f'HDF5 not found: {H5_PATH}')
    with h5py.File(H5_PATH, 'r') as f:
        grp = f['zernike']
        coeff_phase = np.asarray(grp['coeff_phase'])  # (N,21) #type: ignore[attr-defined]
        coords = np.asarray(f['coords'])  # (N,2) (y,x) #type: ignore[attr-defined]
    return coords, coeff_phase


def build_grid_maps(coords: np.ndarray, coeff_phase: np.ndarray) -> np.ndarray:
    """Return interpolated maps array of shape (21, H, W)."""
    N = coords.shape[0]
    if N < 4:
        raise RuntimeError('Too few points for interpolation')
    y_coords, x_coords = coords[:, 0], coords[:, 1]
    H, W = ROI_YX
    xi = np.linspace(0, W - 1, W)
    yi = np.linspace(0, H - 1, H)
    XI, YI = np.meshgrid(xi, yi)

    maps = np.empty((21, H, W), dtype=np.float32)
    points = np.column_stack([x_coords, y_coords])

    for j in range(21):
        values = coeff_phase[:, j]
        grid = griddata(points, values, (XI, YI), method='cubic')
        # fall back for NaNs using nearest
        nan_mask = np.isnan(grid)
        if np.any(nan_mask):
            grid[nan_mask] = griddata(points, values, (XI[nan_mask], YI[nan_mask]), method='nearest')
        maps[j] = grid.astype(np.float32)
    return maps


def plot_maps(maps: np.ndarray):
    n_rows, n_cols = 7, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))

    idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            if idx < 21:
                data = maps[idx]
                vmax_local = np.nanmax(np.abs(data))
                im = ax.imshow(data, cmap=CMAP, origin='lower', vmin=-vmax_local, vmax=vmax_local)
                ax.set_title(f'Z{idx+1}')
                ax.axis('off')
                # individual colour bar
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.01)
            else:
                ax.remove()
            idx += 1
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=300, bbox_inches='tight')
    print(f'Figure saved to {FIG_PATH.resolve()}')


def main():
    coords, coeff_phase = load_data()
    maps = build_grid_maps(coords, coeff_phase)

    # --------------------------------------------------------------
    # Persist maps into HDF5 (/zernike_maps/phase)
    # --------------------------------------------------------------
    with h5py.File(H5_PATH, 'r+') as f:
        zm_grp = f.require_group('z_maps')
        if 'phase' in zm_grp:
            del zm_grp['phase']  # overwrite if exists
        zm_grp.create_dataset('phase', data=maps, compression='gzip')
        print('Saved interpolated maps to /zernike_maps/phase in patches.h5')

    plot_maps(maps)


if __name__ == '__main__':
    main()
