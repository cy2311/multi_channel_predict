#!/usr/bin/env python3
"""Plot Zernike magnitude & phase coefficients of random emitters.

Reads coefficients stored in ``simulated_data/patches.h5`` (/zernike group).
Randomly selects up to ``N_SAMPLES`` valid emitters (mean_ncc not NaN) and
overlays their coefficient curves on two subplots (magnitude & phase).
"""
# mypy: ignore-errors
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import h5py

# -----------------------------------------------------------------------------
# Configuration – adjust if your directory structure is different
# -----------------------------------------------------------------------------
H5_PATH = Path('simulated_data') / 'patches.h5'
FIG_PATH = Path('simulated_data') / 'zernike_coefficients_random10.png'
N_SAMPLES = 50


def main() -> None:
    # ---------------------------------------------------------------------
    # Load coefficients from HDF5
    # ---------------------------------------------------------------------
    if not H5_PATH.exists():
        raise FileNotFoundError(f"HDF5 file not found: {H5_PATH}")

    with h5py.File(H5_PATH, 'r') as f:
        if 'zernike' not in f:
            raise KeyError("/zernike group not found in patches.h5 – run retrieval first")
        grp = f['zernike']
        coeff_mag_set = grp['coeff_mag'] #type: ignore[attr-defined]
        coeff_phase_set = grp['coeff_phase'] #type: ignore[attr-defined]
        mean_ncc = grp['mean_ncc'] #type: ignore[attr-defined]

        valid_idx = np.where(~np.isnan(mean_ncc))[0] #type: ignore[arg-type]
        if valid_idx.size == 0:
            raise RuntimeError('No valid emitters with NCC >= threshold.')

        sample_idx = random.sample(list(valid_idx), k=min(N_SAMPLES, valid_idx.size))
        sample_idx_sorted = np.sort(sample_idx)
        coeff_mag = np.asarray(coeff_mag_set[sample_idx_sorted])  # (k,21)  #type: ignore[attr-defined]
        coeff_phase = np.asarray(coeff_phase_set[sample_idx_sorted]) #type: ignore[attr-defined]

    # restore original random order for plotting labels
    reorder = np.argsort(np.searchsorted(sample_idx_sorted, sample_idx))
    coeff_mag = coeff_mag[reorder]
    coeff_phase = coeff_phase[reorder]
    sample_idx = [sample_idx[i] for i in reorder]

    x = np.arange(1, coeff_mag.shape[1] + 1)  # 1..21

    # ---------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    for i, vec in enumerate(coeff_mag):
        axes[0].plot(x, vec, linewidth=0.2, label=f'Emitter {sample_idx[i]}')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_title('Zernike Magnitude Coefficients (Random)')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    for i, vec in enumerate(coeff_phase):
        axes[1].plot(x, vec, linewidth=0.2, label=f'Emitter {sample_idx[i]}')
    axes[1].set_xlabel('Coefficient Index (1-based)')
    axes[1].set_ylabel('Phase')
    axes[1].set_title('Zernike Phase Coefficients (Random)')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # Place a single consolidated legend outside the plot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5),
               fontsize='small')

    plt.tight_layout()
    fig.savefig(FIG_PATH, dpi=300)
    print(f"Figure saved to {FIG_PATH.resolve()}")


if __name__ == '__main__':
    main()
