#!/usr/bin/env python3
"""Plot Zernike magnitude & phase coefficients.

This script loads the coefficient files produced by `phase_retrieval_gs copy.py` and
creates a two-panel line plot showing the magnitude and phase coefficients.
The resulting figure is saved next to the coefficient files for easy inspection.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration – adjust if your directory structure is different
# -----------------------------------------------------------------------------
DATA_DIR = Path('simulated_data')
MAG_PATH = DATA_DIR / 'zernike_coeff_mag.npy'
PHASE_PATH = DATA_DIR / 'zernike_coeff_phase.npy'
FIG_PATH = DATA_DIR / 'zernike_coefficients.png'


def main() -> None:
    # ---------------------------------------------------------------------
    # Load
    # ---------------------------------------------------------------------
    if not MAG_PATH.exists() or not PHASE_PATH.exists():
        raise FileNotFoundError(
            f"Could not find coefficient files at {MAG_PATH} and {PHASE_PATH}.\n"
            "Make sure you have run the phase-retrieval script first.")

    coeff_mag = np.load(MAG_PATH)
    coeff_phase = np.load(PHASE_PATH)

    if coeff_mag.shape != coeff_phase.shape:
        raise ValueError(
            f"Magnitude and phase coefficient arrays have mismatched shapes: "
            f"{coeff_mag.shape} vs {coeff_phase.shape}")

    x = np.arange(1, len(coeff_mag) + 1)  # 1-based index for readability

    # ---------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(x, coeff_mag, marker='o')
    axes[0].set_ylabel('Magnitude Coefficient')
    axes[0].set_title('Zernike Magnitude Coefficients')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].plot(x, coeff_phase, marker='o', color='tab:red')
    axes[1].set_xlabel('Coefficient Index (1-based)')
    axes[1].set_ylabel('Phase Coefficient')
    axes[1].set_title('Zernike Phase Coefficients')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig.savefig(FIG_PATH, dpi=300)
    print(f"Figure saved to {FIG_PATH.resolve()}")


if __name__ == '__main__':
    main()
