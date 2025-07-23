import os
import math
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# -----------------------------------------------------------------------------
# Load pixel size (anisotropic) from configuration
# -----------------------------------------------------------------------------
CFG_PATH = os.path.join('configs', 'default_config copy.json')

with open(CFG_PATH, 'r') as _f:
    _cfg = json.load(_f)

# Expect the optical section to contain pixel_size_nm_x / y
_optical_cfg = _cfg.get('optical', {})
PIXEL_SIZE_NM_X = _optical_cfg['pixel_size_nm_x']
PIXEL_SIZE_NM_Y = _optical_cfg['pixel_size_nm_y']

# Convert to metres for calculation (physical grid)
PIXEL_SIZE_X = PIXEL_SIZE_NM_X * 1e-9  # m
PIXEL_SIZE_Y = PIXEL_SIZE_NM_Y * 1e-9  # m

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
N_PIXELS = 128           # Grid size for each polynomial (square)
MAX_L   = 21            # Total number of Wyant-indexed polynomials to generate
OUTPUT_DIR = 'simulated_data/zernike_polynomials'  # Destination directory for .npy and .png files
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Helper functions for Wyant ordering
# -----------------------------------------------------------------------------

def wyant_l_to_nm(l: int) -> Tuple[int, int]:
    """Convert Wyant index l (0-based) to (n, m)."""
    n = int(math.floor(math.sqrt(l)))
    rem = l - n * n
    mm = math.ceil((2 * n - rem) / 2)  # magnitude of m
    if rem % 2 == 0:
        m = mm
    else:
        m = -mm
    return n, m


def radial_poly(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Compute the radial component R_n^|m|(rho)."""
    m = abs(m)
    if (n - m) % 2 != 0:
        # For invalid parity, radial component is zero everywhere
        return np.zeros_like(rho)
    R = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        coeff = ((-1) ** k) * math.factorial(n - k) / (
            math.factorial(k) * math.factorial((n + m) // 2 - k) * math.factorial((n - m) // 2 - k)
        )
        R += coeff * rho ** (n - 2 * k)
    return R


def zernike_nm(n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Return the (unnormalized) Zernike polynomial Z_n^m over rho<=1."""
    R = radial_poly(n, m, rho)
    if m > 0:
        Z = R * np.cos(m * theta)
    elif m < 0:
        Z = R * np.sin(-m * theta)
    else:
        Z = R  # m == 0
    return Z


# -----------------------------------------------------------------------------
# Grid preparation (physical coordinate aware)
# -----------------------------------------------------------------------------
# Build physical coordinate arrays with anisotropic pixel pitch, centred at 0.
half_idx = (N_PIXELS - 1) / 2.0
x_phys = (np.arange(N_PIXELS) - half_idx) * PIXEL_SIZE_X  # length in metres
y_phys = (np.arange(N_PIXELS) - half_idx) * PIXEL_SIZE_Y

X, Y = np.meshgrid(x_phys, y_phys)

# Normalize physical radius to unit circle inside the rectangular grid
r_phys = np.sqrt(X**2 + Y**2)
r_max = r_phys.max() if r_phys.max() != 0 else 1.0  # avoid div-by-zero

rho = r_phys / r_max
theta = np.arctan2(Y, X)

inside_mask = rho <= 1.0

# Set rho outside unit circle to 0 (values won’t be used)
rho[~inside_mask] = 0

# -----------------------------------------------------------------------------
# Generate and save basis polynomials
# -----------------------------------------------------------------------------

print(f"Generating first {MAX_L} Wyant-indexed Zernike polynomials on a {N_PIXELS}×{N_PIXELS} grid...")
for l in range(MAX_L):
    n, m = wyant_l_to_nm(l)
    Z = zernike_nm(n, m, rho, theta)
    # Zero out values outside aperture for clarity
    Z[~inside_mask] = 0

    # Optional normalization to unit RMS within aperture
    rms = np.sqrt(np.mean(Z[inside_mask] ** 2))
    if rms > 0:
        Z /= rms

    # Save as .npy
    npy_path = os.path.join(OUTPUT_DIR, f'zernike_{l:02d}_n{n}_m{m:+d}.npy')
    np.save(npy_path, Z.astype(np.float32))

    # Visualize and save figure
    vmax = np.max(np.abs(Z))
    plt.figure(figsize=(3, 3))
    plt.imshow(Z, cmap='seismic', vmin=-vmax, vmax=vmax)
    plt.axis('off')
    plt.title(f'l={l}\n(n={n}, m={m:+d})')
    png_path = os.path.join(OUTPUT_DIR, f'zernike_{l:02d}_n{n}_m{m:+d}.png')
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'  Saved l={l:02d} (n={n}, m={m:+d}) to {npy_path} & {png_path}')

print('Done.')
