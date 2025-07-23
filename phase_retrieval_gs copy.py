import os
import glob
import math
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.optimize import curve_fit  # for Gaussian fit
from scipy.ndimage import gaussian_filter
from numpy.linalg import lstsq

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CFG_PATH = os.path.join('configs', 'default_config.json')
PSF_NPY_PATH = os.path.join('simulated_data', 'patch_03_upsampled.npy')  # use patch 01
ZERNIKE_DIR = os.path.join('simulated_data')  # contains zernike_*.npy files
OUTPUT_DIR = 'simulated_data'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

ITER_MAX = 100
NCC_THRESHOLD = 0.7
SIGMA_OTF = 2  # Gaussian blur sigma for final visualization

# Slice selection parameters (match pre-processing)
CENTER_SLICE = 80
STEP = 2
N_EACH_SIDE = 20  # 20 on each side + center => 41 slices

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Compute NCC between two images (same shape)."""
    a_flat = a.ravel().astype(float)
    b_flat = b.ravel().astype(float)
    a_flat -= a_flat.mean()
    b_flat -= b_flat.mean()
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def load_zernike_basis() -> np.ndarray:
    """Load 21 Zernike basis (128x128) previously computed.

    Tries two locations:
    1. directly under simulated_data/
    2. under simulated_data/zernike_polynomials/
    """
    pattern_root = os.path.join(ZERNIKE_DIR, 'zernike_*_n*_m*.npy')
    pattern_sub  = os.path.join(ZERNIKE_DIR, 'zernike_polynomials', 'zernike_*_n*_m*.npy')
    files = sorted(glob.glob(pattern_root))
    if len(files) < 21:
        files = sorted(glob.glob(pattern_sub))
    if len(files) < 21:
        raise RuntimeError('Expected 21 Zernike basis files (npy) in simulated_data or simulated_data/zernike_polynomials.')
    basis = np.stack([np.load(fp) for fp in files[:21]], axis=0)  # (21, 128, 128)
    return basis


def unwrap_phase_2d(phase: np.ndarray) -> np.ndarray:
    """Simple 2-D phase unwrapping."""
    return np.unwrap(np.unwrap(phase, axis=0), axis=1)

# -----------------------------------------------------------------------------
# Helper functions for OTF Gaussian low-pass (INSPR style)
# -----------------------------------------------------------------------------

def _psf_to_otf(psf: np.ndarray) -> np.ndarray:
    """Convert spatial PSF (shifted) to complex OTF (shifted)."""
    return fftshift(ifft2(ifftshift(psf)))  # type: ignore[arg-type]


def _gauss2d(flat_coords, amp, sigma, bg):
    """Isotropic 2-D Gaussian used for curve_fit (input flattened)."""
    x, y = flat_coords  # type: ignore[arg-type]
    r2 = x ** 2 + y ** 2
    return amp * np.exp(-r2 / (2.0 * sigma ** 2)) + bg  # type: ignore[return-value]


def _fit_gaussian_ratio(ratio_crop: np.ndarray) -> tuple[float, float, float]:
    """Fit cropped ratio OTF to Gaussian, return (amp, sigma_px, bg)."""
    n = ratio_crop.shape[0]
    xx, yy = np.meshgrid(np.arange(n) - n // 2, np.arange(n) - n // 2)
    popt, _ = curve_fit(  # type: ignore[arg-type]
        _gauss2d,
        (xx.ravel(), yy.ravel()),
        ratio_crop.ravel(),
        p0=(1.0, 3.0, 0.0),
        bounds=([0.0, 0.3, -np.inf], [10.0, 10.0, np.inf]),
    )
    amp, sigma, bg = float(popt[0]), float(popt[1]), float(popt[2])
    return (amp, sigma, bg)


def _build_gauss_filter(N: int, sigma_px: float, amp: float = 1.0, bg: float = 0.0) -> np.ndarray:
    """Return N×N isotropic Gaussian filter centered at DC."""
    xx, yy = np.meshgrid(np.arange(N) - N // 2, np.arange(N) - N // 2)
    return amp * np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma_px ** 2)) + bg  # type: ignore[arg-type]

OTF_RATIO_SIZE = 60  # crop size used for Gaussian fitting (pixels)
_EPS = 1e-12

# -----------------------------------------------------------------------------
# Load configuration & PSF measurements
# -----------------------------------------------------------------------------
print('Loading configuration...')
with open(CFG_PATH, 'r') as f:
    cfg = json.load(f)

wavelength_nm = cfg['optical']['wavelength_nm']  # 660
# Anisotropic pixel sizes (nm) read from config
pixel_size_nm_x = cfg['optical']['pixel_size_nm_x']
pixel_size_nm_y = cfg['optical']['pixel_size_nm_y']
NA = cfg['optical']['NA']

wavelength = wavelength_nm * 1e-9  # m
pixel_size_x = pixel_size_nm_x * 1e-9  # m
pixel_size_y = pixel_size_nm_y * 1e-9  # m

print('Loading measured PSF patch...')
meas_patch = np.load(PSF_NPY_PATH)  # shape (201,128,128)
print(f'Loaded patch shape: {meas_patch.shape}')

# Select 41 frames
indices = CENTER_SLICE + np.arange(-N_EACH_SIDE, N_EACH_SIDE + 1) * STEP
meas_psfs = meas_patch[indices]
print(f'Selected {meas_psfs.shape[0]} slices for phase retrieval.')

# Precompute measured amplitudes (sqrt of intensity)
meas_ampls = np.sqrt(np.clip(meas_psfs, 0, None))

# -----------------------------------------------------------------------------
# Frequency grid for pupil plane operations
# -----------------------------------------------------------------------------
N = meas_psfs.shape[1]  # 128
fx = np.fft.fftfreq(N, d=pixel_size_x)  # cycles per meter (x-axis)
fy = np.fft.fftfreq(N, d=pixel_size_y)  # cycles per meter (y-axis)
FY, FX = np.meshgrid(fy, fx, indexing='ij')
RHO2 = FX ** 2 + FY ** 2  # radial spatial frequency squared

# Maximum frequency defined by NA
f_max = NA / wavelength  # cutoff cycles per meter
pupil_mask = (np.sqrt(RHO2) <= f_max).astype(float)

# Defocus phase factor: 
#   H(z) = exp(i * pi * wavelength * z * (FX^2 + FY^2))  [Fresnel approx]
# Original scalar-pixel-size formula removed after introducing anisotropic pixels
# z_list_m = (indices - CENTER_SLICE) * STEP * (pixel_size_nm * 1e-9) * (30 / (STEP * pixel_size_nm))
# However user said slice spacing is 30 nm. So simpler:
z_list_m = (np.arange(-N_EACH_SIDE, N_EACH_SIDE + 1) * 30e-9)

defocus_phases = [np.exp(1j * math.pi * wavelength * z * RHO2) for z in z_list_m]

# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------
P = pupil_mask * np.exp(1j * np.zeros((N, N)))  # initial pupil (unit amplitude, zero phase)

# -----------------------------------------------------------------------------
# Main GS iteration loop
# -----------------------------------------------------------------------------
print('Starting Gerchberg–Saxton iterations...')
for it in range(ITER_MAX):
    P_estimates = []
    preds_psf = []

    for z_idx, H_z in enumerate(defocus_phases):
        # Forward propagation: pupil -> image (via inverse FFT)
        Pz = P * H_z
        field_img = np.asarray(ifft2(ifftshift(Pz)))
        # Replace amplitude with measured amplitude
        new_field_img = meas_ampls[z_idx] * np.exp(1j * np.angle(np.asarray(field_img)))  # type: ignore[arg-type]
        # Back propagate
        Pz_new = np.asarray(fftshift(fft2(new_field_img)))  # type: ignore[arg-type]
        # Remove defocus
        P_est = Pz_new / H_z  # type: ignore[arg-type]
        P_estimates.append(P_est)

        # Save predicted PSF for similarity check
        preds_psf.append(np.abs(field_img) ** 2)

    # Compute average NCC across slices between preds_psf and meas_psfs
    ncc_vals = [normalized_cross_correlation(preds_psf[i], meas_psfs[i]) for i in range(len(preds_psf))]
    mean_ncc = float(np.mean(ncc_vals))
    print(f'Iteration {it+1:02d}: mean NCC = {mean_ncc:.4f}')

    # Update pupil by averaging complex estimates & reapply mask
    P = np.mean(P_estimates, axis=0)
    # Enforce pupil amplitude to ones inside mask (or keep magnitude?)
    P = pupil_mask * np.exp(1j * np.angle(np.asarray(P)))  # type: ignore[arg-type]

    if mean_ncc >= NCC_THRESHOLD:
        print('Stopping early – similarity threshold reached.')
        break

# -----------------------------------------------------------------------------
# Final predicted PSF stack
# -----------------------------------------------------------------------------
print('Generating final predicted PSFs...')
pred_psfs_final = []
for H_z in defocus_phases:
    field_img = np.asarray(ifft2(ifftshift(P * H_z)))
    pred_psfs_final.append(np.abs(field_img) ** 2)
final_psf_stack = np.stack(pred_psfs_final)

# -----------------------------------------------------------------------------
# OTF Gaussian low-pass regularization (INSPR-style)
# -----------------------------------------------------------------------------
print('Applying OTF Gaussian low-pass regularization...')

# Choose central slice (index in meas_psfs list)
center_idx = N_EACH_SIDE

meas_center = meas_psfs[center_idx]
pred_center = final_psf_stack[center_idx]

# 1. Compute magnitude ratio of OTFs
otf_meas = _psf_to_otf(meas_center)
otf_pred = _psf_to_otf(pred_center)
ratio_mag = np.abs(otf_meas) / (np.abs(otf_pred) + _EPS)

# 2. Crop central window and fit isotropic Gaussian
R = ratio_mag.shape[0]
half = OTF_RATIO_SIZE // 2
crop = ratio_mag[R // 2 - half : R // 2 + half, R // 2 - half : R // 2 + half]
amp_g, sigma_px, bg_g = _fit_gaussian_ratio(crop)
print(f'  Fitted Gaussian sigma = {sigma_px:.2f} px')

# 3. Build full-size Gaussian filter
gauss_filter = _build_gauss_filter(R, sigma_px, amp=amp_g, bg=bg_g)

# 4. Apply filter to every predicted OTF and convert back to PSF
psf_lpf_list: list[np.ndarray] = []
for psf_z in final_psf_stack:
    otf_z = _psf_to_otf(psf_z)
    otf_filt = otf_z * gauss_filter
    psf_mod = np.abs(fft2(ifftshift(otf_filt))) ** 2  # type: ignore[arg-type]
    psf_mod /= psf_mod.sum()
    psf_lpf_list.append(psf_mod.astype(np.float32))

mod_psf_stack = np.stack(psf_lpf_list)

# Optionally save for later use
np.save(os.path.join(OUTPUT_DIR, 'psf_stack_otf_lpf.npy'), mod_psf_stack)
print('  Saved LPF-corrected PSF stack to psf_stack_otf_lpf.npy')

# -----------------------------------------------------------------------------
# Zernike decomposition
# -----------------------------------------------------------------------------
print('Performing Zernike decomposition...')
Z_basis = load_zernike_basis()  # (21,128,128)
inside_aperture = pupil_mask.astype(bool)

# Prepare matrices for least squares
basis_vectors = Z_basis[:, inside_aperture].T  # shape (Npix_in, 21)

# Magnitude coefficients
mag_target = np.abs(P)[inside_aperture]
coeff_mag, *_ = lstsq(basis_vectors, mag_target, rcond=None)

# Enforce RMS = 1 for magnitude coefficients
mag_rms = np.linalg.norm(coeff_mag)
if mag_rms != 0:
    coeff_mag = coeff_mag / mag_rms

# Phase coefficients (use unwrapped phase)
phase_target = unwrap_phase_2d(np.angle(np.asarray(P)))[inside_aperture]  # type: ignore[arg-type]
coeff_phase, *_ = lstsq(basis_vectors, phase_target, rcond=None)

# Enforce RMS = 1 for phase coefficients
phase_rms = np.linalg.norm(coeff_phase)
if phase_rms != 0:
    coeff_phase = coeff_phase / phase_rms

np.save(os.path.join(OUTPUT_DIR, 'zernike_coeff_mag.npy'), coeff_mag)
np.save(os.path.join(OUTPUT_DIR, 'zernike_coeff_phase.npy'), coeff_phase)
print('Saved Zernike coefficients.')

# -----------------------------------------------------------------------------
# Visualization – compare original & final central PSF
# -----------------------------------------------------------------------------
orig_center = meas_psfs[N_EACH_SIDE]  # central slice index in meas_psfs list
pred_center = final_psf_stack[N_EACH_SIDE]

# Apply Gaussian blur in Fourier domain equivalent (here just spatial domain)
orig_blur = gaussian_filter(np.asarray(orig_center, dtype=float), sigma=SIGMA_OTF)  # type: ignore[arg-type]
pred_blur = gaussian_filter(np.asarray(pred_center, dtype=float), sigma=SIGMA_OTF)  # type: ignore[arg-type]

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, img, title in zip(axes, [orig_blur, pred_blur], ['Original PSF', 'Predicted PSF (blurred)']):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, 'psf_comparison.png')
plt.savefig(fig_path, dpi=300)
plt.close()
print(f'PSF comparison figure saved to {fig_path}')

# -----------------------------------------------------------------------------
# Visualization – 5-frame comparison (original vs predicted)
# -----------------------------------------------------------------------------
VIS_OFFSETS = [0, 10, 20, 30, 40]
fig, axes = plt.subplots(len(VIS_OFFSETS), 2, figsize=(6, 12))
for row, idx in enumerate(VIS_OFFSETS):
    orig_b = gaussian_filter(np.asarray(meas_psfs[idx], dtype=float), sigma=SIGMA_OTF)  # type: ignore
    pred_b = gaussian_filter(np.asarray(final_psf_stack[idx], dtype=float), sigma=SIGMA_OTF)  # type: ignore
    axes[row, 0].imshow(orig_b, cmap='gray')
    axes[row, 0].set_title(f'Orig frame {idx+1}')
    axes[row, 0].axis('off')
    axes[row, 1].imshow(pred_b, cmap='gray')
    axes[row, 1].set_title(f'Pred frame {idx+1}')
    axes[row, 1].axis('off')
plt.tight_layout()
fig_path2 = os.path.join(OUTPUT_DIR, 'psf_comparison_five.png')
plt.savefig(fig_path2, dpi=300)
plt.close()
print(f'5-frame comparison figure saved to {fig_path2}')

print('Phase retrieval completed.')
