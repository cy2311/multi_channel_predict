import os
import glob
import json
import random
import argparse
from typing import Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import tifffile as tiff  # for OME-TIFF export
from scipy.fft import ifft2, ifftshift, fftshift  # type: ignore
from skimage.transform import resize  # type: ignore


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def load_config() -> Tuple[float, float, float, float]:
    """Load optical parameters from configs/default_config.json.

    Returns
    -------
    wavelength_nm : float
        Wavelength in nanometres.
    pixel_size_nm_x : float
        Pixel size along *x* in nanometres.
    pixel_size_nm_y : float
        Pixel size along *y* in nanometres.
    NA : float
        Numerical aperture.
    """
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    opt_cfg = cfg["optical"]
    return (
        float(opt_cfg["wavelength_nm"]),
        float(opt_cfg["pixel_size_nm_x"]),
        float(opt_cfg["pixel_size_nm_y"]),
        float(opt_cfg["NA"]),
    )


def load_zernike_basis() -> np.ndarray:
    """Load the 21 × 128 × 128 Zernike basis from .npy files located next to this script."""
    zernike_dir = os.path.join(os.path.dirname(__file__), "..", "simulated_data", "zernike_polynomials")
    pattern = os.path.join(zernike_dir, "zernike_*_n*_m*.npy")
    files = sorted(glob.glob(pattern))
    if len(files) < 21:
        raise RuntimeError("Expected at least 21 Zernike .npy files in the current directory.")
    basis = np.stack([np.load(fp) for fp in files[:21]], axis=0).astype(np.float32)
    return basis  # shape (21, 128, 128)


def construct_pupil(
    coeff_mag: np.ndarray,
    coeff_phase: np.ndarray,
    basis: np.ndarray,
    pupil_mask: np.ndarray,
) -> np.ndarray:
    """Construct complex pupil function from coefficients and basis.

    Parameters
    ----------
    coeff_mag : (21,) ndarray
        Magnitude coefficients.
    coeff_phase : (21,) ndarray
        Phase coefficients (radians).
    basis : (21, N, N) ndarray
        Zernike basis.
    pupil_mask : (N, N) ndarray
        Binary pupil mask (ones inside aperture).
    """
    # Amplitude: 1 + linear combination of basis (clamped to >= 0)
    amplitude = 1.0 + np.sum(coeff_mag[:, None, None] * basis, axis=0)  # type: ignore
    amplitude = np.clip(np.asarray(amplitude), 0.0, np.inf)  # type: ignore

    # Phase: linear combination (radians)
    phase = np.sum(coeff_phase[:, None, None] * basis, axis=0)  # type: ignore[arg-type]

    pupil = amplitude * pupil_mask * np.exp(1j * phase)
    return pupil


def generate_psf(pupil: np.ndarray) -> np.ndarray:
    """Generate intensity PSF (128×128) from pupil function."""
    field = ifft2(ifftshift(pupil))
    psf = np.abs(fftshift(field)) ** 2  # type: ignore[arg-type]  # center bright spot
    psf /= psf.sum()
    return psf.astype(np.float32)


def build_pupil_mask(N: int, pixel_size_x: float, pixel_size_y: float, NA: float, wavelength: float) -> np.ndarray:
    """Return binary pupil mask with cutoff defined by NA."""
    fx = np.fft.fftfreq(N, d=pixel_size_x)
    fy = np.fft.fftfreq(N, d=pixel_size_y)
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    rho = np.sqrt(FX ** 2 + FY ** 2)
    f_max = NA / wavelength
    mask = (rho <= f_max).astype(np.float32)
    return mask


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-frame TIFF from emitter H5 dataset")
    parser.add_argument("--h5", type=str, default=os.path.join(os.path.dirname(__file__), "..", "simulated_data", "emitters_sets_raw.h5"), help="Path to input emitters HDF5 file")
    parser.add_argument("--out", type=str, default="all_frames_1200px.ome.tiff", help="Output OME-TIFF filename")
    args = parser.parse_args()

    # Load configuration parameters
    wavelength_nm, pix_x_nm, pix_y_nm, NA = load_config()
    wavelength_m = wavelength_nm * 1e-9
    pix_x_m = pix_x_nm * 1e-9
    pix_y_m = pix_y_nm * 1e-9

    # Basis & pupil mask
    basis = load_zernike_basis()  # shape (21, 128, 128)
    N = basis.shape[1]
    pupil_mask = build_pupil_mask(N, pix_x_m, pix_y_m, NA, wavelength_m)

    # ------------------------------------------------------------------
    # Load emitter data from HDF5 (first frame active emitters)
    # ------------------------------------------------------------------
    with h5py.File(args.h5, "r") as f:
        frame_ix = np.asarray(f["records/frame_ix"])  # type: ignore[index]
        ids_all = np.asarray(f["records/id"])  # type: ignore[index]

        on_mask = frame_ix == 0  # first frame ON emitters
        emitter_ids = np.asarray(ids_all[on_mask])  # type: ignore[index]

        if emitter_ids.size == 0:  # type: ignore[attr-defined]
            raise RuntimeError("No ON emitters found in frame 0.")

        # Preload global per-emitter data
        coeff_mag_all = np.asarray(f["zernike_coeffs/mag"])  # (Ne,21)
        coeff_phase_all = np.asarray(f["zernike_coeffs/phase"])  # (Ne,21)
        emit_xyz_all = np.asarray(f["emitters/xyz"])  # (Ne,3)

        # Records per frame
        ids_rec = ids_all
        xyz_rec = np.asarray(f["records/xyz"])  # (Nr,3)

    # Frequency grid for defocus
    fx = np.fft.fftfreq(N, d=pix_x_m)
    fy = np.fft.fftfreq(N, d=pix_y_m)
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    RHO2 = FX ** 2 + FY ** 2

    # ------------------------------------------------------------------
    # Iterate over unique frames and build low-res images stack
    # ------------------------------------------------------------------

    unique_frames = np.unique(frame_ix)
    low_imgs = []

    HR_SIZE = 6144
    ROI_SIZE = 1200.0
    UPSCALE = HR_SIZE / ROI_SIZE
    half_psf = N // 2

    for fr in unique_frames:
        on_mask = frame_ix == fr
        chosen_ids = ids_rec[on_mask]
        if chosen_ids.size == 0:
            # No active emitters in this frame – keep blank
            low_imgs.append(np.zeros((int(ROI_SIZE), int(ROI_SIZE)), dtype=np.float32))
            continue

        coeff_mag = coeff_mag_all[chosen_ids]
        coeff_phase = coeff_phase_all[chosen_ids]
        z_nm = xyz_rec[on_mask, 2]  # z per record
        xy_pix = xyz_rec[on_mask, :2]  # (M,2)

        RHO2 = FX ** 2 + FY ** 2  # reuse grid

        canvas = np.zeros((HR_SIZE, HR_SIZE), dtype=np.float32)

        for i in range(chosen_ids.size):
            P0 = construct_pupil(coeff_mag[i], coeff_phase[i], basis, pupil_mask)
            pupil_defocus = P0 * np.exp(1j * np.pi * wavelength_m * (z_nm[i] * 1e-9) * RHO2)
            psf_i = generate_psf(pupil_defocus)

            cx = int(round(xy_pix[i,0] * UPSCALE))
            cy = int(round(xy_pix[i,1] * UPSCALE))

            r0 = cy - half_psf; r1 = cy + half_psf
            c0 = cx - half_psf; c1 = cx + half_psf
            psf_r0 = psf_c0 = 0
            if r0 < 0:
                psf_r0 = -r0; r0 = 0
            if c0 < 0:
                psf_c0 = -c0; c0 = 0
            r1 = min(r1, HR_SIZE)
            c1 = min(c1, HR_SIZE)
            canvas[r0:r1, c0:c1] += psf_i[psf_r0:psf_r0+(r1-r0), psf_c0:psf_c0+(c1-c0)]

        # Downsample to camera resolution
        low_img = resize(canvas, (int(ROI_SIZE), int(ROI_SIZE)), order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
        # Photon conservation
        total_h = float(canvas.sum()); total_l = float(low_img.sum())
        if total_l != 0:
            low_img *= (total_h/total_l)

        low_imgs.append(low_img)

    stack = np.stack(low_imgs, axis=0)  # (T, 1200,1200)

    low_tiff = os.path.abspath(args.out)  # use provided path
    out_dir = os.path.dirname(low_tiff)
    ome_meta = {
        "Axes": "TYX",
        "PhysicalSizeX": pix_x_nm,
        "PhysicalSizeXUnit": "nm",
        "PhysicalSizeY": pix_y_nm,
        "PhysicalSizeYUnit": "nm",
    }
    tiff.imwrite(low_tiff, stack, photometric="minisblack", metadata=ome_meta)
    print(f"Saved {stack.shape[0]}-frame stack to {low_tiff}")


if __name__ == "__main__":
    main() 