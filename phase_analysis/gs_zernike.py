"""gs_zernike.py – GPU-accelerated Gerchberg–Saxton phase retrieval and
Zernike decomposition utilities.

This module extracts core logic from the original *phase_retrieval_gs copy.py*
script and wraps it into reusable functions so that multiple PSF patches can be
processed programmatically (e.g. in a batch job).

Key public APIs
---------------
run_gs_iter(meas_patch: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    Perform GS iterations on a single PSF stack and return (pupil, psf_stack,
    ncc_per_iter).

zernike_decompose(P: np.ndarray, pupil_mask: np.ndarray, basis: np.ndarray)
    -> tuple[np.ndarray, np.ndarray]
    Return magnitude & phase Zernike coefficients (21-length each).
"""

from __future__ import annotations

import math
import logging
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F

# -----------------------------
# Helpers for FFT shifts (torch)
# -----------------------------

def _fftshift(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(x, dim=(-2, -1))


def _ifftshift(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifftshift(x, dim=(-2, -1))


# -----------------------------
# Normalised cross correlation  
# -----------------------------

def ncc_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return scalar NCC computed on GPU (a,b: 2-D tensors)."""
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.norm(a) * torch.norm(b) + 1e-12
    return torch.dot(a, b) / denom


# -----------------------------
# GS iteration core            
# -----------------------------

def run_gs_iter(
    meas_patch: np.ndarray,
    cfg: dict,
    *,
    iter_max: int | None = None,
    ncc_threshold: float | None = None,
    device: str | torch.device | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gerchberg-Saxton iterations on a single upsampled patch.

    Parameters
    ----------
    meas_patch : np.ndarray
        Array of shape (Z, H, W) representing the upsampled patch (normalised
        intensity). It must match the slice selection parameters stored in the
        preprocessing stage (PATCH_Z = 201).
    cfg : dict
        Parsed JSON configuration (default_config.json) containing optical &
        camera parameters.
    iter_max : Optional[int]
        Max iterations (defaults to cfg["phase_retrieval"]["iter_max"] or 100).
    ncc_threshold : Optional[float]
        Early stop threshold on mean NCC (default 0.7).
    device : Optional[str | torch.device]
        PyTorch device, defaults to "cuda" if available.

    Returns
    -------
    P_final : np.ndarray (H,W) complex64
    psf_pred : np.ndarray (Z,H,W) float32 – final predicted PSF stack
    ncc_hist : np.ndarray (n_iter,) float32 – mean NCC per iteration
    """

    # -------------------------
    # Basic shapes & device
    # -------------------------
    Z, H, W = meas_patch.shape
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    logger = logging.getLogger(__name__)
    logger.debug("Running GS on device %s", device)

    # Iteration parameters
    iter_max = iter_max or cfg.get("phase_retrieval", {}).get("iter_max", 100)
    ncc_threshold = ncc_threshold or cfg.get("phase_retrieval", {}).get(
        "ncc_threshold", 0.7
    )

    # Slice selection params (must match preprocessing)
    center_slice = cfg["phase_retrieval"].get("center_slice", 80)
    step = cfg["phase_retrieval"].get("step", 2)
    n_each_side = cfg["phase_retrieval"].get("n_each_side", 20)

    # Select 41 slices around centre (assumes consistent with preprocessing)
    indices = center_slice + np.arange(-n_each_side, n_each_side + 1) * step
    meas_psfs_np = meas_patch[indices]
    meas_psfs_t = torch.from_numpy(meas_psfs_np).to(device)
    meas_ampls_np = np.sqrt(np.clip(meas_psfs_np, 0, None))

    # -------------------------
    # Prepare constants (torch)
    # -------------------------
    wavelength_nm = cfg["optical"]["wavelength_nm"]
    pixel_size_nm_x = cfg["optical"]["pixel_size_nm_x"]
    pixel_size_nm_y = cfg["optical"]["pixel_size_nm_y"]
    NA = cfg["optical"]["NA"]

    wavelength = wavelength_nm * 1e-9  # m
    pixel_size_x = pixel_size_nm_x * 1e-9
    pixel_size_y = pixel_size_nm_y * 1e-9

    # Frequency grid
    fx = np.fft.fftfreq(W, d=pixel_size_x)
    fy = np.fft.fftfreq(H, d=pixel_size_y)
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    RHO2_np = FX**2 + FY**2
    f_max = NA / wavelength
    pupil_mask_np = (np.sqrt(RHO2_np) <= f_max).astype(np.float32)

    # Convert to torch tensors
    pupil_mask = torch.from_numpy(pupil_mask_np).to(device)
    RHO2 = torch.from_numpy(RHO2_np).to(device)
    meas_ampls = torch.from_numpy(meas_ampls_np).to(device)

    # Defocus phase factors (list of Z, each (H,W))
    z_spacing_nm = cfg["phase_retrieval"].get("slice_spacing_nm", 30)
    z_list_m = (
        np.arange(-n_each_side, n_each_side + 1) * z_spacing_nm * 1e-9
    )  # length 41
    defocus_phases: List[torch.Tensor] = []
    for z in z_list_m:
        phase = torch.exp(1j * math.pi * wavelength * z * RHO2)
        defocus_phases.append(phase)

    # Initial pupil (unit amp, zero phase)
    P = pupil_mask * torch.exp(1j * torch.zeros((H, W), device=device))

    ncc_hist: List[float] = []

    # -------------------------
    # GS iterations
    # -------------------------
    for it in range(iter_max):
        pupil_estimates: List[torch.Tensor] = []
        preds_psf_it: List[torch.Tensor] = []

        for z_idx, H_z in enumerate(defocus_phases):
            Pz = P * H_z
            field_img = torch.fft.ifft2(_ifftshift(Pz))
            new_field_img = meas_ampls[z_idx] * torch.exp(1j * torch.angle(field_img))
            Pz_new = _fftshift(torch.fft.fft2(new_field_img))
            pupil_estimates.append(Pz_new / H_z)
            preds_psf_it.append(field_img.abs() ** 2)

        # NCC evaluation (move small tensors to CPU for cheap computation)
        ncc_vals = [
            ncc_torch(preds_psf_it[i], meas_psfs_t[i]).item()
            for i in range(len(preds_psf_it))
        ]
        mean_ncc = float(np.mean(ncc_vals))
        ncc_hist.append(mean_ncc)
        logger.debug("Iter %d: mean NCC=%.4f", it + 1, mean_ncc)

        # Update pupil
        P = torch.mean(torch.stack(pupil_estimates, dim=0), dim=0)
        P = pupil_mask * torch.exp(1j * torch.angle(P))

        if mean_ncc >= ncc_threshold:
            logger.info("Early stop at iter %d (NCC=%.3f)", it + 1, mean_ncc)
            break

    # -------------------------
    # Final predicted PSF stack
    # -------------------------
    psf_pred_list: List[torch.Tensor] = []
    for H_z in defocus_phases:
        field_img = torch.fft.ifft2(_ifftshift(P * H_z))
        psf_pred_list.append((field_img.abs() ** 2).real)
    psf_pred = torch.stack(psf_pred_list, dim=0)

    # Move outputs to CPU numpy
    return (
        P.cpu().numpy().astype(np.complex64),
        psf_pred.cpu().numpy().astype(np.float32),
        np.asarray(ncc_hist, dtype=np.float32),
    )


# -----------------------------
# Zernike decomposition         
# -----------------------------

def zernike_decompose(
    P: np.ndarray,
    pupil_mask: np.ndarray,
    basis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (coeff_mag, coeff_phase), each (21,) float32."""

    inside = pupil_mask.astype(bool)
    basis_vecs = basis[:, inside].T  # (Npix_in, 21)

    mag_target = np.abs(P)[inside]
    coeff_mag, *_ = np.linalg.lstsq(basis_vecs, mag_target, rcond=None)
    if (norm := np.linalg.norm(coeff_mag)) != 0:
        coeff_mag = coeff_mag / norm

    phase_target = np.unwrap(np.unwrap(np.angle(P), axis=0), axis=1)[inside]
    coeff_phase, *_ = np.linalg.lstsq(basis_vecs, phase_target, rcond=None)
    if (norm_p := np.linalg.norm(coeff_phase)) != 0:
        coeff_phase = coeff_phase / norm_p

    return coeff_mag.astype(np.float32), coeff_phase.astype(np.float32) 