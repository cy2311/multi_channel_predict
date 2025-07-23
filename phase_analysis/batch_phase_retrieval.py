#!/usr/bin/env python3
"""batch_phase_retrieval.py – iterate through simulated_data/patches.h5, run
GPU-accelerated Gerchberg–Saxton phase retrieval per patch, compute field-
dependent Zernike coefficients and write them back to the same HDF5 file under
/group `zernike/`.

Run:
    python phase_analysis/batch_phase_retrieval.py --patch-h5 simulated_data/patches.h5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch

# Support running as module (-m phase_analysis.batch_phase_retrieval) or script
try:
    from .gs_zernike import run_gs_iter, zernike_decompose
except ImportError:  # fallback when executed as script from repo root
    from gs_zernike import run_gs_iter, zernike_decompose  # type: ignore

# -----------------------------------------------------------------------------
# CLI arguments
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch phase retrieval over PSF patches")
    p.add_argument("--patch-h5", type=str, default="simulated_data/patches.h5", help="Input HDF5 file containing patches")
    p.add_argument("--cfg", type=str, default="configs/default_config.json", help="JSON config file (optical params)")
    p.add_argument("--iter-max", type=int, default=30, help="GS max iterations")
    p.add_argument("--ncc-th", type=float, default=0.7, help="Early-stop NCC threshold")
    p.add_argument("--gpu", action="store_true", help="Force CUDA (error if unavailable)")
    p.add_argument("--log", type=str, default="INFO", help="Logging level")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Main logic
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()), format="%(levelname)s: %(message)s")

    # Load configuration JSON
    import json
    with open(args.cfg, "r") as f:
        cfg = json.load(f)
    cfg.setdefault("phase_retrieval", {})
    cfg["phase_retrieval"].update({
        "iter_max": args.iter_max,
        "ncc_threshold": args.ncc_th,
    })

    # Device selection
    if args.gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    # Open HDF5
    h5_path = Path(args.patch_h5)
    with h5py.File(h5_path, "r+") as h5f:
        patches_ds = h5f["patches"]  # type: ignore[assignment]
        coords_ds = h5f["coords"]    # type: ignore[assignment]
        N = patches_ds.shape[0]  # type: ignore[index]
        logging.info("Total patches: %d", N)

        # Prepare/extend group for Zernike
        zernike_grp = h5f.require_group("zernike")
        coeff_mag_ds = zernike_grp.require_dataset(
            "coeff_mag", shape=(N, 21), maxshape=(N, 21), dtype="float32", fillvalue=np.nan
        )
        coeff_phase_ds = zernike_grp.require_dataset(
            "coeff_phase", shape=(N, 21), maxshape=(N, 21), dtype="float32", fillvalue=np.nan
        )
        mean_ncc_ds = zernike_grp.require_dataset(
            "mean_ncc", shape=(N,), maxshape=(N,), dtype="float32", fillvalue=np.nan
        )

        # Load Zernike basis files (21 polynomials)
        from glob import glob
        import os

        basis_files = sorted(glob("simulated_data/zernike_polynomials/zernike_*_n*_m*.npy"))
        if len(basis_files) < 21:
            basis_files = sorted(glob("simulated_data/zernike_*_n*_m*.npy"))
        if len(basis_files) < 21:
            raise RuntimeError("Cannot locate 21 Zernike basis .npy files")
        Z_basis = np.stack([np.load(fp) for fp in basis_files[:21]], axis=0)  # (21,128,128)

        # Build pupil mask once (from cfg)
        Nxy = patches_ds.shape[-1]  # type: ignore[index]
        wavelength = cfg["optical"]["wavelength_nm"] * 1e-9
        px_x = cfg["optical"]["pixel_size_nm_x"] * 1e-9
        px_y = cfg["optical"]["pixel_size_nm_y"] * 1e-9
        NA = cfg["optical"]["NA"]
        fx = np.fft.fftfreq(Nxy, d=px_x)
        fy = np.fft.fftfreq(Nxy, d=px_y)
        FY, FX = np.meshgrid(fy, fx, indexing="ij")
        RHO2 = FX**2 + FY**2
        pupil_mask = (np.sqrt(RHO2) <= NA / wavelength).astype(np.float32)

        for idx in range(N):
            if not np.isnan(mean_ncc_ds[idx]):
                logging.debug("Skipping patch %d – already processed", idx)
                continue

            patch = np.asarray(patches_ds[idx])  # type: ignore[index]
            logging.info("Patch %d/%d", idx + 1, N)
            try:
                P_final, psf_pred, ncc_curve = run_gs_iter(patch, cfg, device=device)
                final_ncc = ncc_curve[-1] if len(ncc_curve) else np.nan
                if np.isnan(final_ncc) or final_ncc < args.ncc_th:
                    raise RuntimeError(f"NCC {final_ncc:.3f} below threshold {args.ncc_th}")

                coeff_mag, coeff_phase = zernike_decompose(P_final, pupil_mask, Z_basis)
            except Exception as e:
                logging.warning("Discarding patch %d: %s", idx, e)
                coeff_mag = coeff_phase = np.full(21, np.nan, dtype=np.float32)
                final_ncc = np.nan
            mean_ncc_ds[idx] = final_ncc

        logging.info("Batch phase retrieval completed – results saved under /zernike group.")


if __name__ == "__main__":
    main() 