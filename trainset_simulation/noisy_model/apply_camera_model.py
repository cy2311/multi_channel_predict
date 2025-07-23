import os
import json
import argparse
import numpy as np
import tifffile as tiff


def parse_args():
    parser = argparse.ArgumentParser(description="Apply camera QE + EM gain to ideal photon TIFF stack")
    parser.add_argument('--input', type=str,
                        default=os.path.join('simulated_data_multi_frames', 'frames_200f_1200px_origional_photon.ome.tiff'),
                        help='Input OME-TIFF containing photons/frame')
    parser.add_argument('--output', type=str,
                        default=os.path.join('simulated_data_multi_frames', 'frames_200f_1200px_camera_photon.ome.tiff'),
                        help='Output OME-TIFF after camera model')
    parser.add_argument('--cam-json', type=str,
                        default=os.path.join('beads', 'spool_100mW_30ms_3D_1_2', 'camera_parameters.json'),
                        help='JSON file with camera parameters (QE, EMGain)')
    return parser.parse_args()


def main():
    args = parse_args()
    input_tiff = args.input
    output_tiff = args.output
    cam_json = args.cam_json

    # -------------------------------------------------------------------------
    print(f'Loading photon stack: {input_tiff}')
    photon_stack = tiff.imread(input_tiff)
    print('  shape:', photon_stack.shape, 'dtype:', photon_stack.dtype)

    # Camera parameters
    with open(cam_json, 'r') as f:
        cam_cfg = json.load(f)

    qe = cam_cfg.get('QE', cam_cfg.get('qe', 0.9))
    emgain = cam_cfg.get('EMGain', cam_cfg.get('em_gain', 30.0))
    print(f'Camera QE={qe}, EMGain={emgain}')

    rng = np.random.default_rng(seed=42)
    frames_out: list[np.ndarray] = []

    for idx in range(photon_stack.shape[0]):
        photons = photon_stack[int(idx)].astype(np.float64)  # type: ignore[index]
        electrons = rng.poisson(photons * qe)
        if emgain > 1.0:
            flat = electrons.ravel()
            gained = rng.gamma(shape=flat, scale=emgain)
            em_electrons = gained.reshape(electrons.shape)
        else:
            em_electrons = electrons.astype(np.float64)

        # 3) Read noise (Gaussian) & baseline
        read_sigma = cam_cfg.get('read_noise_e', cam_cfg.get('read_noise', 1.0))
        baseline   = cam_cfg.get('offset', 0.0)
        e_with_noise = em_electrons + rng.normal(0.0, read_sigma, em_electrons.shape)

        # 4) A/D conversion
        e_adu = cam_cfg.get('A2D', 1.0)  # electrons per ADU
        adu = np.round(e_with_noise / e_adu + baseline)

        # 5) Clip to 16-bit max (if provided)
        max_adu = cam_cfg.get('max_adu', 65535)
        adu = np.clip(adu, 0, max_adu).astype(np.uint16)

        frames_out.append(adu)

    stack_out = np.stack(frames_out, axis=0)
    print('Output dtype', stack_out.dtype)

    out_dir = os.path.dirname(output_tiff)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    meta = {'Axes': 'TYX', 'CameraModel': 'Simulated_EMCCD', 'QE': qe, 'EMGain': emgain, 'Baseline': baseline, 'A2D': e_adu, 'ReadNoise_e': read_sigma}
    tiff.imwrite(output_tiff, stack_out, photometric='minisblack', metadata=meta)
    print(f'Saved camera-processed stack to {output_tiff}')


if __name__ == '__main__':
    main() 