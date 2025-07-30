import argparse
import random
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch


def overlap(t0: float, te: float, k: int) -> float:
    """Return overlap length between [t0, te) and integer frame [k, k+1)."""
    return max(0.0, min(te, k + 1) - max(t0, k))


def sample_emitters(num_emitters: int, frame_range: tuple, area_px: float, intensity_mu: float,
                    intensity_sigma: float, lifetime_avg: float, z_range_um: float = 1.0, seed: int = 42,
                    enable_dual_channel: bool = False, channel_1_wavelength: float = 561.0, 
                    channel_2_wavelength: float = 680.0):
    """Sample basic emitter attributes.

    Parameters:
        enable_dual_channel: bool, whether to generate dual-channel data
        channel_1_wavelength: float, wavelength for channel 1 in nm (default 561nm)
        channel_2_wavelength: float, wavelength for channel 2 in nm (default 680nm)

    Returns:
        dict with keys xyz, intensity, t0, on_time, id (all torch tensors)
        If dual_channel enabled, also includes channel_ratio, channel_1_wavelength, channel_2_wavelength
    """
    rng = np.random.default_rng(seed)

    # Spatial positions: x,y in pixel units within FOV; z in micrometres within ±z_range_um
    xy = rng.uniform(0, area_px, size=(num_emitters, 2))
    z = rng.uniform(-z_range_um, z_range_um, size=(num_emitters, 1))  # µm
    xyz = torch.tensor(np.concatenate([xy, z], axis=1), dtype=torch.float32)

    # Intensity (photons / frame) from Gaussian, clamp >=1e-8
    intensity = torch.tensor(rng.normal(intensity_mu, intensity_sigma, size=num_emitters),
                             dtype=torch.float32).clamp_min(1e-8)

    # First on-time t0 sampled with buffer of 3×lifetime on both sides
    t0 = torch.tensor(rng.uniform(frame_range[0] - 3 * lifetime_avg,
                                  frame_range[1] + 3 * lifetime_avg, size=num_emitters), dtype=torch.float32)

    # On-time duration from exponential distribution
    on_time = torch.distributions.Exponential(1 / lifetime_avg).sample((num_emitters,))

    ids = torch.arange(num_emitters, dtype=torch.long)

    result = dict(xyz=xyz, intensity=intensity, t0=t0, on_time=on_time, id=ids)
    
    # 添加双通道支持
    if enable_dual_channel:
        # 为每个发射器生成通道比例（0-1之间的均匀分布）
        channel_ratio = torch.tensor(rng.uniform(0, 1, size=num_emitters), dtype=torch.float32)
        result['channel_ratio'] = channel_ratio
        result['channel_1_wavelength'] = torch.full((num_emitters,), channel_1_wavelength, dtype=torch.float32)
        result['channel_2_wavelength'] = torch.full((num_emitters,), channel_2_wavelength, dtype=torch.float32)
    
    return result


def bin_emitters_to_frames(em_attrs: dict, frame_range: tuple):
    """Convert continuous blinking intervals into per-frame records.

    Args:
        em_attrs: dict returned by sample_emitters
        frame_range: (start, end) inclusive frame indices

    Returns:
        records dict with xyz, phot, frame_ix, id tensors
        If dual_channel enabled, also includes channel_ratio
    """
    k_start, k_end = frame_range

    xyz_all = []
    phot_all = []
    frame_all = []
    id_all = []
    channel_ratio_all = []

    xyz = em_attrs['xyz']
    intensity = em_attrs['intensity']
    t0 = em_attrs['t0']
    on_time = em_attrs['on_time']
    ids = em_attrs['id']
    
    # 检查是否有双通道数据
    has_dual_channel = 'channel_ratio' in em_attrs
    if has_dual_channel:
        channel_ratio = em_attrs['channel_ratio']

    for i in range(len(ids)):
        te = float(t0[i] + on_time[i])
        for k in range(k_start, k_end + 1):
            dt = overlap(float(t0[i]), te, k)
            if dt > 0.0:
                xyz_all.append(xyz[i])
                phot_all.append(intensity[i] * dt)
                frame_all.append(k)
                id_all.append(ids[i])
                if has_dual_channel:
                    channel_ratio_all.append(channel_ratio[i])

    records = dict(
        xyz=torch.stack(xyz_all) if xyz_all else torch.zeros((0, 3), dtype=torch.float32),
        phot=torch.tensor(phot_all, dtype=torch.float32),
        frame_ix=torch.tensor(frame_all, dtype=torch.long),
        id=torch.tensor(id_all, dtype=torch.long)
    )
    
    # 添加双通道数据
    if has_dual_channel:
        records['channel_ratio'] = torch.tensor(channel_ratio_all, dtype=torch.float32)
    
    return records


def save_to_h5(out_path: Path, em_attrs: dict, records: dict):
    """Save emitter attributes and per-frame records into an HDF5 file."""
    with h5py.File(out_path, 'w') as f:
        grp_em = f.create_group('emitters')
        # 保存基本属性
        for key in ('xyz', 'intensity', 't0', 'on_time', 'id'):
            grp_em.create_dataset(key, data=em_attrs[key].cpu().numpy())
        
        # 保存双通道相关属性（如果存在）
        dual_channel_keys = ('channel_ratio', 'channel_1_wavelength', 'channel_2_wavelength')
        for key in dual_channel_keys:
            if key in em_attrs:
                grp_em.create_dataset(key, data=em_attrs[key].cpu().numpy())

        grp_rec = f.create_group('records')
        for key in ('xyz', 'phot', 'frame_ix', 'id'):
            grp_rec.create_dataset(key, data=records[key].cpu().numpy())
        
        # 保存双通道records数据（如果存在）
        if 'channel_ratio' in records:
            grp_rec.create_dataset('channel_ratio', data=records['channel_ratio'].cpu().numpy())


 
def visualise(em_attrs: dict, records: dict, num_plot: int = 20, set_idx: int = None, out_dir: Path = None):
    """Plot spatial distribution and t0/te lines for random emitters."""
    import matplotlib.pyplot as plt

    # 根据set_idx生成文件名后缀
    suffix = f"_set{set_idx}" if set_idx is not None else ""
    
    # 确保输出目录存在
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成输出文件路径
    spatial_path = out_dir / f'emitters_spatial{suffix}.png' if out_dir else Path(f'emitters_spatial{suffix}.png')
    timeline_path = out_dir / f'emitters_timeline{suffix}.png' if out_dir else Path(f'emitters_timeline{suffix}.png')
    
    # Spatial scatter (x vs y)
    plt.figure(figsize=(6, 6))
    plt.scatter(em_attrs['xyz'][:, 0].numpy(), em_attrs['xyz'][:, 1].numpy(), s=5, alpha=0.6)
    plt.title('Emitter spatial distribution')
    plt.xlabel('x (px)')
    plt.ylabel('y (px)')
    plt.gca().set_aspect('equal')
    plt.savefig(spatial_path)
    plt.close()

    # Timeline plot for random emitters
    total_emitters = len(em_attrs['id'])
    chosen = random.sample(range(total_emitters), k=min(num_plot, total_emitters))

    plt.figure(figsize=(8, 3))                       # ⬅ 更瘦长一点
    for idx, eid in enumerate(chosen):
        t0 = float(em_attrs['t0'][eid])
        te = float(em_attrs['t0'][eid] + em_attrs['on_time'][eid])
        color = plt.cm.tab20(idx % 20)               # ⬅ 从 colormap 挑颜色 #type: ignore
        plt.hlines(y=idx, xmin=t0, xmax=te,
                   color=color, linewidth=2)
        plt.scatter([t0, te], [idx, idx],
                    color=color, s=12)

    plt.yticks([])                                   # ⬅ 隐藏 y 轴刻度 / 标签
    plt.xlabel('Frame')
    plt.title(f'Blinking intervals of {len(chosen)} random emitters')
    plt.tight_layout()
    plt.savefig(timeline_path)
    plt.close()



def main():
    parser = argparse.ArgumentParser(description='Generate synthetic emitter records and save to HDF5.')
    parser.add_argument('--num_emitters', type=int, default=1000, help='Number of emitters to sample')
    parser.add_argument('--frames', type=int, default=10, help='Total number of frames (starts at 0)')
    parser.add_argument('--area_px', type=float, default=1200.0, help='FOV side length in pixel units')
    parser.add_argument('--intensity_mu', type=float, default=2000.0, help='Mean photon flux per frame')
    parser.add_argument('--intensity_sigma', type=float, default=400.0, help='Std of photon flux')
    parser.add_argument('--lifetime_avg', type=float, default=2.5, help='Average on-time (frames)')
    parser.add_argument('--z_range_um', type=float, default=1.0, help='Half-range of z positions (±value, in µm)')
    parser.add_argument('--out', type=str, default='emitters.h5', help='Output HDF5 filename')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory for all files (default: current directory)')
    parser.add_argument('--filename_pattern', type=str, default='{stem}_set{idx}{ext}', 
                        help='Filename pattern for multiple sets. Available variables: {stem}, {idx}, {ext}')
    parser.add_argument('--no_plot', action='store_true', help='Disable plots')
    parser.add_argument('--plot_first_only', action='store_true', help='Only generate plots for the first set')
    parser.add_argument('--num_sets', type=int, default=1, help='Number of emitter sets to generate')
    parser.add_argument('--seed_start', type=int, default=42, help='Starting seed for random number generation')
    
    # 参数变化范围
    parser.add_argument('--vary_intensity', action='store_true', help='Vary intensity across sets')
    parser.add_argument('--intensity_range', type=str, default='1500,2500', 
                        help='Range of intensity_mu values (min,max) when vary_intensity is enabled')
    parser.add_argument('--vary_lifetime', action='store_true', help='Vary lifetime across sets')
    parser.add_argument('--lifetime_range', type=str, default='1.5,3.5', 
                        help='Range of lifetime_avg values (min,max) when vary_lifetime is enabled')
    parser.add_argument('--vary_z_range', action='store_true', help='Vary z_range across sets')
    parser.add_argument('--z_range_values', type=str, default='0.5,1.0,1.5', 
                        help='Comma-separated z_range values to use when vary_z_range is enabled')
    parser.add_argument('--vary_emitter_count', action='store_true', help='Vary number of emitters across sets')
    parser.add_argument('--emitter_count_range', type=str, default='500,2000', 
                        help='Range of emitter counts (min,max) when vary_emitter_count is enabled')
    parser.add_argument('--vary_frames', action='store_true', help='Vary number of frames across sets')
    parser.add_argument('--frames_range', type=str, default='5,20', 
                        help='Range of frame counts (min,max) when vary_frames is enabled')
    
    # 双通道相关参数
    parser.add_argument('--enable_dual_channel', action='store_true', help='Enable dual-channel data generation')
    parser.add_argument('--channel_1_wavelength', type=float, default=561.0, help='Wavelength for channel 1 in nm')
    parser.add_argument('--channel_2_wavelength', type=float, default=680.0, help='Wavelength for channel 2 in nm')
    
    args = parser.parse_args()

    # 处理帧数变化范围
    frame_counts = [args.frames] * args.num_sets
    if args.vary_frames and args.num_sets > 1:
        frames_min, frames_max = map(int, args.frames_range.split(','))
        frame_counts = [int(x) for x in np.linspace(frames_min, frames_max, args.num_sets)]
    
    # 处理输出目录
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理参数变化范围
    intensity_values = [args.intensity_mu] * args.num_sets
    if args.vary_intensity and args.num_sets > 1:
        intensity_min, intensity_max = map(float, args.intensity_range.split(','))
        intensity_values = np.linspace(intensity_min, intensity_max, args.num_sets)
    
    lifetime_values = [args.lifetime_avg] * args.num_sets
    if args.vary_lifetime and args.num_sets > 1:
        lifetime_min, lifetime_max = map(float, args.lifetime_range.split(','))
        lifetime_values = np.linspace(lifetime_min, lifetime_max, args.num_sets)
    
    z_range_values = [args.z_range_um] * args.num_sets
    if args.vary_z_range and args.num_sets > 1:
        z_values = list(map(float, args.z_range_values.split(',')))
        # 如果提供的值少于num_sets，则循环使用
        z_range_values = [z_values[i % len(z_values)] for i in range(args.num_sets)]
    
    emitter_counts = [args.num_emitters] * args.num_sets
    if args.vary_emitter_count and args.num_sets > 1:
        count_min, count_max = map(int, args.emitter_count_range.split(','))
        emitter_counts = [int(x) for x in np.linspace(count_min, count_max, args.num_sets)]
    
    for set_idx in range(args.num_sets):
        # 使用不同的种子生成每组数据
        current_seed = args.seed_start + set_idx
        
        # 获取当前组的参数值
        current_intensity = intensity_values[set_idx]
        current_lifetime = lifetime_values[set_idx]
        current_z_range = z_range_values[set_idx]
        current_emitter_count = emitter_counts[set_idx]
        current_frames = frame_counts[set_idx]
        
        # 设置当前帧范围
        frame_range = (0, current_frames - 1)
        
        # 为每组数据创建唯一的输出文件名
        if args.num_sets > 1:
            # 使用用户指定的文件名模式
            stem = Path(args.out).stem
            ext = Path(args.out).suffix
            out_filename = args.filename_pattern.format(stem=stem, idx=set_idx, ext=ext)
        else:
            out_filename = args.out
        
        # 设置完整的输出路径
        out_path = out_dir / out_filename if out_dir else Path(out_filename)
        
        # 记录参数变化
        param_changes = []
        if args.vary_intensity:
            param_changes.append(f"intensity={current_intensity:.1f}")
        if args.vary_lifetime:
            param_changes.append(f"lifetime={current_lifetime:.1f}")
        if args.vary_z_range:
            param_changes.append(f"z_range={current_z_range:.1f}")
        if args.vary_emitter_count:
            param_changes.append(f"emitters={current_emitter_count}")
        if args.vary_frames:
            param_changes.append(f"frames={current_frames}")
        
        param_str = ", ".join(param_changes)
        if param_str:
            print(f"Set {set_idx+1}/{args.num_sets} parameters: {param_str}")
        
        # 1) sample emitters
        em_attrs = sample_emitters(current_emitter_count, frame_range, args.area_px,
                                 current_intensity, args.intensity_sigma,
                                 current_lifetime, z_range_um=current_z_range, seed=current_seed,
                                 enable_dual_channel=args.enable_dual_channel,
                                 channel_1_wavelength=args.channel_1_wavelength,
                                 channel_2_wavelength=args.channel_2_wavelength)

        # 2) bin to frames
        records = bin_emitters_to_frames(em_attrs, frame_range)

        # 3) save
        save_to_h5(out_path, em_attrs, records)
        print(f'Saved dataset {set_idx+1}/{args.num_sets} to {out_path.resolve()}')

        # 4) visualise (根据参数决定是否为每组数据生成可视化)
        if not args.no_plot and (set_idx == 0 or not args.plot_first_only):
            visualise(em_attrs, records, set_idx=set_idx, out_dir=out_dir)


if __name__ == '__main__':
    main()