A.「工程化清单」——告诉你在代码、数据结构、流程上究竟要做哪些事、各步骤输入输出是什么；
B.「参考代码」——一套可直接 copy-paste 运行的最小示例（仅依赖 numpy、torch），把清单逐条落地。读完你就能把自己的闪烁模型/PSF/背景/噪声替换进去，或者快速映射到 DECODE 的现成类。
A. 工程化清单
阶段 0 准备
0-1 确定整体模拟窗口：frame_range = (k_start, k_end)
0-2 决定每帧的曝光时间单位（本文默认 1）
阶段 1 生成（frame-binning）
1-1 采样分子总数 N（Poisson 或固定）
1-2 为每个 emitter 生成基本属性
 • xyz : (x,y,z) 坐标
 • intensity : 光子流量 I (photons / frame)
 • t0 : 首次点亮时刻（float，单位＝帧）
 • on_time : 连续发光时长（float，指数分布 etc.）
 • id : 0 … N-1
1-3 将连续闪烁区间 [t0, te) 切块到整数帧
 for k in frame_range
  Δt = max(0, min(te, k+1) − max(t0, k))
  if Δt > 0: 记录一条
   xyz = xyz (不变)
   phot = intensity × Δt
   frame_ix = k
   id = id
1-4 把所有记录拼成四个 tensor
 xyz (M × 3)
 phot (M)
 frame_ix (M, int)
 id (M, int) # 同一 id 可出现多行
（可选）用这些 tensor 构造 DECODE 的 EmitterSet(...)，否则自定义 dict 亦可。
阶段 2 渲染
2-1 PSF.forward(xyz, phot, frame_ix)
 • 先按 frame_ix 分组
 • 对每帧把所有 emitters 的光斑叠加 → raw photon frame
2-2 Background.forward(frames) → frames += B 或返回 (frames+bg, bg)
2-3 Camera/Noise.forward(frames)
 • Poisson(λ = QE·frame + spur)
 • Gamma(EM-gain) 可选
 • Gaussian(read_noise)
 • 量化、baseline …  → camera ADU
结果
 模拟输出：camera frames (以及 bg, raw photon frames — 视需要保留)

渲染阶段的处理顺序可以概括为「一次生成所有理想帧 → 对整块张量依次套用背景、相机噪声模型」，而不是逐像素或逐帧手动循环。具体流程（对应 Simulation.forward 的实现）：
PSF  frames = psf.forward(...)
  • 得到尺寸为 (N_frames, H, W) 的理想光子数张量。
Background  frames, bg = background.forward(frames)
  • 例如加常数、加空间分布噪声等。
  • 仍然保持同样尺寸的张量。
Camera / Noise  frames = camera.forward(frames)
Photon2Camera 中依次执行：
  a) Poisson：模拟光子到电子 (shot-noise)
  b) Gamma：EM-gain（若 em_gain ≠ None）
  c) Gaussian：读出噪声
  d) 量化、baseline、clamp → 得到最终 ADU
这些操作都以向量化方式一次性作用在整个 3-D 张量 (N,H,W) 上；所以「对所有 frames 统一做 Poisson→Gamma→Gaussian」的理解是正确的。