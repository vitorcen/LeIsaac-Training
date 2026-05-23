# LeIsaac Training Perf Toolkit

通用训练性能优化工具集 — 适用于任何 HF Trainer + LeRobot v2.x 数据集训练任务
（GR00T-N1.7 / DreamZero / π0.5 / X-VLA / SmolVLA / ACT / DP / OpenVLA / OpenVLA-OFT）。

> 判断锚：**GPU util mid-window %** 不是 step/s，不是 loss。
> 详细心智模型见 `feedback-gpu-util-as-efficiency-anchor.md` 记忆。

## 工具一览

| 工具 | 作用 | 通用性 |
|---|---|---|
| `sample_gpu.sh` | 1Hz GPU+CPU util 采样 → CSV | 完全通用 |
| `analyze_gpu_csv.py` | 解析 CSV → mid-window mean / p50 / p90 / max + CPU | 完全通用 |
| `precache_videos.py` | mp4 → npy memmap，按 LeRobot v2.x 标准布局 | LeRobot v2.x 数据集 |
| `bench_dataloader.py` | dataloader 路径单点 throughput (5 mode) | LeRobot v2.x 数据集 |
| `pipeline_patches.py` | HF Trainer + DataLoader 通用 monkey-patch | HF Trainer 项目 |

## 推荐工作流

### 1. 建立 baseline

```bash
# Terminal A: 启动 sampler（与训练同时跑）
LeIsaac/scripts/training/perf/sample_gpu.sh /tmp/gpu_baseline.csv 1500 &

# Terminal B: 启动正常训练（不打 patch）
bash LeIsaac/scripts/finetune/gr00t/train_n17.sh   # 或你的训练 launcher

# 等若干 step 后 Ctrl-C 训练 + kill sampler
LeIsaac/scripts/training/perf/analyze_gpu_csv.py /tmp/gpu_baseline.csv
```

输出例：
```
   name |     n | gpu_mid% | gpu_p50 | gpu_p90 | gpu_max | mem_MB | cpu_p50 | cpu_max
   gpu_baseline | 114 |  50.0   |   9     |   85    |   98    | 15011  | 100.0   | 100.0
```

按 GPU util mid 判断瓶颈层（见 memory `feedback-gpu-util-as-efficiency-anchor`）：

| gpu_mid% | CPU | 攻击方向 |
|---|---|---|
| < 30% | 100% | 先 precache + non_blocking |
| 30-50% | 100% | non_blocking + prefetch_factor |
| 50-70% | 100% | 主线程 collator 瓶颈，移 image_proc → worker |
| 70-90% | < 80% | bf16 / mixed / batch 调 |
| > 90% | * | GPU 已饱和 |

### 2. 预解码视频缓存

```bash
python LeIsaac/scripts/training/perf/precache_videos.py \
    --dataset_dir /home/david/work/.../leisaac-pick-orange \
    --cache_dir /home/david/cache/leisaac_pick_orange_frames \
    --workers 8
```

- 60 ep × 2 cam × 480×640 H264 → 62 GB uint8 cache，~3 min（NVMe SSD）
- 默认幂等：已 cache 的 file skip

### 3. 在 launcher 里启用 patch

在你的训练 launcher（如 `launch_finetune_ckpt_n17.py`）头部加：

```python
# Inside the launcher, after gr00t imports
import sys
sys.path.insert(0, "/home/david/work/isaaclab-experience/LeIsaac/scripts/training")
from perf.pipeline_patches import apply_all
apply_all()
```

然后用 env 变量控制：

```bash
# 最小推荐配置
LEISAAC_FRAME_CACHE_DIR=/home/david/cache/<task>_frames \
DATALOADER_NUM_WORKERS=4 \
DATALOADER_PREFETCH_FACTOR=4 \
bash <your_train_launcher.sh>
```

### 4. Profile per-phase 找瓶颈

```bash
PROFILE_PHASES=1 \
PROFILE_TARGETS="gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.Gr00tN1d7DataCollator.__call__" \
bash <your_train_launcher.sh> 2>&1 | grep perf-profile
```

输出例：
```
[perf-profile] Gr00tN1d7DataCollator.__call__: n=20 mean50=27.4ms p90=34.3ms
[perf-profile] Gr00tN1d7DataCollator.__call__: n=40 mean50=24.9ms p90=35.3ms
```

mean50 是最近 50 次调用平均。乘以 grad_accum 步数 = 每 step 该 phase 主线程时间。

### 5. 微基准（不跑模型）

```bash
# 对比 5 mode
for m in baseline memmap memmap-cached workers-decode workers-memmap; do
    python LeIsaac/scripts/training/perf/bench_dataloader.py \
        --dataset_dir /home/david/work/.../leisaac-pick-orange \
        --cache_dir /home/david/cache/leisaac_pick_orange_frames \
        --mode $m --num_samples 400 --frames_per_call 16
done
```

## 已验证的优化 ROI（GR00T-N1.7 PickOrange 实测，2026-05-23）

| 配置 | runtime (80 step) | step/s | GPU mid | 备注 |
|---|---|---|---|---|
| baseline (4w decode) | 92.6s | 0.86 | 50.0% | 起点 |
| memmap (4w) | 78.0s | 1.03 | 50.9% | -16%，全在 warmup |
| memmap + 8w | 82.5s | 0.97 | 56.3% | i9-13900KF 8 P-core，超 reverse |
| memmap + 4w + non_blocking | **76.5s** | **1.05** | **61.3%** | 最佳，GPU util 真涨 |
| memmap + 4w + pipe + pf8 | 78.7s | 1.02 | 55.2% | pf=8 反 -3% |

**结论**：减少 startup + H2D 重叠 ≈ 17% wall-clock + 11pp GPU util。剩余 30% 卡在主线程 collator（VLM tokenize + image_processor），需移到 worker 或预处理缓存。

## 跨项目复用清单

| 项目 | 适用 patch | 备注 |
|---|---|---|
| **GR00T-N1.7** | 全部 | 主战场，已验证 |
| **GR00T-N1.5/1.6** | 同上，PROFILE_TARGETS 改 N1.5/1.6 collator 类名 | 同 LeRobot v2.x |
| **DreamZero** | precache + non_blocking + prefetch；profile target 改 DreamZero 自己的 collator | 14B Wan video diffusion，CPU 端类似 |
| **π0.5 PyTorch** | precache + non_blocking | 用 openpi 训练框架，prepare_input 改写位置可能不同 |
| **X-VLA** | precache + non_blocking + prefetch | lerobot 框架，apply_all() 直接生效 |
| **SmolVLA** | 同 X-VLA | lerobot 框架 |
| **ACT / DP** | precache + non_blocking + prefetch | lerobot 框架 |
| **OpenVLA / OpenVLA-OFT** | precache + non_blocking；profile target 改 collator | HF 直接 |

## 关联

- 设计文档：`LeIsaac/docs/training/gpu_dataloader_zero_copy.html`（含 SVG 流程图 + 三模型 brainstorm）
- 心智锚 memory：`feedback-gpu-util-as-efficiency-anchor`
- 数据集格式参考：LeRobot v2.x `meta/info.json` + `videos/chunk-{NNN}/observation.images.{cam}/`
