# StarVLA — SO-101 PickOrange 训练套件 / training kit

[StarVLA](https://github.com/starVLA/starVLA) (MIT) 在 LeIsaac **SO-101 PickOrange**
上的微调套件。云端跑 AutoDL（RTX 4080 SUPER 32G / kernel 5.4 / CUDA 12.4）。
承接选型 [[vla-pickorange-vision-resolution-selection]]，是 isaaclab-experience 的第二条 VLA 路线。

**一个 kit，多个 VLM×head 变体**：env / data_registry / modality / launcher / smoke / watchdog
**全部共享**；每个变体 = `configs/` 下**一个 yaml**。新增 `Gemma4 / Cosmos / MiniCPM × GR00T / PI_v3`
不复制目录，只加一个 config。
_One kit, many variants. A new VLM×head = ONE config yaml, not a new dir._

## 结构 / Layout

```
starvla/
  configs/
    so101_qwen_gr00t.yaml         # Run-1: Qwen3-VL-4B (冻) + GR00T flow-matching head
    so101_qwen3vl8b_gr00t.yaml    # 变体: 同框架同 head, 换 8B 冻结骨干 (bs=4 / 60k step)
    # so101_gemma4_pi_v3.yaml     # ← 再加变体只加这一个文件（含自己的 framework 块 + base_vlm）
  common/
    modality.json              # SO-101 6-DOF modality（→ 拷进数据集 meta/）
    data_registry/data_config.py   # 自包含 registry（robot_type + mixture so101_pickorange）
  env/
    env_build.sh               # 建 env（py3.10 + torch2.6cu124 + transformers4.57 + flash-attn）
    dl_base.sh                 # 下 VLM backbone（REPO=… 参数化，复用于所有变体）
  smoke/
    smoke_dataloader.py        # dataloader 冒烟（registry + shape + av1 + 分辨率）
    smoke_train.sh             # 3-step forward 冒烟
  run_train.sh                 # 启动器（VLM-agnostic；CONFIG=… 选变体；RESUME=1 续训）
  watchdog.sh                  # 崩溃自动 RESUME=1
```

eval 不在此 —— 在 `LeIsaac/scripts/evaluation/`：`serve_starvla.py`（websocket 适配器，
从 ckpt 的 `config.yaml` 重建框架，VLM-agnostic）+ `starvla_sweep_watcher.sh`（formal
120s/180s 口径）+ `smoke_starvla_client.py`。

## 云端部署映射 / Cloud deploy mapping

本目录是**源真相**；上云时 rsync 进 starVLA repo（data_registry 需被 starVLA import，
config_yaml 相对 repo cwd 读取）：

```
starvla/configs/        →  $REPO/examples/SO101_PickOrange/train_files/configs/
starvla/common/         →  $REPO/examples/SO101_PickOrange/train_files/{modality.json,data_registry/}
```

数据集 `/root/autodl-tmp/datasets/`（LeRobot v2.1，经 v2.0 路径读），输出
`/root/autodl-tmp/starvla-outputs/`，base `/root/autodl-tmp/models/<backbone>/`。

## StarVLA repo 源码 patch（不在本目录）/ upstream patches

`dependencies/starVLA` 已转 submodule + 3 patch 维护，见 `patches/starvla/`（apply 顺序见其 README）：
- **0001** 224→448 vision 死穴（橙子 10–40px，224 判死）
- **0002** dataloader `num_workers 16→4 / prefetch 4→2`（16 worker 爆 62G RAM-cap）
- **0003** 原子 save + keep-last-N 裁剪（无裁剪填满盘 → ENOSPC 崩）

## Run-1 配方 / recipe（`so101_qwen_gr00t.yaml`）

`QwenGR00T`（Qwen3-VL-4B 冻 + GR00T flow-matching head，`freeze_modules: qwen_vl_interface`），
action/state_dim=6，horizon=16，bs=8，30k steps。实测 GPU 25/32G · util 97% · ~1.0s/step ·
loss(action_dit)~1.2 · ETA ~8.5–9h ≈ 6.6 epoch。

**结果**（strict 20-round，leaderboard 同口径）：E(🍊)/ep=35.0% (21/60)，P(3)=10% (2/20)。
倒 U 过拟合：峰值 ~15k、>21k 塌陷。发布于 [`wsagi/StarVLA-PickOrange`](https://huggingface.co/wsagi/StarVLA-PickOrange)。

## 加一个变体 / Add a variant

1. `env/dl_base.sh REPO=<vlm-repo-id>` 下 backbone（若与 Run-1 不同）。
2. 复制 `configs/so101_qwen_gr00t.yaml` → `configs/so101_<vlm>_<head>.yaml`，改 `framework.name`
   （= starVLA `model/framework/VLM4A/[VLM][Head].py` 的类名）+ `framework.<vlm>.base_vlm` + `run_id`。
3. `CONFIG=…/configs/so101_<vlm>_<head>.yaml RUN_ID=so101_<vlm>_<head> bash run_train.sh`。

### 实例：换更大的同族骨干 Qwen3-VL-8B（`so101_qwen3vl8b_gr00t.yaml`）

同 `framework.name: QwenGR00T` + 同 head，仅换冻结骨干 4B→8B。**starVLA 源码零改动**：
QwenGR00T 运行时把 `cross_attention_dim` 对齐到所加载 VLM 的真实 `model.config.hidden_size`
（8B = 4096），`vl_hidden_dim` 对本框架是死字段（仅 QwenPI/Adapter/Layerwise 那些 head 才回读）。
eval 的 `serve_starvla.py:repoint_base_vlm` 命中的 `framework.qwenvl.base_vlm` 对 8B 仍成立，**eval 也不改**。

```bash
REPO=Qwen/Qwen3-VL-8B-Instruct bash env/dl_base.sh        # 下 8B 骨干（~16G，冻结也要全下，见下）

# ① 先 500 步冒烟：拉回 ckpt-500 eval 看机械臂会不会动，再决定跑全程（沿用 Run-1 方法论）
CONFIG=$REPO/examples/SO101_PickOrange/train_files/configs/so101_qwen3vl8b_gr00t.yaml \
MAX_STEPS=500 SAVE_INTERVAL=500 RUN_ID=so101_qwen3vl8b_gr00t_smoke500 BATCH=4 bash run_train.sh

# ② 冒烟 OK 才跑全程 60k
CONFIG=$REPO/examples/SO101_PickOrange/train_files/configs/so101_qwen3vl8b_gr00t.yaml \
RUN_ID=so101_pickorange_qwen3vl8b_gr00t bash run_train.sh
```

config 相对 Run-1 只动 4 处：`base_vlm`、`vl_hidden_dim 2048→4096`（卫生）、`per_device_batch_size 8→4`、`run_id`。
**显存（48G 4090）**：8B 冻结权重 ~16G bf16，bs=4 时峰值 ~33–40G 安全（bs=8 ~41–50G 踩边/OOM）。
注意三处**别误算省显存**：Qwen3 路径 `attn` 被 `QWen3.py` 强制 `sdpa`（flash-attn 配了无效）、
`trainer.gradient_checkpointing` 是死字段且冻结 VLM 无 backward 可省、单卡 ZeRO-2 不分片参数。
真正峰值大头是 `output_hidden_states=True` + `repeated_diffusion_steps`（默认 8）把 4096 宽 hidden
repeat 8 倍喂 DiT cross-attn——**OOM 时第一杠杆是 `repeated_diffusion_steps` 降到 4，再动 batch**。
bs 砍半 → `max_train_steps` 翻到 60k 才覆盖 Run-1 同等 ~6.6 epoch，~1.5s/step ≈ 16–20h。

> eval 侧脚：`serve_starvla.py` 的 `--base` 走 `os.path.abspath`，**必须传本地 8B 目录**（传 HF repo id
> 会被拼成假本地路径）；既有行为，非 8B 新增。

⚠️ **eval 待修的特殊情况**：`serve_starvla.py:repoint_base_vlm` 现硬编码 `framework.qwenvl.base_vlm`
key。加非 Qwen 变体时需改成从 config 读 framework 类型再定位 `base_vlm` 键，否则 repoint 失效。

⚠️ **Gemma4+PI_v3 现实**：无现成 `Gemma4PI_v3.py`（只有 `Gemma4PI.py` v2 与绑 Qwen 的
`QwenPI_v3.py`），且 Gemma4 骨干**无官方 Bridge 预训练 base**。最高 ROI 替代见
[[starvla-so101-cloud-training]] 第二轮段（从 `StarVLA/Qwen3VL-PI_v3-Bridge-RT_1` init）。
