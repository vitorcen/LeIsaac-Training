# LeIsaac 训练脚本 · Training scripts

LeIsaac 数据集上的策略训练入口，**from-scratch 和 fine-tune 统一在这里**。

```
scripts/training/
├── lerobot_finetune.sh   # 通用 launcher（所有策略共用底层）
├── act/                  # ACT — from-scratch（社区 1/1 baseline）
├── diffusion_policy/     # Diffusion Policy — from-scratch（概率 0/3~3/3）
├── dit/                  # multi_task_dit — from-scratch
├── smolvla/              # SmolVLA v1/v2 — fine-tune（含 prepare_base.sh）
└── openpi/               # π0 / π0.5 / π0-FAST — fine-tune (PyTorch / MLX)
```

底层 launcher 是 `lerobot_finetune.sh`：

- 传 `BASE_MODEL=<HF_repo>` → fine-tune 路径
- 传 `POLICY_TYPE=<name>` → from-scratch 路径
- 两者互斥 / mutually exclusive

## 前置 · Prerequisites

```bash
# 1) 下载数据集
bash datasets/download.sh <ORG>/<DATASET>

# 2) 转 v3.0（如果是 v2.1 数据集）
bash datasets/convert_to_v30.sh <ORG>/<DATASET>
```

详见 [`datasets/README.md`](../../datasets/README.md)。

## Family-specific 入口（推荐普通用户）· Family entry points

| Family | 文档 | 命令 |
| --- | --- | --- |
| ACT | [`act/train.sh`](act/train.sh) | `bash scripts/training/act/train.sh` |
| Diffusion Policy | [`diffusion_policy/train.sh`](diffusion_policy/train.sh) | `bash scripts/training/diffusion_policy/train.sh` |
| DiT | [`dit/train.sh`](dit/train.sh) | `bash scripts/training/dit/train.sh` |
| SmolVLA | [`smolvla/README.md`](smolvla/README.md) | `bash scripts/training/smolvla/prepare_base.sh` + `BASE_MODEL=... bash scripts/training/lerobot_finetune.sh` |
| π0 / π0.5 | [`openpi/README.md`](openpi/README.md) | `bash scripts/training/openpi/pytorch/train.sh` |

## 后续训练计划 · Roadmap

完整路线图（含分级 / 维度矩阵 / 横评空白点分析）见
[`docs/training/training_roadmap.html`](../../docs/training/training_roadmap.html)。

下面按"是否 LeRobot 原生 / 是否可转 LeRobot"分类。

### A. LeRobot ≥0.5 原生支持，待训 · Native, pending

LeRobot 仓库 `lerobot/policies/` 已自带、可直接 `POLICY_TYPE=<name>` 走 `lerobot_finetune.sh`：

| 优先级 | Policy | Backbone / Arch | 备注 |
| --- | --- | --- | --- |
| ⭐⭐ S | `vqbet` | ResNet + VQ-VAE + GPT-2 | 离散动作横评空白点，<200M，4090 4-6h 可训 |
| ⭐⭐ S | `pi0` | PaliGemma + flow | π0.5 的"前一代"对照，验证 v1→v2 升级是否有效 |
| ⭐ A | `pi0_fast` | PaliGemma + FAST 离散 token | 跟 π0 / π0.5 三角对照"同 backbone 不同 action head" |
| ⭐ A | `wall_x` | Qwen2.5-VL + flow | 新代 VL backbone，开源版 LingBot 替身 |
| ⭐ A | `multi_task_dit` | DiT (Diffusion Transformer) | 跟 Diffusion Policy 直接对比 UNet1D vs DiT |
| 探索 B | `xvla` | Florence2 + soft transformer | 独特 VL 路线，扩展 "different VL pretrain" 维度 |

**示例命令**（待写各自 `<policy>/train.sh` 包装）：
```bash
# from-scratch
POLICY_TYPE=vqbet OUTPUT_NAME=vqbet-leisaac-pick-orange \
  bash scripts/training/lerobot_finetune.sh

# fine-tune from HF base
BASE_MODEL=lerobot/wall_x_base OUTPUT_NAME=wall_x-leisaac-pick-orange \
  bash scripts/training/lerobot_finetune.sh
```

### B. 外部 VLA，需自建 server + client 接入 · External VLAs

不在 LeRobot 范围，需独立训练 + 服务化（参照 `server/pi05_leisaac/` 模式）：

| 优先级 | Model | 详细计划 |
| --- | --- | --- |
| ⭐⭐ S | OpenVLA-7B | [`docs/training/openvla_finetune_plan.html`](../../docs/training/openvla_finetune_plan.html) — 7B 离散 token，单臂直接可用 |
| 探索 B | LingBot-VLA (Ant Robbyant) | [`docs/training/lingbot_vla_finetune_plan.html`](../../docs/training/lingbot_vla_finetune_plan.html) — 双臂only，需 swap 单臂 head |

> NVIDIA DreamZero-14B 已评估为**不可行**（4090 训练 / 推理都过不去），详见调研结论。
> _NVIDIA DreamZero-14B has been ruled out — 4090 cannot host training or inference._

### C. 不在路线图 · Out of scope

- **`sac` / `tdmpc`**：纯 RL，需要 reward function + env 闭环，与我们 demo-only dataset 不匹配
- **`sarm`**：是 reward model 不是 policy（给 RL 用）
- **`rtc`**：是推理 wrapper，所有 chunk-based policy 都可套用，不构成独立 baseline
- **`pi_gemma.py`**：π0 系列内部 backbone 实现，非独立 policy

### 横评维度空白点 · Benchmark gaps

按"参数规模 × 动作表示"二维矩阵，<b>"Discrete token autoregressive" 整行目前是空白</b>。
新增 `vqbet`（tiny 档）+ `OpenVLA-7B`（large 档）后这条线就立起来 — <b>当前最大信息增益方向</b>。
_The "discrete token autoregressive" row is currently empty across all parameter sizes — filling it via VQ-BeT (tiny) + OpenVLA-7B (large) is the highest-information-gain next step._
