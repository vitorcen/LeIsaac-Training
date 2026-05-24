# π0.5 PyTorch expert-only FT（freeze-VLM）

主流 PyTorch 栈下重训 π0.5 —— 从 MLX LoRA (5M trainable, 0/15) → PyTorch
expert-only FT (693M trainable, ?/15)。与 GR00T 的"冻 VLM 训 DiT"同范式。

## 与之前 MLX LoRA 路径的关键差异

| 项 | MLX LoRA (老) | PyTorch expert-FT (新) |
|---|---|---|
| trainable | 5M (q/v_proj LoRA) | **693M (whole expert)** |
| backend | MLX (Apple Silicon) | PyTorch + lerobot-train |
| 冻什么 | 不冻，加 LoRA | **冻 PaliGemma 全部（VLM + LM + vision）** |
| 接口 | 自己写的 `pi05_leisaac.train` | lerobot 主线 + auto-eval watcher |
| Leaderboard | 0/15 | **目标 6-9/15（SmolVLA 同档）** |

## 用法

```bash
# 2500-step smoke (4090 24G)
bash scripts/training/pi05_pt/train.sh

# 10k full
STEPS=10000 SAVE_FREQ=1000 bash scripts/training/pi05_pt/train.sh
```

## 关键 hyperparams（来源 + 决策依据）

| 参数 | 值 | 来源 |
|---|---|---|
| `train_expert_only` | `true` | lerobot pi05 原生开关，冻整个 PaliGemma |
| `freeze_vision_encoder` | `true` | 显式（被 train_expert_only 覆盖也无害） |
| `gradient_checkpointing` | `true` | 必须 — 693M trainable + chunk=50 不开会 OOM |
| `chunk_size` / `n_action_steps` | 50 / 50 | openpi 默认 |
| `optimizer_lr` | 2.5e-5 | openpi cosine peak 默认 |
| `batch_size` | 1 | lerobot-train 无 grad-accum，要更大 batch 看 VRAM |
| `dtype` | bfloat16 | 与 frozen VLM 一致 |
| `max_state_dim` / `max_action_dim` | 32 / 32 | LeIsaac SO-101 6D 自动 pad |

## 后续 eval

训练 wrapper 自动 spawn `eval_watcher.sh`，每个 ckpt 跑 3-round 60s sanity，
连续 3 个 0-orange → `.eval_abort` 自动 SIGTERM 训练（见 LeIsaac/CLAUDE.md）。

**EVAL_HORIZON=35**（已在 train.sh 强制 export）：v3 MLX 验证过的 chunk-execution
甜点（chunk=50 训，部署只消费前 35 步），见 [`pi05_finetune_pick_orange.html`](../../../docs/training/pi05_finetune_pick_orange.html) §3。

## VRAM 实测（5-step smoke, 2026-05-23）

| 阶段 | VRAM | 备注 |
|---|---|---|
| 模型加载完（before training） | **9.4 GB** | bf16, 全 4B 参数 on GPU |
| 训练稳态 (batch=1, grad-ckpt) | **待实测**（5-step 太短没到稳态） | 估 16-22 GB（保守留 2GB buffer） |
| 4090 24G 余量 | ~2-8 GB | 够，但 batch=2 风险高 |

**实测训练速度**：~**0.4 s/step**（2.5 step/s, batch=1, 4090）

| 步数 | 估时 | ckpt 数 (SAVE_FREQ=STEPS/5) | disk |
|---|---|---|---|
| 2500 (smoke) | ~17 min | 5 × 9.4 GB | 47 GB |
| 10000 (full) | ~67 min | 5 × 9.4 GB | 47 GB |
| 30000 (large) | ~3.3 h | 5 × 9.4 GB | 47 GB |

⚠️ **每 ckpt 9.4 GB** —— lerobot 默认保存全 4B 参数（不只是 693M trainable expert）。`save_total_limit` 在 lerobot 框架里**不会自动滚动**，每个 ckpt 永久保留。后续按需手动 prune。

OOM 应急（如果稳态超 24G）：
1. 降 `--policy.chunk_size=32`（开销最大杠杆，但要重新对齐 eval horizon）
2. 关 cosine schedule 跑 constant lr（省 scheduler buffer）
3. 8-bit AdamW (`paged_adamw_8bit`)：能省 ~5 GB optimizer state，但 lerobot-train 默认是 AdamW

## 路径

| 用途 | 路径 |
|---|---|
| launcher | `scripts/training/pi05_pt/train.sh` |
| 共享框架 | `scripts/training/lerobot_finetune.sh` (BASE_MODEL=lerobot/pi05_base, EXTRA_ARGS) |
| 输出 | `outputs/pi05-expert-leisaac-pick-orange/` |
| auto-eval CSV | `outputs/<name>/auto_eval.csv` |
| 训练日志 | `outputs/.logs/<name>.log` |

## 关联

- [`pi05_finetune_pick_orange.html`](../../../docs/training/pi05_finetune_pick_orange.html) §2.5 新增 bf16 重训资源估算
- `pi05-pytorch-training` memory — 此次工作的延续
- GR00T `train.sh` — 同范式 (冻 VLM + 训 DiT)，参考其 grad-ckpt + bf16 策略
