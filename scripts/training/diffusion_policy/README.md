# Diffusion Policy 从头训练

[Diffusion Policy](https://arxiv.org/abs/2303.04137) 在 LeIsaac 数据集上的 from-scratch 训练。

## 工作流

```bash
bash scripts/train/diffusion_policy/train.sh
# 默认 100000 step, batch=32, resize_shape=[240,320]
```

或自定义:

```bash
STEPS=200000 BATCH_SIZE=64 OUTPUT_NAME=dp-pickorange-v3 \
bash scripts/train/diffusion_policy/train.sh
```

## 关键 default

- `POLICY_TYPE=diffusion` → 走 lerobot 的 `--policy.type=diffusion`，ResNet18 + UNet 默认架构
- `resize_shape=[240,320]` → 480×640 在 batch=32 上 OOM，下采样到 1/2 解决
- `video_backend=pyav` → torchcodec 长跑 segfault

## 已发布 ckpt

- [`wsagi/DiffusionPolicy-PickOrange`](https://huggingface.co/wsagi/DiffusionPolicy-PickOrange) — 100k step，eval 🟢 1/3 @ 60s（round 2 完成全任务）

## 推理

通过 LeRobot async-inference policy_server 提供，详见 HF 模型卡片。**需要 vitorcen/lerobot fork 的 DP patch**（`predict_action_chunk` self-populate queue），否则 server 会 `torch.stack([])` 失败。
