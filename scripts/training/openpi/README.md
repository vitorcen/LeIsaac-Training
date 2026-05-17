# OpenPI 系列微调（π0 / π0.5 / π0-FAST）

适配 Physical Intelligence 的 [OpenPI](https://github.com/Physical-Intelligence/openpi) 模型家族（包括 [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) 等 HF 镜像）在 LeIsaac 数据集上的微调流程。

## 当前状态

π0.5 LoRA fine-tune 的完整训练 + 推理脚手架（Mac MLX 训练 / Linux PyTorch 训练 / 双端 ZMQ 推理 server）维护在独立仓:

→ **[`vitorcen/pi05-mlx-experience`](https://github.com/vitorcen/pi05-mlx-experience)**（拟开源；当前路径 `~/work/pi05-mlx-experience/`）

## 现有 baseline 结果

| 策略 | 训练步数 | eval 3×60s |
| --- | --- | --- |
| GR00T N1.5 [`LightwheelAI/leisaac-pick-orange-v0`](https://huggingface.co/LightwheelAI/leisaac-pick-orange-v0) | — | ✅ 1/1 |
| π0.5 + LoRA r=16（MLX, 3000 step） | 3000 | ❌ 0/3 |
| π0.5 + LoRA r=16（MLX, 10000 step） | 10000 | ❌ 0/3 |

详情和缺陷分析见 pi05-mlx-experience 仓的 `docs/training_design.html`。

## 计划

- [ ] PyTorch 训练 pipeline（`scripts/training/openpi/pytorch/`）— 修掉 MLX recipe 的 6 个缺陷
- [ ] 全 lerobot processor pipeline（真 tokenizer / state input / AdamW / 多 camera）
- [ ] 跑通后输出 thin wrapper 调用 `lerobot_finetune.sh`（如果适配 `--policy.path=lerobot/pi05_base` + LoRA hook 可行）
