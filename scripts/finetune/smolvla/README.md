# SmolVLA 微调

适配 HuggingFace 的 [SmolVLA](https://huggingface.co/lerobot/smolvla_base) 在 LeIsaac 数据集上的微调流程。

## 工作流

```bash
# 1) 准备一个"剥光"的 base：清掉占位的 input_features，方便后续按数据集真实键名自动填
bash scripts/finetune/smolvla/prepare_base.sh
# → outputs/.bases/smolvla_base_no_features/

# 2) 走通用 launcher 训练
BASE_MODEL=$PWD/outputs/.bases/smolvla_base_no_features \
DATASET_REPO_ID=LightwheelAI/leisaac-pick-orange \
OUTPUT_NAME=smolvla-leisaac-pick-orange \
STEPS=30000 BATCH_SIZE=8 NUM_WORKERS=2 \
EXTRA_ARGS='--dataset.video_backend=pyav' \
bash scripts/finetune/lerobot_finetune.sh
```

## 已知坑

- **base 的 input_features 必须先剥**：smolvla_base 自带 3 个占位 camera slot @ 256×256，draccus 的 dict-merge 语义让 CLI override 不生效，必须本地 clone 后剥掉
- **video_backend=pyav**：torchcodec + 4 worker 长跑会 segfault，改 pyav 稳定
- **batch=8 比 batch=64 快 7.5×**：PCIe / pyav IO 是瓶颈，不是 GPU

## 已发布 ckpt

- [`edge-inference/smolvla-so101-pick-orange`](https://huggingface.co/edge-inference/smolvla-so101-pick-orange) v1 — eval 0/3 @ 60s
- 本机 `smolvla-leisaac-pick-orange` 30k step — eval 🟡 2/5 @ 120s
