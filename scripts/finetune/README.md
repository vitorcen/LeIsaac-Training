# LeIsaac fine-tune 脚本

适用于**有预训练 backbone** 的 VLA / 策略模型在 LeIsaac 数据集上的微调。

```
scripts/finetune/
├── lerobot_finetune.sh   # 通用 launcher（finetune + train 共用）
├── smolvla/              # SmolVLA v1 / v2 微调（含 prepare_base.sh）
└── openpi/               # π0 / π0.5 / π0-FAST 微调
```

底层 launcher 是 `lerobot_finetune.sh`：

- 传 `BASE_MODEL=<HF_repo>` → fine-tune 路径
- 传 `POLICY_TYPE=<name>` → from-scratch 路径（被 [`scripts/train/`](../train/) 用上）
- 两者互斥

## 前置

```bash
# 1) 下载数据集
bash datasets/download.sh <ORG>/<DATASET>

# 2) 转 v3.0（如果是 v2.1 数据集）
bash datasets/convert_to_v30.sh <ORG>/<DATASET>
```

详见 [`datasets/README.md`](../../datasets/README.md)。

## Family-specific 入口（推荐普通用户）

| Family | README | 入口 |
| --- | --- | --- |
| SmolVLA | [`smolvla/README.md`](smolvla/README.md) | `bash scripts/finetune/smolvla/prepare_base.sh` + 用 `BASE_MODEL` 走 `lerobot_finetune.sh` |
| OpenPI (π0 / π0.5) | [`openpi/README.md`](openpi/README.md) | 脚手架维护在 [`vitorcen/pi05-mlx-experience`](https://github.com/vitorcen/pi05-mlx-experience) |

## 通用 launcher（高级用户）

```bash
bash scripts/finetune/lerobot_finetune.sh
```

行为完全由环境变量驱动；下表是全部 knob：

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `BASE_MODEL` | `lerobot/smolvla_base` | `--policy.path` 值（HF repo 或本地目录）|
| `POLICY_TYPE` | (空) | `--policy.type` 值，触发 from-scratch；与 `BASE_MODEL` 互斥 |
| `DATASET_REPO_ID` | `LightwheelAI/leisaac-pick-orange` | `--dataset.repo_id` |
| `DATASET_ROOT` | `datasets/raw/<basename>` | 本地 v3.0 数据路径 |
| `OUTPUT_NAME` | `<base>-<dataset>` | 输出子目录名（位于 `outputs/`）|
| `STEPS` | `20000` | 训练步数 |
| `BATCH_SIZE` | `64` | per-device batch |
| `NUM_WORKERS` | `4` | dataloader workers |
| `SAVE_FREQ` | `5000` | ckpt 保存间隔 |
| `RENAME_MAP` | (空) | JSON dict：sim 键 → policy 期望键。仅当 base 模型不用自然键时需要 |
| `EXTRA_ARGS` | (空) | 透传给 `lerobot-train` 的额外 flag |
| `CONDA_ENV` | `lerobot` | conda env 名 |

最终 ckpt：`outputs/<OUTPUT_NAME>/checkpoints/last/pretrained_model`。

## 推理验证

训完用 `policy_inference.py` 跑仿真。模型类型 `lerobot-<model_type>`（由 ckpt 的 `config.json` 的 `type` 字段决定）：

```bash
# 1) 启动 LeRobot policy_server（端口 8080）
bash ~/work/isaaclab-experience/scripts/policy_server.sh start lerobot

# 2) 启动 Isaac Sim 客户端
cd ~/work/isaaclab-experience/LeIsaac && \
  PYTHONUNBUFFERED=1 python -u scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --eval_rounds=10 --episode_length_s=120 \
    --policy_type=lerobot-smolvla \
    --policy_host=127.0.0.1 --policy_port=8080 \
    --policy_timeout_ms=15000 \
    --policy_language_instruction='Pick the orange to the plate' \
    --policy_checkpoint_path=/path/to/outputs/<OUTPUT_NAME>/checkpoints/last/pretrained_model \
    --policy_action_horizon=16 --device=cuda --enable_cameras
```

LeIsaac 客户端的 `_build_camera_feature_map(ckpt_path, sim_cameras)` 会读 ckpt 的 `config.json/input_features`，自然键命中时返回 `None`，否则自动构造 rename map——所以推理 client 端不需要再传 RENAME_MAP。
