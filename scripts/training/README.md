# LeIsaac from-scratch 训练脚本

适用于**没有预训练 backbone** 的策略类（Diffusion Policy / DiT / ACT / VQ-BeT 等），直接在 LeIsaac 数据集上从随机初始化训练。

```
scripts/training/
├── diffusion_policy/    # Diffusion Policy（已发布 1/3 baseline）
├── dit/                 # multi_task_dit policy
└── act/                 # ACT（社区 1/1 baseline）
```

每个子目录的 `train.sh` 都是 `scripts/finetune/lerobot_finetune.sh` 的薄包装：固定 `POLICY_TYPE=<name>` + 各策略调好的 default（batch / resize / video backend），其余 knob 跟 `lerobot_finetune.sh` 一致。

## 对比：finetune vs train

| 类别 | 触发条件 | 入口 |
| --- | --- | --- |
| `finetune/` | 有预训练 base（SmolVLA / π0.5 / GR00T / RDT） | `scripts/finetune/<family>/` |
| `training/` | 从头训练 | `scripts/training/<policy>/` |

底层都调通用 `scripts/finetune/lerobot_finetune.sh`；finetune 路径设 `BASE_MODEL`，training 路径设 `POLICY_TYPE`，互斥。
