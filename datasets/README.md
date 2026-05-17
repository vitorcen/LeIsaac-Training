# LeIsaac fine-tune datasets

公开 LeRobot 数据集的本地下载脚手架。本目录为微调（`scripts/training/`）提供输入。

## 目录

```
datasets/
├── README.md                # 本文件
├── download.sh              # 从 HF 下载任意 LeRobot 数据集到 raw/
├── convert_to_v30.sh        # v2.1 → v3.0 in-place 转换（lerobot ≥0.5 要求）
└── raw/                     # 实际数据（git-ignored）
    └── <basename>/          # 例如 leisaac-pick-orange/
```

`raw/` 整目录被 gitignore，仅保留 `.gitkeep` 占位。

## 典型流程

```bash
# 1) 下载（默认 LightwheelAI/leisaac-pick-orange，~670 MB）
bash datasets/download.sh

# 2) 转换格式（仅 v2.1 数据集需要；v3.0 直接 no-op）
bash datasets/convert_to_v30.sh
```

下载其他数据集：

```bash
bash datasets/download.sh         <ORG>/<DATASET>
bash datasets/convert_to_v30.sh   <ORG>/<DATASET>
```

参数也支持环境变量：`REPO_ID=foo/bar bash datasets/download.sh`。

## 命名约定

- HF repo `LightwheelAI/leisaac-pick-orange` → 本地路径 `raw/leisaac-pick-orange/`
- 转换后 v2.1 备份在同级目录 `raw/leisaac-pick-orange_old/`

## 已知数据集（截至 2026-05）

| 数据集 | 说明 | 场景 |
| --- | --- | --- |
| `LightwheelAI/leisaac-pick-orange` | 60 ep，SO-101 抓橙子放盘子 | `LeIsaac-SO101-PickOrange-v0` |

新增数据集时把它登记进本表。
