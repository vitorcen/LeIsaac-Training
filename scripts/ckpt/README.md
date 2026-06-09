# `scripts/ckpt/` — checkpoint disk-saving scaffolding (committed)

冻骨干训练（frozen-backbone）收尾后省存储的**唯一权威工具**。代码提交进仓库；产物（base / delta，几个 G 到几十 G）留在 gitignored 的 `outputs/`，**代码与产物分家**。

> **Why here, not in `outputs/`** — `LeIsaac/.gitignore` 有 `**/outputs/*`，整个 outputs 树不提交。工具早期放在 `outputs/_head_sweep_tools/` 里 = fresh clone 即丢、协作者拿不到。代码搬到提交目录 `scripts/ckpt/`，靠 `--args` 传路径（位置无关）；大 `.pt`/`.safetensors`（base、deltas、ckpts）继续留 `outputs/`（正确地被忽略）。

## 核心思想 / Core idea

冻骨干训练里 backbone 每个 step **逐字节相同**，可训的只有 head/expert（StarVLA action head、Wall-X expert、π0.5 expert…）。N 个 full ckpt = (N-1)× 死重。塌缩成：**1 个 base（冻结部分，存一次） + N 个 delta（只存变了的张量）**。

**方法 = 对 base 逐张量 diff，不是按前缀切**。这点是通用性的关键：
- **前缀干净**（StarVLA：`action_model.*`+`project_layers.*`）→ diff 出的 delta 与前缀法**逐键完全一致**（实测 0 个额外键）。
- **层内交错**（Wall-X expert 嵌在 `model.layers.*`、π0.5 嵌在 `paligemma_with_expert.*`，没有干净前缀）→ diff 照样精确隔离变动张量。
- **全量 FT**（ACT/DP/SmolVLA/X-VLA，无冻结骨干）→ delta ≈ 整个模型 → 工具**自动拒绝**抽取，提示只留 best。**防呆**：永远不会产出没用的 delta。

**重建按构造字节精确**：delta 内的键用 ckpt 的值，delta 外的键 == base 的值（== ckpt 的值，因为 delta 已捕获所有差异）。删任何 full 前还会跑一次显式 GOLD 张量相等校验，**全过才删**。

## 工具 / Tools

| 文件 | 作用 |
|---|---|
| `prune_ckpts.py` | 通用抽取+验证+裁剪。dry-run 默认，`--apply` 才删。`--min-frozen`（默认 0.5）以下判为全量 FT 直接跳过。支持 `.pt` 与 `.safetensors`。 |
| `merge_ckpt.py`  | 通用重建：`base + delta → full`，格式按扩展名推断（两种都支持）。 |

```bash
# 抽取（dry-run 看省多少 + GOLD）
python scripts/ckpt/prune_ckpts.py \
  --fulls 'outputs/<run>/checkpoints/steps_*_pytorch_model.pt' \
  --keep  'outputs/<run>/checkpoints/steps_<best>_pytorch_model.pt' \
  --base  'outputs/_head_sweep_tools/vlm_base_<fam>.pt' \
  --heads 'outputs/<run>/heads'
# 确认无误后真删非-best full
#   …上面命令 + --apply

# 临时重建某个非-best step 去 eval（eval 完即删，绝不覆盖 best）
python scripts/ckpt/merge_ckpt.py \
  outputs/_head_sweep_tools/vlm_base_<fam>.pt \
  outputs/<run>/heads/steps_<step>_..._head.pt \
  /tmp/steps_<step>_full.pt
```

## 防呆默认流程 / Foolproof default for **every new run**

新模型训练完，无脑跑一次 `prune_ckpts.py`：
1. 是冻骨干 → 自动塌缩成 base+deltas（GOLD 门控）。
2. 是全量 FT → 自动跳过，提示只留 best + 删中间 sweep ckpt（见 [[feedback-training-output-cleanup]]）。

**无损 + 续训不受影响**：工具只删 `model.safetensors`（base+delta 字节级可重建），**从不碰 `training_state/optimizer_state.safetensors`**。续训某个被抽过 head 的 step = `merge_ckpt` 还原 model + 现成 optimizer state，与删前完全一致。**`optimizer_state.safetensors` 是续训料，一律保留，别为省盘删**（用户 2026-06-09 明确）。

判据：full ckpt 是否 = 该 run 上榜/发布的 best？是→留；否→删（delta+base 随时字节级重建）。删前 `pgrep` 确认无训练/eval 进程在用。

## 全家族判定 / Per-family verdict (`outputs/`, 2026-06-09)

| 家族 | 架构 | frozen_frac | 处理 |
|---|---|---|---|
| **StarVLA** qwen35-2b/4b/9b · 8b-GR00T · 8b-PI_v3 · cosmos · pi_v3 · sweep | 冻 VLM + 可分离 head | ~0.55+ | ✅ **已做**（前缀法；现也被 diff 法覆盖，结果一致） |
| **Wall-X** sweep/oss05/smoke | freeze_vlm，expert 交错 | **0.945** | ✅ 适合 → diff 法（base 7.9G 共享 + delta 0.46G） |
| **π0.5-expert** | expert-only FT，交错 | **0.883** | ✅ 适合 → diff 法（base 8.3G + delta 1.10G） |
| **GR00T** N1.6/N1.7 | 冻 VLM + 扩散 head | 高 | ⏸️ 跳过：已发布 HF（有副本）+ 每 run 仅 2 ckpt + HF 分片 safetensors 格式摩擦，不值当；只留 best |
| **ACT/DP/SmolVLA/X-VLA** | 全量 FT | **~0.0** | ❌ 不适合抽 head（工具自动拒绝）→ 只留 best；**optimizer_state 保留（续训料，别删）** |
| **OpenVLA / dreamzero** | LoRA adapter | — | ❌ 无需：adapter 仅 65–208M，base 不在本地，已极小 |
| **pi05-leisaac-pt-v3** | 157M | — | ❌ 太小，无 ckpt |

关联记忆：[[feedback-vla-ckpt-best-only-head-rest]]（存储纪律规则）、[[feedback-training-output-cleanup]]（一族一 dir）、[[feedback-cloud-env-reuse-disk-cleanup]]（起训前清死重防 ENOSPC）。
