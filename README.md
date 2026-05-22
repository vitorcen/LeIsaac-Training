# LeIsaac — vitorcen fork

[LightwheelAI/leisaac](https://github.com/LightwheelAI/leisaac) (Apache-2.0) 的 fork。在 upstream 提供的 SO-101 遥操 + GR00T N1.5/N1.6 微调配方之上，扩展了**通用 LeRobot 微调脚手架**、**PickOrange 多策略横评**、以及一组让非平凡 VLA 能在 Isaac Sim 上跑通的 client 端补丁。
_A fork of [LightwheelAI/leisaac](https://github.com/LightwheelAI/leisaac) (Apache-2.0). Extends the upstream SO-101 teleop + GR00T fine-tune recipes with a generic LeRobot fine-tune scaffold, a PickOrange multi-policy benchmark, and client-side fixes that make non-trivial VLAs evaluable in Isaac Sim._

![ACT eval — SO-101 PickOrange](docs/assets/pick-orange.jpg)

- **Upstream / 原仓库**: https://github.com/LightwheelAI/leisaac
- **Upstream docs**: https://lightwheelai.github.io/leisaac/
- **本 fork**: https://github.com/vitorcen/LeIsaac

原 LeIsaac repo 已包含 SO-101 Isaac Sim 遥操作和 `LeIsaac-SO101-PickOrange-v0` 任务的 GR00T N1.5 / N1.6 微调配方。本 README 只描述**本 fork 在 upstream 之上新增的内容**。Upstream 原生功能请看 [upstream docs](https://lightwheelai.github.io/leisaac/)。
_The original LeIsaac repo already covers SO-101 teleop + GR00T fine-tuning. This README only covers **what this fork adds on top of upstream**._

---

## 本 fork 新增内容
_What this fork adds_

### 1. 通用 LeRobot 微调脚手架
_Reusable LeRobot fine-tune scaffold_

端到端、环境变量驱动的脚本：拉 dataset → v2.1→v3.0 转换 → `lerobot-train`。同一套 scaffold 适配 SmolVLA / ACT / Diffusion Policy / DiT / 以及（多一步准备）未来其他 LeRobot policy。
_End-to-end, env-driven scripts for dataset pull → v2.1→v3.0 conversion → `lerobot-train`. The same scaffold works for SmolVLA / ACT / Diffusion Policy / DiT and (with one prep step) other LeRobot policies._

| Script | Purpose |
| --- | --- |
| [`datasets/download.sh`](datasets/download.sh) | `bash datasets/download.sh <ORG>/<DATASET>` — 拉任何 LeRobot dataset 到 `datasets/raw/<basename>/` |
| [`datasets/convert_to_v30.sh`](datasets/convert_to_v30.sh) | v2.1 → v3.0 原地转换（lerobot ≥ 0.5.x 必须），幂等 |
| [`scripts/training/lerobot_finetune.sh`](scripts/training/lerobot_finetune.sh) | 通用 `lerobot-train` wrapper，所有 knob 走 env vars（`BASE_MODEL` / `DATASET_REPO_ID` / `STEPS` / `BATCH_SIZE` / `RENAME_MAP` / `EXTRA_ARGS` / ...） |
| [`scripts/training/smolvla/prepare_base.sh`](scripts/training/smolvla/prepare_base.sh) | SmolVLA 专用：clone `lerobot/smolvla_base` 后剥光 `input_features` + `empty_cameras` — 因为 draccus CLI override 是 dict-merge 不是 replace，原 base 自带的 `camera1/2/3 @ 256×256` 占位会污染微调路径 |

目录按语义分类（[[feedback-style]] 约定）：
_Directory layout follows semantic split:_
- `scripts/training/` = 从 pretrained base 微调 / fine-tune from a pretrained base
- `scripts/training/` = 从头训练 / train-from-scratch (ACT, Diffusion Policy, DiT)

详细文档：
- [`datasets/README.md`](datasets/README.md)
- [`scripts/training/README.md`](scripts/training/README.md)
- [`scripts/training/README.md`](scripts/training/README.md)

### 2. PickOrange 多策略横评
_PickOrange multi-policy benchmark_

把 `LeIsaac-SO101-PickOrange-v0` 当 benchmark，统一 eval harness 跑 7 个 baseline，3 round × 每 round 3 颗橙子 = 共 9 颗。
_Treating `LeIsaac-SO101-PickOrange-v0` as a benchmark; 7 baselines × 3 rounds × 3 oranges = 9 oranges total per policy._

**Eval config**: `eval_rounds=3`, `episode_length_s=120s` (sim time), `max_round_wall_s=180s` (wall-clock cap), step_hz=30 except GR00T family which uses step_hz=60 (per [§step_hz hypothesis](#-关键-inference-配置--policy_action_horizon32))。Eval 复现：`bash scripts/benchmark/run_all_baselines.sh`，详见 [`scripts/benchmark/`](../scripts/benchmark/)。

**Success criteria — 双口径** (snapshot 2026-05-18):
- **Strict ✅** = 全 3 颗 sticky `put_orange_to_plate` 至少捕到一帧（要求 EE-near + gripper-open + xy-in-plate 同时满足）— 严格下界，可能漏 <33ms 瞬态。
- **🍊 (n/9)** = sticky 累计计数，部分功劳。
- **Env (env-only)** = `task_done` (orange xyz in plate box + arm rest)，可能假成功（橙子被碰到盘边桌面、高度仍 ≈plate 时误判）。

_Sort: strict Rounds DESC → 🍊 DESC → time ASC._

**Note (2026-05-21)**：所有标 ✅ 5-round 的行用了 post-fix eval (≥5 round = 15 ep 必要 + pre-step placement snapshot + dz_max=0.20 + plate_r=0.10 + arm-only stuck + home_return)；标 ✗ 还是旧 3-round 数据待重测。see [`scripts/benchmark/README.html`](../scripts/benchmark/README.html).

| Policy | Params | `config.type` | Rounds | 🍊 | Pick rate | Avg round | Peak VRAM | Per-round detail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **[`hi-space/GR00T-N1.7-3B-Pick-Orange`](https://huggingface.co/hi-space/GR00T-N1.7-3B-Pick-Orange) (h=40, step_hz=60)** 🥇 ✅ | ~3B | `gr00t_n1_7` | **4/5** | **14/15** | **93.3%** | **69s** | 17.3 GB | 3🍊@35s✅ / 3🍊@35s✅ / 2🍊@180s / 3🍊@50s✅ / 3🍊@48s✅ |
| **[`wsagi/GR00T-N1.6-PickOrange`](https://huggingface.co/wsagi/GR00T-N1.6-PickOrange) (自训 / ours, ckpt-6500, h=40)** 🥈 ✅ | ~3B | `gr00t_n1_6` | **4/5** | **13/15** | **86.7%** | 92s | ~22 GB / 17.3 GB | 3🍊@58s✅ / 3🍊@70s✅ / 1🍊@180s / 3🍊@28s✅ / 3🍊@126s✅ |
| [`hi-space/GR00T-N1.6-3B-Pick-Orange`](https://huggingface.co/hi-space/GR00T-N1.6-3B-Pick-Orange) (h=40, step_hz=60) 🥉 ✅ | ~3B | `gr00t_n1_6` | 2/5 | 9/15 | 60.0% | 122s | 17.3 GB | 3🍊@31s✅ / 2🍊@41s✅ / 1🍊@180s / 1🍊@180s / 2🍊@180s |
| [`wsagi/SmolVLA-PickOrange`](https://huggingface.co/wsagi/SmolVLA-PickOrange) **(自训 / ours, main=15k, sweep best)** ✅ | ~450M | `smolvla` | 2/5 | 8/15 | 53.3% | 133s | 10.0 GB | 3🍊@85s✅ / 1🍊@180s / 3🍊@42s✅ / 0🍊@180s / 1🍊@180s |
| [`edge-inference/smolvla-so101-pick-orange`](https://huggingface.co/edge-inference/smolvla-so101-pick-orange) ✅ | ~450M | `smolvla` | 1/5 | 6/15 | 40.0% | 135s | 10.2 GB | 3🍊@93s✅ / 1🍊@180s / 0🍊@42s(stuck) / 0🍊@180s / 2🍊@180s |
| **[`wsagi/ACT-PickOrange`](https://huggingface.co/wsagi/ACT-PickOrange) (自训 / ours, lerobot v0.4.0 ckpt-18k, h=70)** ✅ | ~52M | `act` | **1/5** | **5/15** | **36.7%** | 154s | 8.6 GB | 4-run trimmed mean (剔除 lucky 13/15 outlier)；per-run strict [3,0,2,1,2] oranges [13,2,8,5,5]，含 outlier mean=2/7，剔除后 1/5 5/15。pool 含 outlier 33/75=44.0%, Welch t vs shadowHokage p=0.034 ✅ |
| [`LightwheelAI/leisaac-pick-orange-v0`](https://huggingface.co/LightwheelAI/leisaac-pick-orange-v0) **(N1.5, h=16, step_hz=60)** ✅ | ~3B | `gr00t_n1_5` | 0/5 | 8/15 | 53.3% | 140s | 16.2 GB | 1🍊@180s / 2🍊@60s(stuck) / 2🍊@180s / 1🍊@101s(stuck) / 2🍊@180s |
| [`shadowHokage/act_policy`](https://huggingface.co/shadowHokage/act_policy) **(others, h=70)** ✅ | ~52M | `act` | 0/5 | 4/15 | 26.7% | 169s | 8.6 GB | h=70 single 5-round, per-ep [0,0,1,0,3] = 4 oranges. h-sweep {16,32,64} 也都 0/5 strict 2-3/15 oranges。旧 9/15 anchor 是 sticky-env-success 协议（env→success 但只 1-2 颗 placed 也算 strict），post-bug-fix 严格协议下 0/5 strict |
| [`wsagi/X-VLA-PickOrange`](https://huggingface.co/wsagi/X-VLA-PickOrange) **(自训 / ours, weak-aug 17k, h=32, no-stuck)** ✅ | 0.9B | `xvla` | 0/5 | 3/15 | 20.0% | 180s | ~5 GB | 1🍊@180s / 0🍊@180s / 0🍊@180s / 1🍊@180s / 1🍊@180s |
| [`wsagi/DiffusionPolicy-PickOrange`](https://huggingface.co/wsagi/DiffusionPolicy-PickOrange) **(自训 / ours, v0.5)** ✅ | ~267M | `diffusion` | 0/5 | 0/15 | 0.0% | 33s* | 10.6 GB | all stuck @ 33s — **真因不是 policy dead**，是 lerobot async server bug: `predict_action_chunk` 不调 `populate_queues` → n_obs_steps=2 stack 空 deque → server stream crash → client 8-retry 拿不到 action → 静默"不动"。fix 见 [[lerobot-dp-async-server-bug]] |
| DP **(自训 / ours, v0.4 fullres patched, ckpt-70k)** ✅ | ~267M | `diffusion` | 0/5 | 0/15 | 0.0% | 183s | ~13 GB | crop_shape=null + 上述 server patch + 100k step 重训，ckpt-70k 5-round 在 h={8,12,16} 全 0/15。framework bug fix 让 policy 能产合理 action (joint ±0.6 rad) 但 50 demo 上 DP 学不到稳定 task。watcher 3-round h=8 sweep 在 60/70/80k 各拿 1/9 后被 5-round 全部证伪为 single-ep noise |
| OpenVLA-7B **(自训 / ours, ckpt-6300 / 8bit-1000, h=16, no-stuck)** ✅ | 7B + 32 LoRA | `openvla` | 0/5 | 0/15 | 0.0% | 180s | ~8 GB | all wall_cap (both ckpts identical: 0) |
| π0.5 **(自训 / ours)** — pt-v3 final_lora.npz, h=35 ✅ | 3.36B + 5M LoRA | `pi05` | 0/5 | 0/15 | 0.0% | 180s | ~16 GB | all wall_cap 0/3 |

> 历史快照在 [`results/benchmark/snapshots/`](../results/benchmark/snapshots/) — 包含 round 1 (step_hz=30 全部) / round 2 (sticky-strict + GR00T step_hz=60 fix)。原始 JSON + 1Hz GPU CSV 都在内。

**核心结论 / Headlines**：

- 🥇 **wsagi 自训 GR00T-N1.6 (ckpt-6500) 是新 SOTA** — 2/3 strict, **8/9** 🍊, avg 115s/round。在 4090 24GB 上极限挤进 N1.6 全参 FT（bf16 + grad-ckpt use_reentrant=False + adafactor + DISABLE_ADDMM_CUDA_LT=1 + watchdog auto-resume），同 strict 但 +2🍊 vs hi-space baseline。
- 🥈 hi-space N1.6 (公开 baseline) — 2/3 strict, 6/9 🍊, avg 96s。同 family、同 strict，但少 2 颗 🍊 — N1.6 family 上限随训练投入提升。
- ⚙️ **step_hz=60 对 GR00T 系列关键**：N1.5 step_hz=30 → 1🍊；step_hz=60 → 4🍊（4x boost）。dataset 是 30fps 但 GR00T 的 chunk action 输出预计高于 30Hz 应用以达自然速度。ACT/SmolVLA/DP 在 30Hz 表现一致，未做 60Hz sweep。
- 🟡 **SmolVLA (self) 数据上限 5/9** ≫ SmolVLA (other) 0/9 — 同架构差异完全来自训练（local 30k step vs edge-inference 早期 ckpt）。
- ⚠️ **80M ACT 当前 0/3** — 但记忆里 horizon=32 配合曾经 1/1。回归疑似来自 `sim_warmup_steps=30` 默认值变化（commit 1e1bae6）— 仍在 diagnose。
- 🍊 **第 3 颗橙子普遍卡** — dataset 60 ep × 每集 1 次"放最后一颗"演示导致；与历史结论一致（**数据问题，不是模型问题**）。

详细 debug / hypothesis tracking：
- step_hz=30 vs 60 对 GR00T 的影响 — see [`docs/training/policy_step_hz_postmortem.html`](docs/training/) (TBD)
- sticky vs env.task_done 双口径 — see [`scripts/benchmark/aggregate.py`](../scripts/benchmark/aggregate.py)
- 完整 reproducer：[`scripts/benchmark/run_all_baselines.sh`](../scripts/benchmark/run_all_baselines.sh)

更详细的 round-by-round eval 数据 + DiT / SmolVLA2 / Octo / RDT 后续 priority 见 [`docs/finetune/policy_comparison_priorities.html`](docs/finetune/policy_comparison_priorities.html)。

### 3. 设计文档与 postmortem
_Design docs and postmortems_

- [`docs/training/act_eval_debug_postmortem.html`](docs/training/act_eval_debug_postmortem.html) — ACT eval 三个 sim-side 根因（`sim_warmup` / `step_hz=30` / `policy_action_horizon`）的完整诊断
  _Three sim-side root causes diagnosed for ACT eval._
- [`docs/training/dp_inference_speedup_and_dynamic_timeout.html`](docs/training/dp_inference_speedup_and_dynamic_timeout.html) — Diffusion Policy DDPM→DDIM hot-swap（393→147 ms/chunk）+ user-patience-cap eval timeout 完整 postmortem，含 SVG 拟合曲线
  _DP inference speedup via DDPM→DDIM hot-swap + dynamic timeout postmortem, with inline SVG fit curves._
- [`docs/finetune/smolvla2_finetune_pick_orange.html`](docs/finetune/smolvla2_finetune_pick_orange.html) — SmolVLA 微调 v1 失败 / v2 部分成功 + schema-free base recipe
- [`docs/finetune/policy_comparison_priorities.html`](docs/finetune/policy_comparison_priorities.html) — 横评 + DiT / SmolVLA2 / Octo / RDT-1B 后续优先级

### 4. LeIsaac client 端补丁
_LeIsaac client-side fixes for non-trivial VLAs_

主要改动在 `source/leisaac/leisaac/policy/service_policy_clients.py` 和 `scripts/evaluation/policy_inference.py`。Upstream LeIsaac 只针对 GR00T 验证；SmolVLA / DP / 我们的 ACT 暴露出几个 edge case：
_Main changes in `service_policy_clients.py` and `policy_inference.py`. Upstream only validates against GR00T; SmolVLA / DP / our ACT exposed several edge cases:_

- **Auto camera schema mapping** — `_build_camera_feature_map(ckpt_path, sim_cameras)` 读 ckpt 的 `config.json` 自动判别命名风格：natural keys (front/wrist) → 不 rename；占位 keys (camera1/2/3) → 位置式 rename + 给未用 slot 补零 → 避免 `KeyError: 'camera1'`
- **`must_go=True` on every observation** — 绕过 server 的 "Observation too similar to last obs predicted" dedup filter；不绕过的话 sim 静止帧会被 server 主动丢弃，client 永远拿不到 action
- **Bounded retry without deadlock** — `_receive_action()` 重试 8× (200ms cap) 应对首次慢推理；去掉了原版 `skip_send_observation` flag（一次重试耗尽后死锁 client）
- **`run_eval.sh` wrapper** — user-patience cap timeout (`startup + n_rounds × per_round`), inference probe + slowdown warning, ckpt config 自动解析 `n_action_steps` 得 effective_chunk

---

## ⚠️ 关键 inference 配置 — `policy_action_horizon=32`
_Critical inference setting — `policy_action_horizon=32`_

**对 chunk_size=100 的 ACT，LeIsaac.ipynb 默认 `policy_action_horizon=16` 是隐性陷阱。** 第二颗橙子永远过不去（爪子抖 / muting）。
_**For ACT with chunk_size=100, LeIsaac.ipynb's default `policy_action_horizon=16` is a hidden trap.** The policy deadlocks on the 2nd orange (gripper jitter / muting effect)._

### 根因 / Root cause

ACT 每 chunk 输出 100 步动作（一段**完整规划**）：前 ~10 步是"启动 / 加速"，中段 (step 20-80) 才是真正的**宏观运动**（接近 → 夹起 → 提起 → 运送 → 释放）。LeRobot async client 用直接窗口 receding horizon，每 `policy_action_horizon` 步重新 query 一次。
_Each ACT chunk outputs a 100-step planned trajectory: first ~10 steps = startup, steps 20-80 = macro-motion. LeRobot async client uses sliding-window receding horizon, re-querying every `policy_action_horizon` steps._

| horizon | 1st orange | 2nd orange | 3rd orange | 1/1 |
| --- | --- | --- | --- | --- |
| 8 | 🔴 卡死（夹着不动） | — | — | 0/1 |
| 16 (LeIsaac.ipynb 默认 / default) | ✅ | 🟡 muting / 爪子抖 | — | 0/1 |
| **32 (推荐 / recommended)** | ✅ | ✅ 折腾后成功 | ✅ 折腾后成功 | **1/1 ✅** |

### 推荐设定 / Recommended

```bash
--policy_action_horizon=32        # 不要用默认 16 / NOT the default 16
--step_hz=30                      # 对齐 dataset 30Hz / matches dataset
--episode_length_s=120
```

完整诊断和 chunk-execution 假设的 falsification 推理见 [`docs/training/act_eval_debug_postmortem.html`](docs/training/act_eval_debug_postmortem.html)。
_Full diagnosis and falsification reasoning in the postmortem doc._

---

## Quick start

```bash
# 1) 拉 dataset / Pull dataset (LeIsaac SO-101 PickOrange, ~670 MB)
bash datasets/download.sh
bash datasets/convert_to_v30.sh

# 2a) 从头训练 ACT / Train ACT from scratch (~5h on RTX 4090)
bash scripts/training/act/train.sh

# 2b) 或：从头训练 Diffusion Policy / Or: train DP from scratch
bash scripts/training/diffusion_policy/train.sh

# 2c) 或：微调 SmolVLA / Or: fine-tune SmolVLA from base
bash scripts/training/smolvla/prepare_base.sh
BASE_MODEL=$(pwd)/outputs/.bases/smolvla_base_no_features \
DATASET_REPO_ID=LightwheelAI/leisaac-pick-orange \
OUTPUT_NAME=smolvla-leisaac-pick-orange \
STEPS=30000 BATCH_SIZE=8 NUM_WORKERS=2 SAVE_FREQ=5000 \
EXTRA_ARGS='--dataset.video_backend=pyav' \
bash scripts/training/lerobot_finetune.sh

# 3) 启 LeRobot async server / Start LeRobot async server
bash ~/work/isaaclab-experience/scripts/policy_server.sh start lerobot

# 4) Isaac Sim eval — 注意 horizon=32 / Note horizon=32!
cd ~/work/isaaclab-experience/LeIsaac && \
  bash scripts/evaluation/run_eval.sh -- \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --eval_rounds=3 --episode_length_s=120 --step_hz=30 \
    --policy_type=lerobot-act \
    --policy_host=127.0.0.1 --policy_port=8080 \
    --policy_timeout_ms=10000 \
    --policy_language_instruction='Pick up the orange and place it on the plate' \
    --policy_checkpoint_path=$(pwd)/outputs/act-leisaac-pick-orange/checkpoints/010000/pretrained_model \
    --policy_action_horizon=32 --device=cuda --enable_cameras
```

upstream 原生功能（teleoperation / datagen state machine / GR00T 微调等）请直接看 [upstream docs](https://lightwheelai.github.io/leisaac/)。
_For upstream-native features, see upstream docs._

---

## 致谢 / Acknowledgements

本 fork 基于 [LightwheelAI/leisaac](https://github.com/LightwheelAI/leisaac)，其自身致谢 [IsaacLab](https://github.com/isaac-sim/IsaacLab) 与 [LeRobot](https://github.com/huggingface/lerobot)。所有 upstream 贡献者保留其归属；本 fork 的改动是**增量式**的（新增 scripts / docs + 一组小补丁到 LeRobot service client）。
_This fork builds on [LightwheelAI/leisaac](https://github.com/LightwheelAI/leisaac); all upstream attributions retained. Changes here are additive (new scripts/docs + small client-side patches)._

## 引用 / Citation

按 upstream 约定引用：
_Cite the upstream project per their convention:_

```bibtex
@software{Lightwheel_and_LeIsaac_Project_Developers_LeIsaac_2025,
  author = {{Lightwheel} and {LeIsaac Project Developers}},
  license = {Apache-2.0},
  title = {{LeIsaac}},
  url = {https://github.com/LightwheelAI/leisaac},
  version = {0.4.0},
  year = {2026}
}
```

ACT 论文：
_ACT paper:_

```bibtex
@inproceedings{zhao2023learning,
  title={Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  author={Zhao, Tony Z. and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  booktitle={Robotics: Science and Systems},
  year={2023}
}
```

## License

Apache-2.0，与 upstream 一致 / same as upstream. 见 [LICENSE](LICENSE).
