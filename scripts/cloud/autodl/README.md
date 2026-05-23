# AutoDL cloud training scaffolds

LeIsaac VLA finetune 在 AutoDL 上的最小可复用脚手架。配套 `LeIsaac/docs/training/autodl_cloud_finetune_playbook.html`。

## 推荐路径（v2，全程无卡 → 切 GPU 只为训练）

| 步骤 | 在哪跑 | 脚本 | 说明 |
|------|--------|------|------|
| 1. **本机** 预下 uv cache（一次性，10-30 min）| 本机 | `local/prefetch_uv_cache.sh` | 本机 uv sync + 打包成 tar.gz |
| 2. AutoDL 无卡模式开机（选 140 GB 盘） | AutoDL 面板 | — | ¥0.1/h，便宜 |
| 3. AutoDL 环境引导 | AutoDL | `bootstrap.sh` | uv / git-lfs / hf cli + .bashrc |
| 4. clone repo | AutoDL | `prep_repo.sh` | 浅克隆 + Isaac-GR00T submodule |
| 5. 下 dataset（公开）| AutoDL | `download_dataset.sh` | hf-mirror ~12 MB/s |
| 6a. 下 base model（云）| AutoDL | `download_gated_model.sh` | curl 单流，~2-3h（不推荐）|
| 6b. **推 base model（本机）** | 本机 | `local/scp_upload_model.sh` | ~15 min @ 5 MB/s ✓ |
| 7. **推 uv cache（本机）** | 本机 | `local/scp_upload_bundle.sh` | ~30 min @ 5 MB/s ✓ |
| 8. **AutoDL 无卡模式装 deps（离线）** | AutoDL | `uv_sync_offline.sh` | 2-5 min，零网络 |
| 9. verify | AutoDL | `verify_env.sh` | 21/21 ✓ |
| 10. 切 GPU 模式 + smoke 50 step | AutoDL | `train_n17.sh MAX_STEPS=50` | OOM 早暴露 |
| 11. 正式训练 | AutoDL | `train_n17.sh` + `resource_monitor.sh &` | 10k step + 自动 prune |
| 12. 训完 best ckpt 上 HF Hub | AutoDL | `upload_best_ckpt.sh` | 关机前必做 |
| 13. 训练 telemetry → 文档 | 本机 | scp `monitor.csv` 回本机 + `analyze_run.sh` | 生成 §7 实测数据 |

**关键改进 vs v1**: 步骤 7-8 把 uv sync 从 GPU 模式 30-60 min（被代理限速）改成无卡模式 2-5 min（离线）。
省 ~25 min GPU 租金 + 避开 AutoDL proxy 限速。

## 备用路径（v1）

如果本机不方便 uv sync（无 nvidia driver / 磁盘紧），用旧路径：步骤 1 跳过，步骤 8 改成 GPU 模式跑 `uv_sync.sh`（含 tensorrt build），预算 30-60 min。

## 脚本清单

| 文件 | 跑在 | 作用 |
|------|------|------|
| `local/prefetch_uv_cache.sh` | **本机** | 本机 uv sync + 打包成 tar.gz |
| `local/scp_upload_bundle.sh` | **本机** | uv cache bundle → AutoDL |
| `local/scp_upload_model.sh` | **本机** | HF cache 模型权重 → AutoDL |
| `bootstrap.sh` | AutoDL | uv + git-lfs + hf cli + .bashrc |
| `prep_repo.sh` | AutoDL | clone + lfs pull |
| `download_dataset.sh` | AutoDL | 公开 dataset |
| `download_gated_model.sh` | AutoDL | gated repo（备用，proxy 慢）|
| `uv_sync.sh` | AutoDL（**GPU 模式**）| 在线装 deps（v1 路径）|
| `uv_sync_offline.sh` | AutoDL（**无卡也行**）| 离线装 deps（v2 推荐路径）|
| `verify_env.sh` | AutoDL | sanity check |
| `resource_monitor.sh` | AutoDL | 训练时并行跑，记 CSV |
| `analyze_run.sh` | AutoDL/本机 | 训完生成 HTML telemetry snippet |
| `upload_best_ckpt.sh` | AutoDL | 训完推 HF |

## 关键 env vars（全在 bootstrap.sh 写入 .bashrc）

```bash
export HF_ENDPOINT=https://hf-mirror.com           # 公开 repo 默认走镜像
export HF_HOME=/root/autodl-tmp/hf_cache           # cache 落数据盘不爆系统盘
export HF_XET_HIGH_PERFORMANCE=1                   # 替代废弃的 HF_HUB_ENABLE_HF_TRANSFER
export PATH=/root/.local/bin:/root/miniconda3/bin:$PATH
```

## 安全提示

- HF token 不要 hard-code 进脚本，用 env `HF_TOKEN` 传入
- bootstrap.sh 完成后立即 `huggingface-cli logout` 删除 token 文件（或保留在 ~/.cache/huggingface/token 但记得 rotate）
- scp 上传脚本用 `sshpass -e SSHPASS=...` 而不是 inline `-p` 防止 ps 泄露

## 详细背景 / 故障排查

见 `LeIsaac/docs/training/autodl_cloud_finetune_playbook.html`（含 SVG 决策树 / 速度对比表 / 10 条故障 playbook）。
