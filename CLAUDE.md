# LeIsaac Project Instructions

## Training: incremental sanity eval (mandatory, auto-on)

**Rule**: split total steps into **10 slices** (`SAVE_FREQ = STEPS/10`). After each ckpt, run X-VLA-style quick eval: `EVAL_ROUNDS=3 EPISODE_LENGTH_S=60 MAX_ROUND_WALL_S=90` (~3-5 min). Abort if 3 consecutive slices show 0 oranges or arm stuck < 30s.

**Default ON in `scripts/training/lerobot_finetune.sh`** (env `AUTO_EVAL=1`): the wrapper spawns `scripts/training/eval_watcher.sh` in background, which polls `$OUTPUT_DIR/checkpoints/` and runs quick eval per new ckpt. If 3 consecutive slices fail, watcher writes `$OUTPUT_DIR/.eval_abort` and the wrapper SIGTERMs training — so you do not burn N more hours on a broken config. CSV at `$OUTPUT_DIR/auto_eval.csv`, log at `$OUTPUT_DIR/auto_eval.log`. Override with `AUTO_EVAL=0` only when intentionally training a non-evalable variant.

**Why**: DP v0.4.0 trained 10h to 100k step with loss 0.554 → 0.011 (looked fine), but default `crop_shape=(84,84)` cropped 1.7% of frame → degenerate "do nothing" policy → 0/15 every eval. A single 10k-step quick eval would have caught it. **And** a parallel bug — `predict_action_chunk` in `lerobot/policies/diffusion/modeling_diffusion.py` does not call `populate_queues` (only `select_action` does), so n_obs_steps>1 + async server → `stack expects a non-empty TensorList` and arm never moves. Both v0.4 and v0.5 have this bug; the fix is to populate queues in `predict_action_chunk` before stacking (see commit on local lerobot-v040 checkout).

**Pre-flight**: diff new `train_config.json` against a known-good public baseline (shadowHokage/act_policy for ACT, wsagi/DiffusionPolicy-PickOrange for DP). Any field diff = decide deliberately. The DP `crop_shape: [84,84]` vs `None` would have been caught at train start.

**Reference**: X-VLA sweep `scripts/auto_sweep_xvla_ckpts.sh` is the canonical sweep pattern; `eval_watcher.sh` is its live-poll cousin tailored to lerobot async server.

## Post-training cleanup (mandatory)

**Rule**: once a training run is **evaluated and benchmarked** (logged on leaderboard / model card), prune its `outputs/<run>/` to the minimum needed for reproduction and archival:

- **One dir per model family** (winner + same-family negative archive — keep one failed variant to prove the negative, delete duplicate failures of the same family).
- **3-6 ckpts per kept dir** (the best ckpt by eval + 2-5 neighbors spanning the eval window; for LeRobot-style training keep `last`).
- **Delete** any `*-baseline-v2`, `*-patched`, `*-cont`, `*-phased`, `*-smoke`, `*.<param>-sweep` directory that has a successor/winner; delete wire-debug / bug-residue dirs (ckpts that cannot inference).
- **Never** prune a dir whose training process is still running (`pgrep` first), and never prune the strict-leaderboard winner's best ckpt.

**Why**: 14 retired runs × ~40 GB/run = ~600 GB dead weight. ckpt sweeps inside one run (every 500 step × 40 = 40 ckpts × 2.8 GB) = another ~100 GB per run. Cleaning to family-winner + 3-6 ckpts gave us 1 TB → 196 GB (2026-05-29). Disk runs out → next train SIGSEGV on `torch.save` → looks like a flash-attn crash, actually ENOSPC.

**Trigger**: after a model card / leaderboard row is published (the eval result is now external truth, the intermediate ckpts are no longer load-bearing). Confirm with the user before `rm -rf` — irreversible.

**Frozen-backbone runs — collapse to base+deltas (mandatory, default for EVERY new run)**: if the run froze its backbone (StarVLA frozen-VLM, Wall-X `freeze_vlm`, π0.5 expert-only FT, GR00T frozen-VLM…), the frozen weights are byte-identical in every ckpt → keeping N fulls = (N-1)× dead weight. Run the **committed** universal tool once the run is benchmarked:

```bash
python scripts/ckpt/prune_ckpts.py \
  --fulls 'outputs/<run>/checkpoints/steps_*_pytorch_model.pt' \
  --keep  'outputs/<run>/checkpoints/steps_<best>_pytorch_model.pt' \
  --base  'outputs/_head_sweep_tools/vlm_base_<fam>.pt' \
  --heads 'outputs/<run>/heads'      # dry-run; add --apply to delete non-best fulls
```

It diffs each full against a shared frozen base (not prefix slicing → works for cleanly-prefixed StarVLA *and* interleaved Wall-X/π0.5), GOLD-verifies byte-exact reconstruction before deleting anything, and **auto-refuses on full-FT models** (`frozen_frac < 0.5` → keep-best only). Reconstruct any step with `scripts/ckpt/merge_ckpt.py base delta out`. Keep the best full intact (serve/publish); base+delta reconstruct every other step. Code lives in `scripts/ckpt/` (committed); the big `.pt`/`.safetensors` base+deltas stay in gitignored `outputs/`. **This is lossless even for resume**: the tool only deletes `model.safetensors` (reconstructible byte-exact from base+delta) and never touches `training_state/optimizer_state.safetensors` — resume from any extracted step = merge model + the kept optimizer state. **Keep `optimizer_state.safetensors`** (do NOT delete it for disk — it is the resume料). See `scripts/ckpt/README.md` for the per-family verdict table.

## AutoDL cloud training

When fine-tuning on AutoDL (no-local-GPU mode for setup + GPU mode for training), see [docs/training/autodl_cloud_finetune_playbook.html](docs/training/autodl_cloud_finetune_playbook.html) for: HF gated vs public download paths, `/etc/network_turbo` quirks, single-stream curl recipe for big LFS files, git-lfs prep, `uv sync` + tensorrt-cu12 GPU-mode requirement, 140 GB disk budget with `LossDrivenPruneCallback(top_k=5)`, and a failure playbook.
